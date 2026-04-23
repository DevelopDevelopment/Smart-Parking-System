import os
import time
import threading
import secrets
import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# suppress the weights_only warning from newer versions of torch
_orig_load = torch.load
torch.load = lambda f, *a, **kw: _orig_load(f, *a, **{**kw, "weights_only": False})

MODEL_PATH = os.environ.get("YOLO_MODEL", "models/yolo26s.pt")
ALLOWED_MODELS = {
    "yolo26n.pt": "models/yolo26n.pt",
    "yolo26s.pt": "models/yolo26s.pt",
    "yolo26x.pt": "models/yolo26x.pt",
}
# COCO class IDs for cars, motorcycles, buses, trucks
VEHICLE_CLASSES = {2, 3, 5, 7}
SNAPSHOT_INTERVAL = int(os.environ.get("SNAPSHOT_INTERVAL", "15"))
CAMERA_STATUS_INTERVAL = int(os.environ.get("CAMERA_STATUS_INTERVAL", "5"))

# loaded models are cached so we don't load the same weights twice
_model_cache: dict[str, YOLO] = {}

# pre-encoded JPEG frames shared between the detector thread and MJPEG responses
encoded_frame_cache: dict[str, bytes] = {}
encoded_frame_lock = threading.Lock()

# static thumbnails for cameras that aren't running
thumbnail_cache: dict[str, bytes] = {}
thumbnail_lock = threading.Lock()

# DB config is cached for a few seconds to avoid hammering Supabase on every request
_config_cache: dict | None = None
_config_cache_time: float = 0.0
_CONFIG_TTL = 8


# true if the source is something a remote browser can't reach (local file, device index, rtsp, etc.)
def _is_local_source(source: str) -> bool:
    if not source:
        return True
    s = source.strip().lower()
    if s.startswith("/") or s.startswith("./") or s.startswith("../"):
        return True
    # bare integer device IDs e.g. 0, 1, 2
    if s.isdigit():
        return True
    if s.startswith("rtsp://") or s.startswith("rtsps://"):
        return True
    if s.startswith("http://localhost") or s.startswith("https://localhost"):
        return True
    if s.startswith("http://127.") or s.startswith("https://127."):
        return True
    if not (s.startswith("http://") or s.startswith("https://")):
        return True
    return False


# load a YOLO model, or return the cached one if we've already loaded it
def _create_model(model_path: str = MODEL_PATH):
    if model_path in _model_cache:
        return _model_cache[model_path]
    try:
        model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")
        _model_cache[model_path] = model
        return model
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None


# pull all active cameras and their parking spots from Supabase
def _fetch_config_from_db() -> dict:
    cams = (supabase.table("cameras").select("*").eq("is_active", True).execute().data or [])
    if not cams:
        return {"cameras": []}

    cam_ids = [c["id"] for c in cams]
    spots_raw = (supabase.table("camera_parking_spots").select("*").in_("camera_id", cam_ids).execute().data or [])

    # group spots by camera id
    spots_by_cam: dict[str, list] = {cid: [] for cid in cam_ids}
    for s in spots_raw:
        spots_by_cam[s["camera_id"]].append({"id": s["id"], "points": s["points"]})

    cameras_out = []
    for c in cams:
        # generate an API key if the camera doesn't have one yet
        api_key = c.get("api_key")
        if not api_key:
            api_key = secrets.token_urlsafe(24)
            try:
                supabase.table("cameras").update({"api_key": api_key}).eq("id", c["id"]).execute()
            except Exception:
                pass
        cameras_out.append({
            "id": c["id"], "name": c["name"], "source": c["source"],
            "source_type": c.get("source_type", "url"), "crop": c.get("crop"),
            "api_key": api_key,
            "parking_spots": spots_by_cam.get(c["id"], []),
            "settings": c.get("settings") or {
                "confidence_threshold": 0.25, "show_boxes": True,
                "show_parking_spots": True, "fps_limit": 0, "detection_interval": 3,
            },
        })
    return {"cameras": cameras_out}


# returns cached config if fresh, otherwise fetches from DB
def load_config() -> dict:
    global _config_cache, _config_cache_time
    now = time.time()
    if _config_cache is not None and (now - _config_cache_time) < _CONFIG_TTL:
        return _config_cache
    try:
        _config_cache = _fetch_config_from_db()
        _config_cache_time = now
    except Exception as e:
        print(f"[supabase] load_config error: {e}")
        if _config_cache is None:
            _config_cache = {"cameras": []}
    return _config_cache


# force the next load_config call to actually hit the DB
def _invalidate_cache():
    global _config_cache
    _config_cache = None


class ParkingSpot:
    def __init__(self, spot_id, points):
        self.id = spot_id
        self.points = points
        self.occupied = False

    def contains_vehicle(self, bbox, frame_shape):
        x1, y1, x2, y2 = bbox
        return self._pip(((x1 + x2) / 2, (y1 + y2) / 2), self.points)

    # ray-casting point-in-polygon test
    def _pip(self, point, polygon):
        x, y = point
        n, inside = len(polygon), False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xi = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xi:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_bounding_rect(self):
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# runs YOLO detection on a camera stream in a background thread
class VehicleDetector:
    def __init__(self, cam_config):
        self.camera_id = cam_config["id"]
        self.camera_source = cam_config["source"]
        self.parking_spots = [ParkingSpot(s["id"], s["points"]) for s in cam_config.get("parking_spots", [])]
        s = cam_config.get("settings", {})
        self.confidence_threshold = s.get("confidence_threshold", 0.25)
        self.show_boxes = s.get("show_boxes", True)
        self.show_parking_spots = s.get("show_parking_spots", True)
        self.fps_limit = s.get("fps_limit", 0)
        self.detection_interval = max(1, s.get("detection_interval", 3))
        model_name = s.get("model", "yolo26s.pt")
        self.model_path = ALLOWED_MODELS.get(model_name, MODEL_PATH)
        self.last_detection_time = 0
        self.current_frame = self.raw_frame = None
        self.is_running = False
        self.connection_status = "Connecting"
        self.vehicle_count = self.fps = 0
        self._thread = None
        self._frame_lock = threading.Lock()

    # update settings live without restarting the thread
    def update_settings(self, s):
        if "confidence_threshold" in s: self.confidence_threshold = float(s["confidence_threshold"])
        if "show_boxes" in s: self.show_boxes = bool(s["show_boxes"])
        if "show_parking_spots" in s: self.show_parking_spots = bool(s["show_parking_spots"])
        if "fps_limit" in s: self.fps_limit = int(s["fps_limit"])
        if "detection_interval" in s: self.detection_interval = max(1, int(s["detection_interval"]))

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def get_frame(self):
        with self._frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_raw_frame(self):
        with self._frame_lock:
            return self.raw_frame.copy() if self.raw_frame is not None else None

    # pre-encode at thumbnail and stream sizes so serving is fast
    def _encode_frames(self, frame):
        try:
            for size, key, q in [((480, 270), "thumb", 70), ((1280, 720), "feed", 70)]:
                _, jpeg = cv2.imencode(".jpg", cv2.resize(frame, size), [cv2.IMWRITE_JPEG_QUALITY, q])
                with encoded_frame_lock:
                    encoded_frame_cache[f"{self.camera_id}_{key}"] = jpeg.tobytes()
        except Exception as e:
            print(f"Error encoding frames for {self.camera_id}: {e}")

    # the main loop — reads frames, runs YOLO, updates spot states, draws overlays
    def _loop(self):
        model = _create_model(self.model_path)
        if model is None:
            self.connection_status = "ModelLoadError"
            self.is_running = False
            return

        cap = cv2.VideoCapture(self.camera_source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        last_detections = []
        while self.is_running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                self.connection_status = "Offline"
                time.sleep(2)
                cap = cv2.VideoCapture(self.camera_source)
                continue

            raw = frame.copy()
            self.connection_status = "DETECTING"

            # only run inference every N seconds to save CPU
            if t0 - self.last_detection_time >= self.detection_interval:
                h, w = frame.shape[:2]
                scale = min(1.0, 640.0 / max(w, h))
                small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
                results = model(small, verbose=False, conf=self.confidence_threshold)
                last_detections = []
                if results:
                    for result in results:
                        boxes = getattr(result, "boxes", None)
                        if boxes is None:
                            continue
                        for box in boxes:
                            if int(box.cls[0]) not in VEHICLE_CLASSES:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # scale coords back up to original resolution
                            last_detections.append((x1/scale, y1/scale, x2/scale, y2/scale))
                self.last_detection_time = t0

            self.vehicle_count = len(last_detections)

            # check each spot and draw overlays
            for spot in self.parking_spots:
                spot.occupied = any(spot.contains_vehicle(d, frame.shape) for d in last_detections)
                if self.show_parking_spots:
                    color = (0, 0, 255) if spot.occupied else (0, 255, 0)
                    rx1, ry1, rx2, ry2 = spot.get_bounding_rect()
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 3)

            if self.show_boxes:
                for d in last_detections:
                    cv2.rectangle(frame, tuple(map(int, d[:2])), tuple(map(int, d[2:])), (255, 165, 0), 2)

            elapsed = time.time() - t0
            self.fps = round(1.0 / max(elapsed, 0.001), 1)

            with self._frame_lock:
                self.current_frame = frame
                self.raw_frame = raw

            self._encode_frames(frame)

            if self.fps_limit > 0:
                remaining = (1.0 / self.fps_limit) - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            else:
                time.sleep(0.005)

        cap.release()


# keeps track of which cameras are running
detectors: dict[str, VehicleDetector] = {}
detectors_lock = threading.Lock()


def start_camera(cam_id: str) -> bool:
    cam = next((c for c in load_config()["cameras"] if c["id"] == cam_id), None)
    if not cam:
        return False
    with detectors_lock:
        if cam_id in detectors:
            return False
        det = VehicleDetector(cam)
        det.start()
        detectors[cam_id] = det
    return True


def stop_camera(cam_id: str) -> bool:
    with detectors_lock:
        det = detectors.pop(cam_id, None)
    if det:
        det.stop()
        return True
    return False


def get_detector(cam_id: str) -> VehicleDetector | None:
    with detectors_lock:
        return detectors.get(cam_id)


def active_camera_ids() -> list[str]:
    with detectors_lock:
        return list(detectors.keys())


# grabs a single frame from a source in a background thread and stashes it in the thumbnail cache
def capture_thumbnail(source: str, cam_id: str):
    def _grab():
        cap = cv2.VideoCapture(source)
        frame = None
        for _ in range(8):
            ret, f = cap.read()
            if ret:
                frame = f
                break
            time.sleep(0.4)
        cap.release()
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", cv2.resize(frame, (480, 270)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            with thumbnail_lock:
                thumbnail_cache[cam_id] = jpeg.tobytes()
    threading.Thread(target=_grab, daemon=True).start()


def _spot_stats(det: VehicleDetector):
    total = len(det.parking_spots)
    occ = sum(1 for s in det.parking_spots if s.occupied)
    return total, occ, total - occ


# returns the first active detector — used by the old single-camera setup endpoints
def _active_det():
    ids = active_camera_ids()
    return get_detector(ids[0]) if ids else None


app = Flask(__name__)
CORS(app)

# start any cameras that were already marked active in the DB when the server boots
for _cam in load_config().get("cameras", []):
    try:
        start_camera(_cam["id"])
    except Exception as e:
        print(f"[boot] Could not start camera {_cam['id']}: {e}")


# encode a frame and upload it to Supabase Storage as a preview image
def _upload_camera_preview(camera_id: str, frame) -> str | None:
    try:
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        path = f"{camera_id}.jpg"
        supabase.storage.from_("camera-previews").upload(
            path,
            jpeg.tobytes(),
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        url = f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/camera-previews/{path}"
        return url
    except Exception as e:
        print(f"[preview_upload] {camera_id}: {e}")
        return None


# background thread: writes occupancy snapshots and preview images to Supabase every N seconds
def _snapshot_writer_loop():
    while True:
        time.sleep(SNAPSHOT_INTERVAL)
        try:
            cfg = load_config()
            cam_map = {c["id"]: c for c in cfg.get("cameras", [])}
            cam_ids = list(cam_map.keys())
            if not cam_ids:
                continue

            # only write snapshots for cameras that are linked to a carpark
            linked = {
                row["id"]: row["carpark_id"]
                for row in (supabase.table("cameras").select("id, carpark_id")
                            .in_("id", cam_ids).not_.is_("carpark_id", "null").execute().data or [])
            }

            rows = []
            for cam_id, carpark_id in linked.items():
                det = get_detector(cam_id)
                online = det is not None and det.connection_status not in ("Offline", "Inactive", "ModelLoadError")
                if det and online:
                    total, occ, avail = _spot_stats(det)
                else:
                    total = len(cam_map.get(cam_id, {}).get("parking_spots", []))
                    occ, avail = 0, total
                rows.append({"carpark_id": carpark_id, "total": total, "available": avail,
                             "occupied": occ, "is_online": online})

            if rows:
                supabase.table("parking_snapshots").insert(rows).execute()

            # upload a still preview for each online camera
            for cam_id in linked:
                det = get_detector(cam_id)
                if det is None:
                    continue
                frame = det.get_raw_frame()
                if frame is None:
                    continue
                url = _upload_camera_preview(cam_id, frame)
                if url:
                    supabase.table("camera_status").update(
                        {"preview_url": url}
                    ).eq("camera_id", cam_id).execute()
        except Exception as e:
            print(f"[snapshot_writer] Error: {e}")


# background thread: upserts live camera state to camera_status so the frontend can subscribe via Realtime
def _camera_status_writer_loop():
    while True:
        time.sleep(CAMERA_STATUS_INTERVAL)
        try:
            cfg = load_config()
            rows = []
            for cam in cfg.get("cameras", []):
                cam_id = cam["id"]
                source = cam.get("source", "")
                local = _is_local_source(source)
                det = get_detector(cam_id)
                if det:
                    online = det.connection_status not in ("Offline", "Inactive", "ModelLoadError")
                    total, occ, avail = _spot_stats(det)
                    rows.append({
                        "camera_id": cam_id,
                        "name": cam["name"],
                        "is_online": online,
                        "status": det.connection_status,
                        "fps": det.fps,
                        "total_spots": total,
                        "occupied": occ,
                        "available": avail,
                        "is_local": local,
                        "updated_at": "now()",
                    })
                else:
                    total = len(cam.get("parking_spots", []))
                    rows.append({
                        "camera_id": cam_id,
                        "name": cam["name"],
                        "is_online": False,
                        "status": "Inactive",
                        "fps": 0,
                        "total_spots": total,
                        "occupied": 0,
                        "available": total,
                        "is_local": local,
                        "updated_at": "now()",
                    })
            if rows:
                supabase.table("camera_status").upsert(rows, on_conflict="camera_id").execute()
        except Exception as e:
            print(f"[camera_status_writer] Error: {e}")


threading.Thread(target=_snapshot_writer_loop, daemon=True, name="snapshot-writer").start()
threading.Thread(target=_camera_status_writer_loop, daemon=True, name="camera-status-writer").start()


def _mjpeg_frame(jpeg_bytes: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"


def _encode_jpeg(frame, size=(1280, 720), quality=80) -> bytes:
    _, jpeg = cv2.imencode(".jpg", cv2.resize(frame, size), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


# yields MJPEG frames forever; falls back to reading the source directly if the detector isn't running
def _video_generator(cam_id: str, use_raw: bool = False):
    cap = None
    cfg_cam = next((c for c in load_config()["cameras"] if c["id"] == cam_id), None)
    while True:
        det = get_detector(cam_id)
        if det:
            if not use_raw:
                with encoded_frame_lock:
                    cached = encoded_frame_cache.get(f"{cam_id}_feed")
                if cached:
                    yield _mjpeg_frame(cached)
                    time.sleep(0.04)
                    continue
            frame = det.get_raw_frame() if use_raw else det.get_frame()
            if frame is not None:
                yield _mjpeg_frame(_encode_jpeg(frame))
        elif cfg_cam:
            # detector isn't running, open the source directly so the feed still works
            if cap is None:
                cap = cv2.VideoCapture(cfg_cam["source"])
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            ret, frame = cap.read()
            if ret and frame is not None:
                yield _mjpeg_frame(_encode_jpeg(frame))
            else:
                if cap:
                    cap.release()
                cap = None
                time.sleep(2)
        else:
            break
        time.sleep(0.04)
    if cap:
        cap.release()


@app.route("/")
def index():
    cfg = load_config()
    api_keys = {c["id"]: c.get("api_key", "") for c in cfg["cameras"]}
    return render_template("index.html", cameras=cfg["cameras"], active_cameras=active_camera_ids(), api_keys=api_keys)


@app.route("/setup")
def setup():
    cam_id = request.args.get("cam", "")
    return render_template("setup.html", cam_id=cam_id)


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    return Response(_video_generator(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/raw_feed/<cam_id>")
def raw_feed(cam_id):
    return Response(_video_generator(cam_id, use_raw=True), mimetype="multipart/x-mixed-replace; boundary=frame")


# kept for backwards compatibility, uses the first active camera
@app.route("/video_feed")
def video_feed_old():
    ids = active_camera_ids()
    cam_id = ids[0] if ids else None
    if not cam_id:
        return Response(b"", mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(_video_generator(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/toggle_camera", methods=["POST"])
def toggle_camera():
    cam_id = request.json.get("camera_id")
    if not cam_id:
        return jsonify({"success": False, "error": "Missing camera_id"})
    if cam_id in active_camera_ids():
        stop_camera(cam_id)
        with thumbnail_lock:
            thumbnail_cache.pop(cam_id, None)
        return jsonify({"success": True, "active": False, "active_cameras": active_camera_ids()})
    ok = start_camera(cam_id)
    if not ok:
        return jsonify({"success": False, "error": "Camera not found in config"})
    return jsonify({"success": True, "active": True, "active_cameras": active_camera_ids()})


@app.route("/api/cameras", methods=["GET"])
def get_cameras_api():
    return jsonify(load_config()["cameras"])


@app.route("/api/update_camera_settings", methods=["POST"])
def update_camera_settings():
    data = request.json
    cam_id = data.get("camera_id")
    new_settings = data.get("settings", {})
    if not cam_id:
        return jsonify({"success": False, "error": "Missing camera_id"})
    cam = next((c for c in load_config()["cameras"] if c["id"] == cam_id), None)
    if not cam:
        return jsonify({"success": False, "error": "Camera not found"})
    if "model" in new_settings and new_settings["model"] not in ALLOWED_MODELS:
        return jsonify({"success": False, "error": "Invalid model"}), 400
    model_changed = "model" in new_settings and new_settings["model"] != cam.get("settings", {}).get("model", "yolo26s.pt")
    try:
        supabase.table("cameras").update({"settings": {**cam.get("settings", {}), **new_settings}}).eq("id", cam_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    _invalidate_cache()
    det = get_detector(cam_id)
    if det:
        if model_changed:
            # model swap needs a full restart to load the new weights because it will just break
            stop_camera(cam_id)
            start_camera(cam_id)
        else:
            det.update_settings(new_settings)
    return jsonify({"success": True, "restarted": model_changed})


@app.route("/api/add_camera", methods=["POST"])
def add_camera_api():
    data = request.json or {}
    cam_id, name, source = data.get("id"), data.get("name"), data.get("source")
    if not all([cam_id, name, source]):
        return jsonify({"success": False, "error": "Missing id/name/source"}), 400
    if any(c["id"] == cam_id for c in load_config().get("cameras", [])):
        return jsonify({"success": False, "error": "Camera ID already exists"}), 400
    try:
        supabase.table("cameras").insert({
            "id": cam_id, "name": name, "source": source,
            "source_type": data.get("source_type", "url"), "crop": data.get("crop"),
            "api_key": secrets.token_urlsafe(24),
            "settings": {"confidence_threshold": 0.25, "show_boxes": True,
                         "show_parking_spots": True, "fps_limit": 0, "detection_interval": 2},
        }).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    _invalidate_cache()
    return jsonify({"success": True})


@app.route("/api/delete_camera", methods=["POST"])
def delete_camera_api():
    cam_id = (request.json or {}).get("camera_id")
    if not cam_id:
        return jsonify({"success": False, "error": "Missing camera_id"}), 400
    if not any(c["id"] == cam_id for c in load_config().get("cameras", [])):
        return jsonify({"success": False, "error": "Camera not found"}), 404
    stop_camera(cam_id)
    try:
        supabase.table("cameras").delete().eq("id", cam_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    _invalidate_cache()
    return jsonify({"success": True})


@app.route("/api/regenerate_api_key", methods=["POST"])
def regenerate_api_key():
    cam_id = (request.json or {}).get("camera_id")
    if not cam_id:
        return jsonify({"success": False, "error": "Missing camera_id"}), 400
    new_key = secrets.token_urlsafe(24)
    try:
        supabase.table("cameras").update({"api_key": new_key}).eq("id", cam_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    _invalidate_cache()
    return jsonify({"success": True, "api_key": new_key})


@app.route("/api/list_local_cameras")
def list_local_cameras():
    index = 0
    available_indices = []
    # Check first 8 indices for cameras
    while index < 8:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_indices.append(index)
            cap.release()
        index += 1
    return jsonify({"success": True, "cameras": available_indices})

# link a camera to a carpark so its occupancy shows up in snapshots
@app.route("/api/link_camera", methods=["POST"])
def link_camera_api():
    data = request.json or {}
    cam_id = data.get("camera_id")
    if not cam_id:
        return jsonify({"success": False, "error": "Missing camera_id"}), 400
    try:
        supabase.table("cameras").update({"carpark_id": data.get("carpark_id")}).eq("id", cam_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    _invalidate_cache()
    return jsonify({"success": True})


# serves a 480x270 thumbnail; returns 202 if we don't have one yet so the client can retry later
@app.route("/api/camera_thumbnail/<cam_id>")
def camera_thumbnail(cam_id):
    with encoded_frame_lock:
        cached = encoded_frame_cache.get(f"{cam_id}_thumb")
    if cached:
        return Response(cached, mimetype="image/jpeg")
    det = get_detector(cam_id)
    if det:
        frame = det.get_frame()
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", cv2.resize(frame, (480, 270)), [cv2.IMWRITE_JPEG_QUALITY, 82])
            return Response(jpeg.tobytes(), mimetype="image/jpeg")
        return Response(status=202)
    with thumbnail_lock:
        cached = thumbnail_cache.get(cam_id)
    if cached:
        return Response(cached, mimetype="image/jpeg")
    cam = next((c for c in load_config()["cameras"] if c["id"] == cam_id), None)
    if cam:
        capture_thumbnail(cam["source"], cam_id)
        return Response(status=202)
    return "Camera not found", 404


# bust the thumbnail cache and trigger a fresh capture
@app.route("/api/refresh_thumbnail/<cam_id>", methods=["POST"])
def refresh_thumbnail(cam_id):
    with thumbnail_lock:
        thumbnail_cache.pop(cam_id, None)
    cam = next((c for c in load_config()["cameras"] if c["id"] == cam_id), None)
    if cam:
        if not get_detector(cam_id):
            capture_thumbnail(cam["source"], cam_id)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Camera not found"})


# should be polled every 2s by the dashboard to update occupancy counts
@app.route("/api/camera_stats")
def camera_stats():
    stats = []
    for cam in load_config()["cameras"]:
        det = get_detector(cam["id"])
        if det:
            total, occ, avail = _spot_stats(det)
            stats.append({"id": cam["id"], "active": True, "total": total, "occupied": occ,
                          "available": avail, "fps": det.fps, "status": det.connection_status})
        else:
            total = len(cam.get("parking_spots", []))
            stats.append({"id": cam["id"], "active": False, "total": total,
                          "occupied": None, "available": None, "fps": 0, "status": "Inactive"})
    return jsonify(stats)


# public status endpoint for a single camera, requires a valid API key ofc
@app.route("/api/cameras/<cam_id>/status")
def camera_status_api(cam_id):
    cam = next((c for c in load_config()["cameras"] if c["id"] == cam_id), None)
    if not cam:
        return jsonify({"error": "Camera not found"}), 404
    provided_key = request.args.get("key") or request.headers.get("X-API-Key")
    if not provided_key or provided_key != cam.get("api_key"):
        return jsonify({"error": "Unauthorized. Provide a valid API key via ?key= or X-API-Key header."}), 401
    stream_url = f"{request.url_root.rstrip('/')}/video_feed/{cam_id}"
    det = get_detector(cam_id)
    if det:
        total, occ, avail = _spot_stats(det)
        return jsonify({
            "id": cam_id, "name": cam["name"], "total": total, "occupied": occ,
            "available": avail, "fps": det.fps, "connection_status": det.connection_status,
            "stream_url": stream_url,
            "is_local": _is_local_source(cam.get("source", "")),
            "spots": [{"id": s.id, "occupied": s.occupied, "points": s.points} for s in det.parking_spots],
        })
    total = len(cam.get("parking_spots", []))
    return jsonify({
        "id": cam_id, "name": cam["name"], "total": total, "occupied": 0, "available": total,
        "fps": 0, "connection_status": "inactive", "stream_url": stream_url,
        "is_local": _is_local_source(cam.get("source", "")),
        "spots": [{"id": s["id"], "occupied": False, "points": s["points"]} for s in cam.get("parking_spots", [])],
    })


# setup page endpoints — all accept ?cam=<camera_id> or cam_id in the request body
def _setup_det():
    cam_id = request.args.get("cam") or (request.json or {}).get("cam_id")
    if cam_id:
        det = get_detector(cam_id)
        if det:
            return det
    return _active_det()


@app.route("/get_parking_spots")
def get_spots():
    det = _setup_det()
    if not det:
        return jsonify({"total": 0, "occupied": 0, "available": 0, "fps": 0,
                        "connection_status": "Inactive", "spots": []})
    total, occ, avail = _spot_stats(det)
    return jsonify({"total": total, "occupied": occ, "available": avail, "fps": det.fps,
                    "connection_status": det.connection_status,
                    "spots": [{"id": s.id, "points": s.points} for s in det.parking_spots]})


@app.route("/add_parking_spot", methods=["POST"])
def add_spot():
    det = _setup_det()
    if not det:
        return jsonify({"success": False, "error": "No active camera"})
    points = [[int(p["x"]), int(p["y"])] for p in request.json["points"]]
    try:
        resp = supabase.table("camera_parking_spots").insert({"camera_id": det.camera_id, "points": points}).execute()
        new_id = resp.data[0]["id"]
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    det.parking_spots.append(ParkingSpot(new_id, points))
    _invalidate_cache()
    return jsonify({"success": True, "spot_id": new_id})


@app.route("/update_parking_spot", methods=["POST"])
def update_spot():
    det = _setup_det()
    if not det:
        return jsonify({"success": False, "error": "No active camera"})
    data = request.json
    spot_id = data["spot_id"]
    points = [[int(p["x"]), int(p["y"])] for p in data["points"]]
    target = next((s for s in det.parking_spots if s.id == spot_id), None)
    if not target:
        return jsonify({"success": False, "error": "Spot not found"})
    try:
        supabase.table("camera_parking_spots").update({"points": points}).eq("id", spot_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    target.points = points
    _invalidate_cache()
    return jsonify({"success": True})


@app.route("/delete_parking_spot", methods=["POST"])
def delete_spot():
    det = _setup_det()
    if not det:
        return jsonify({"success": False, "error": "No active camera"})
    spot_id = request.json["spot_id"]
    before = len(det.parking_spots)
    det.parking_spots = [s for s in det.parking_spots if s.id != spot_id]
    if len(det.parking_spots) == before:
        return jsonify({"success": False, "error": "Spot not found"})
    try:
        supabase.table("camera_parking_spots").delete().eq("id", spot_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    _invalidate_cache()
    return jsonify({"success": True})


@app.route("/clear_parking_spots", methods=["POST"])
def clear_spots():
    det = _setup_det()
    if not det:
        return jsonify({"success": False, "error": "No active camera"})
    try:
        supabase.table("camera_parking_spots").delete().eq("camera_id", det.camera_id).execute()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    det.parking_spots = []
    _invalidate_cache()
    return jsonify({"success": True})


@app.route("/get_frame_for_setup")
def get_frame_setup():
    det = _setup_det()
    if det:
        frame = det.get_frame()
        if frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame)
            return Response(jpeg.tobytes(), mimetype="image/jpeg")
    return "No frame", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8070))
    print(f"\n{'='*50}\n  SmartParking Detection Server\n{'='*50}\n  http://localhost:{port}\n{'='*50}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)