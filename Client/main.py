
from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import torch
import os

# Get directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_original_torch_load = torch.load

def _force_weights_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

torch.load = _force_weights_load

# Import YOLO
from ultralytics import YOLO

# Patch ultralytics for compatibility 
try:
    from ultralytics.nn import tasks
    if hasattr(tasks, 'torch_safe_load'):
        def _patched_torch_safe_load(file, *args, **kwargs):
            return torch.load(file, map_location='cpu', weights_only=False), file
        tasks.torch_safe_load = _patched_torch_safe_load
except Exception as e:
    print(f"Patch note: {e}")


WEBCAM_STREAM_URL = "http://127.0.0.1:5000/video/2"
MODEL_SIZE = 'n'
CONFIDENCE_THRESHOLD = 0.5
WEB_PORT = 8080
RESIZE_WIDTH = 480
DETECTION_SKIP = 2

class ParkingSpot:
    # Represents a parking spot defined by 4 corner points
    
    def __init__(self, spot_id, points):
        self.id = spot_id
        self.points = points
        self.occupied = False
        self.last_check = None
        
    def contains_vehicle(self, vehicle_bbox):
        # Check if vehicle center is inside parking spot polygon
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return self._point_in_polygon(vehicle_center, self.points)
    
    def _point_in_polygon(self, point, polygon):
        # Ray casting algorithm for point-in-polygon test
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def draw(self, frame):
        # Draw parking spot polygon on frame
        points_array = np.array(self.points, dtype=np.int32)
    
        # Color based on occupancy
        color = (0, 0, 255) if self.occupied else (0, 255, 0)
        
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points_array], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw border
        cv2.polylines(frame, [points_array], True, color, 2)
        
        # Draw label
        center_x = int(sum(p[0] for p in self.points) / 4)
        center_y = int(sum(p[1] for p in self.points) / 4)
        
        status = "OCCUPIED" if self.occupied else "FREE"
        label = f"Spot {self.id}: {status}"
        
        cv2.putText(frame, label, (center_x - 50, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

class VehicleDetector:
    # Main vehicle detection class using YOLO
    
    def __init__(self):
        print(f"Loading YOLOv8{MODEL_SIZE} model...")
        
        # Build absolute path to model file
        model_path = os.path.join(SCRIPT_DIR, f'yolov8{MODEL_SIZE}-seg.pt')
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
        
        self.model = YOLO(model_path)
        print("Model loaded")
        
        # COCO dataset vehicle class IDs
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        self.parking_spots = []
        self.fps = 0
        self.vehicle_count = 0
        self.frame = None
        self.is_running = False
        
        # Camera configuration
        self.camera_url = WEBCAM_STREAM_URL
        self.detection_thread = None
        self.connection_status = "Not connnected"
        
    def set_camera_url(self, url):
        # Change camera and restart detection
        print(f"Changing camera to: {url}")
        self.camera_url = url
        
        if self.is_running:
            self.stop_detection()
            time.sleep(1)
        
        self.start_detection()
        
    def stop_detection(self):
        # Stop detection loop
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        
    def detect_vehicles(self, frame):
        # Run YOLO detection on frame
        results = self.model(frame, verbose=False, stream=False, imgsz=640, half=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                if class_id in self.vehicle_classes and confidence > CONFIDENCE_THRESHOLD:
                    vehicle_type = self.vehicle_classes[class_id]
                    detections.append({
                        'type': vehicle_type,
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
                    
                    # Draw bounding box
                    color = self._get_color(vehicle_type)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Draw label
                    label = f"{vehicle_type} {confidence:.2%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.7, 2)
                    
                    cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10),
                                (int(x1) + text_width, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                              font, 0.7, (255, 255, 255), 2)
        
        return frame, detections
    
    def update_parking_spots(self, detections):
        # Update parking spot occupancy based on detections
        for spot in self.parking_spots:
            spot.occupied = False
        
        for detection in detections:
            vehicle_bbox = detection.get('bbox')
            if vehicle_bbox:
                for spot in self.parking_spots:
                    if spot.contains_vehicle(vehicle_bbox):
                        spot.occupied = True
    
    def draw_parking_spots(self, frame):
        # Draw all parking spots
        for spot in self.parking_spots:
            frame = spot.draw(frame)
        return frame
    
    def add_parking_spot(self, points):
        # Add new parking spot
        spot_id = len(self.parking_spots) + 1
        spot = ParkingSpot(spot_id, points)
        self.parking_spots.append(spot)
        print(f"Added parking spot {spot_id}")
        return spot
    
    def _get_color(self, vehicle_type):
        # Get color for vehicle type
        colors = {
            'car': (0, 255, 0),
            'motorcycle': (255, 0, 0),
            'bus': (0, 165, 255),
            'truck': (0, 0, 255)
        }
        return colors.get(vehicle_type, (255, 255, 255))
    
    def add_stats_overlay(self, frame, vehicle_count, fps):
        # Add statistics overlay to frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        stats = [
            f"FPS: {fps:.1f}",
            f"Vehicles: {vehicle_count}",
            f"Status: {self.connection_status}",
            f"Model: YOLOv8{MODEL_SIZE}",
            f"Camera: ...{self.camera_url[-25:]}" if len(self.camera_url) > 25 else f"Camera: {self.camera_url}"
        ]
        
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset), font, 0.6, (0, 255, 0), 2)
            y_offset += 30
        
        return frame
    
    def start_detection(self):
        # Start detection thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print(f"Connecting to: {self.camera_url}")
    
    def _detection_loop(self):
        # Main detection loop
        self.connection_status = "Connecting..."
        cap = cv2.VideoCapture(self.camera_url)
        
        if not cap.isOpened():
            print(f"ERROR: Could not connect to {self.camera_url}")
            self.connection_status = "Connection Failed"
            self.is_running = False
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Connected")
        self.connection_status = "DETECTING"
        
        frame_count = 0
        last_detections = []
        detection_start_time = time.time()
        
        # Main the beep boop
        while self.is_running:
            ret, frame = cap.read()
            
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Resize frame for speed
            if RESIZE_WIDTH and frame.shape[1] != RESIZE_WIDTH:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(RESIZE_WIDTH * aspect_ratio)
                frame = cv2.resize(frame, (RESIZE_WIDTH, new_height))
            
            # Run detection every N frames
            if frame_count % DETECTION_SKIP == 0:
                detection_start_time = time.time()
                
                frame, detections = self.detect_vehicles(frame)
                last_detections = detections
                self.vehicle_count = len(detections)
                
                self.update_parking_spots(detections)
                
                detection_end_time = time.time()
                detection_time = detection_end_time - detection_start_time
                self.fps = 1 / detection_time if detection_time > 0 else 0
            else:
                # Draw previous detections
                for detection in last_detections:
                    bbox = detection['bbox']
                    vehicle_type = detection['type']
                    confidence = detection['confidence']
                    
                    x1, y1, x2, y2 = bbox
                    color = self._get_color(vehicle_type)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{vehicle_type} {confidence:.2%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.7, 2)
                    
                    cv2.rectangle(frame, (x1, y1 - text_height - 10),
                                (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              font, 0.7, (255, 255, 255), 2)
                
                self.vehicle_count = len(last_detections)
            
            frame = self.draw_parking_spots(frame)
            frame = self.add_stats_overlay(frame, self.vehicle_count, self.fps)
            self.frame = frame
        
        cap.release()
    
    def get_frame(self):
        return self.frame

# Basic Flask application
app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'),
            static_folder=os.path.join(SCRIPT_DIR, 'static'))
detector = VehicleDetector()

@app.route('/')
def index():
    return render_template('index.html', 
                         stream_url=detector.camera_url,
                         model_size=MODEL_SIZE, 
                         confidence=int(CONFIDENCE_THRESHOLD * 100))

@app.route('/set_camera', methods=['POST'])
def set_camera():
    # Update camera URL
    data = request.get_json()
    camera_url = data.get('camera_url', '').strip()
    
    if not camera_url:
        return jsonify({'success': False, 'error': 'Camera URL is required'})
    
    try:
        detector.set_camera_url(camera_url)
        return jsonify({'success': True, 'camera_url': camera_url})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/setup')
def setup():
    return render_template('setup.html')

@app.route('/get_frame_for_setup')
def get_frame_for_setup():
    # Get single frame for setup page
    frame = detector.get_frame()
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = blank
    
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    
    return Response(frame_bytes, mimetype='image/jpeg')

@app.route('/add_parking_spot', methods=['POST'])
def add_parking_spot():
    # Add parking spot from setup page
    data = request.get_json()
    points = data.get('points')
    
    if points and len(points) == 4:
        points_tuples = [(int(p['x']), int(p['y'])) for p in points]
        detector.add_parking_spot(points_tuples)
        return {'success': True, 'spot_id': len(detector.parking_spots)}
    
    return {'success': False, 'error': 'Need exactly 4 points'}

@app.route('/get_parking_spots')
def get_parking_spots():
    # Get all parking spots
    spots = []
    for spot in detector.parking_spots:
        spots.append({
            'id': spot.id,
            'points': spot.points,
            'occupied': spot.occupied
        })
    return {'spots': spots}

@app.route('/clear_parking_spots', methods=['POST'])
def clear_parking_spots():
    # Clear all parking spots
    detector.parking_spots = []
    return {'success': True}

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = detector.get_frame()
            
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Connecting...", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = blank
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    print("=" * 60)
    print("Smart Parking System - Vehicle Detection")
    print("=" * 60)
    print(f"\nScript Directory: {SCRIPT_DIR}")
    print(f"Default Camera: {WEBCAM_STREAM_URL}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Port: {WEB_PORT}\n")
    
    detector.start_detection()
    time.sleep(2)
    
    print("=" * 60)
    print("System Starting")
    print("=" * 60)
    print(f"\nOpen: http://localhost:{WEB_PORT}")
    print("\nYou can change the camera URL from the web interface")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        detector.stop_detection()

if __name__ == "__main__":
    main()