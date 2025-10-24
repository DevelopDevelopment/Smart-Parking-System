import cv2
import socket
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        # Dictionary to store camera information (id: camera_info)
        self.cameras = {}
        # Dictionary to store active camera capture objects (id: VideoCapture)
        self.streaming_cameras = {}
        
    def scan_cameras(self, max_cameras=10):
        available_cameras = {}
        
        # Try to open each camera index
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to read a frame to verify camera actually works
                    ret, frame = cap.read()
                    if ret:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        # Store camera information
                        camera_info = {
                            'id': i,
                            'name': f'Camera {i}',
                            'resolution': f'{width}x{height}',
                            'fps': fps,
                            'status': 'available'
                        }
                        available_cameras[i] = camera_info
                    
                # Always release the camera after checking
                cap.release()
                    
            except Exception as e:
                pass
                
        self.cameras = available_cameras
        return available_cameras
    
    def start_streaming(self, camera_id):
        # If already streaming, return success
        if camera_id in self.streaming_cameras:
            return True
            
        try:
            # Open the camera
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Add to streaming cameras dictionary
                self.streaming_cameras[camera_id] = cap
                # Update status in cameras dictionary
                if camera_id in self.cameras:
                    self.cameras[camera_id]['status'] = 'streaming'
                return True
        except Exception as e:
            logger.error(f"Failed to start streaming camera {camera_id}: {e}")
            
        return False
    
    def stop_streaming(self, camera_id):
        if camera_id in self.streaming_cameras:
            try:
                # Get the camera capture object
                cap = self.streaming_cameras[camera_id]
                cap.release()
                # Remove from streaming dictionary
                del self.streaming_cameras[camera_id]
                
                # Update status in cameras dictionary
                if camera_id in self.cameras:
                    self.cameras[camera_id]['status'] = 'available'
                return True
            except Exception as e:
                logger.error(f"Error stopping camera {camera_id}: {e}")
                return False
        return True  # Return True even if not streaming 
    
    def generate_frames(self, camera_id):
        # Check if camera is in streaming dictionary
        if camera_id not in self.streaming_cameras:
            return
            
        cap = self.streaming_cameras[camera_id]
        
        # Continuously capture and yield frames
        while camera_id in self.streaming_cameras:
            try:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Encode frame as JPEG with 85% quality
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    # Yield frame in multipart format for streaming
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                           
            except Exception as e:
                logger.error(f"Error generating frames for camera {camera_id}: {e}")
                break
                
        # Cleanup if streaming stopped
        self.stop_streaming(camera_id)

camera_manager = CameraManager()

# Setting up Flask app
app = Flask(__name__)

CORS(app)

def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except:
        return "127.0.0.1"

# Flask Stuff

@app.route('/')
def index():
    local_ip = get_local_ip()
    return render_template('index.html', server_ip=local_ip, server_port=5000)

@app.route('/api/scan')
def api_scan():
    cameras = camera_manager.scan_cameras()
    return jsonify({
        'success': True,
        'cameras': cameras,
        'count': len(cameras)
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        'success': True,
        'cameras': camera_manager.cameras
    })

@app.route('/api/stream/start/<int:camera_id>', methods=['POST'])
def api_start_stream(camera_id):
    success = camera_manager.start_streaming(camera_id)
    if success:
        return jsonify({
            'success': True,
            'message': f'Started streaming camera {camera_id}',
            'stream_url': f'http://{get_local_ip()}:5000/video/{camera_id}'
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Failed to start streaming camera {camera_id}'
        }), 400

@app.route('/api/stream/stop/<int:camera_id>', methods=['POST'])
def api_stop_stream(camera_id):
    success = camera_manager.stop_streaming(camera_id)
    return jsonify({
        'success': True,
        'message': f'Camera {camera_id} stopped'
    })

@app.route('/video/<int:camera_id>')
def video_feed(camera_id):
    if camera_id not in camera_manager.streaming_cameras:
        # Try to start streaming if not already started
        if not camera_manager.start_streaming(camera_id):
            return "Camera not available", 404
    
    return Response(
        camera_manager.generate_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # Print server information on startup
    print(f"Server running at http://{get_local_ip()}:5000")
    
    try:
        # host='0.0.0.0' allows access from other devices 
        # threaded=True enables improves perforamce
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        # Clean shutdown on quit
        for camera_id in list(camera_manager.streaming_cameras.keys()):
            camera_manager.stop_streaming(camera_id)