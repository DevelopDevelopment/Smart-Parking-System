# Smart Parking System - User Interface
# Customer-facing map showing parking availability

from flask import Flask, render_template, jsonify
import requests
import time
import os

# Get directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
USER_PORT = 8081
DETECTION_SYSTEM_URL = "http://localhost:8080"

app = Flask(__name__, 
            template_folder=os.path.join(SCRIPT_DIR, 'templates'),
            static_folder=os.path.join(SCRIPT_DIR, 'static'))

@app.route('/')
def index():
    # Main map view
    return render_template('map.html')

@app.route('/api/parking_status')
def parking_status():
    # Get parking status from detection system
    try:
        response = requests.get(f'{DETECTION_SYSTEM_URL}/get_parking_spots', timeout=5)
        
        if response.status_code != 200:
            return jsonify({
                'error': 'Could not connect to detection system',
                'total': 0,
                'available': 0,
                'occupied': 0,
                'spots': []
            })
        
        data = response.json()
        spots = data.get('spots', [])
        
        total_spots = len(spots)
        occupied_spots = sum(1 for spot in spots if spot.get('occupied', False))
        available_spots = total_spots - occupied_spots
        
        return jsonify({
            'total': total_spots,
            'available': available_spots,
            'occupied': occupied_spots,
            'spots': spots,
            'last_updated': time.time()
        })
        
    except requests.exceptions.RequestException:
        return jsonify({
            'error': 'Detection system not reachable. Make sure main.py is running on port 8080.',
            'total': 0,
            'available': 0,
            'occupied': 0,
            'spots': []
        })

def main():
    print("=" * 60)
    print("Smart Parking System - User Interface")
    print("=" * 60)
    print(f"\nScript Directory: {SCRIPT_DIR}")
    print(f"User Port: {USER_PORT}")
    print(f"Detection System: {DETECTION_SYSTEM_URL}")
    print("\n" + "=" * 60)
    print("User Interface Starting")
    print("=" * 60)
    print(f"\nOpen: http://localhost:{USER_PORT}")
    print("\nMake sure the main detection system is running on port 8080")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=USER_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()