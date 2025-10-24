// Camera data storage
let cameras = {};
// These will be set from the HTML template
let serverIP = '';
let serverPort = '';

// Scan for available USB cameras
function scanCameras() {
    document.getElementById('camera-list').innerHTML = '';
    
    fetch('/api/scan')
        .then(response => response.json())
        .then(data => {
            cameras = data.cameras;
            displayCameras();
        })
        .catch(error => console.error('Scan error:', error));
}

// Display all cameras in the UI
function displayCameras() {
    const cameraList = document.getElementById('camera-list');
    
    if (Object.keys(cameras).length === 0) {
        cameraList.innerHTML = '<div class="no-cameras">No cameras found. Make sure your USB cameras are connected and try using the camera app on Windows to verify.</div>';
        return;
    }

    cameraList.innerHTML = '';
    
    Object.values(cameras).forEach(camera => {
        const cameraCard = document.createElement('div');
        cameraCard.className = 'camera-card';
        cameraCard.innerHTML = `
            <div class="camera-info">
                <div class="camera-name">${camera.name}</div>
                <div class="camera-details">Resolution: ${camera.resolution}</div>
                <div class="camera-details">FPS: ${camera.fps}</div>
                <div class="status ${camera.status}">${camera.status.toUpperCase()}</div>
            </div>
            
            <div class="stream-controls">
                <button class="stream-btn" onclick="startStream(${camera.id})" 
                        ${camera.status === 'streaming' ? 'style="display:none"' : ''}>
                    Start Stream
                </button>
                <button class="stream-btn stop" onclick="stopStream(${camera.id})"
                        ${camera.status !== 'streaming' ? 'style="display:none"' : ''}>
                    Stop Stream
                </button>
            </div>
            
            ${camera.status === 'streaming' ? `
            <div class="stream-info">
                LIVE STREAM 
                <a href="#" onclick="copyToClipboard('http://${serverIP}:${serverPort}/video/${camera.id}', this); return false;" class="copy-link">Copy Camera IP</a>
                View: <a href="/video/${camera.id}" target="_blank">Open Stream</a>
            </div>
            ` : ''}
        `;
        cameraList.appendChild(cameraCard);
    });
}

// Copy camera IP URL to clipboard
function copyToClipboard(text, linkElement) {
    navigator.clipboard.writeText(text).then(() => {
        showCopySuccess(linkElement, 'Copied!');
    }).catch(err => {
        showCopyError(linkElement, 'Copy failed');
    });
}

// Show success message after copying
function showCopySuccess(linkElement, message) {
    const originalText = linkElement.textContent;
    linkElement.textContent = message;
    linkElement.style.color = '#10b981';
    linkElement.style.fontWeight = 'bold';
    
    setTimeout(() => {
        linkElement.textContent = originalText;
        linkElement.style.color = '#3b82f6';
        linkElement.style.fontWeight = 'bold';
    }, 2000);
}

// Show error message after failed copy
function showCopyError(linkElement, message) {
    const originalText = linkElement.textContent;
    linkElement.textContent = message;
    linkElement.style.color = '#ef4444';
    
    setTimeout(() => {
        linkElement.textContent = originalText;
        linkElement.style.color = '#3b82f6';
    }, 2000);
}

// Start streaming a camera
function startStream(cameraId) {
    fetch(`/api/stream/start/${cameraId}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                cameras[cameraId].status = 'streaming';
                displayCameras();
            }
        })
        .catch(error => console.error('Start stream error:', error));
}

// Stop streaming a camera
function stopStream(cameraId) {
    fetch(`/api/stream/stop/${cameraId}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            cameras[cameraId].status = 'available';
            displayCameras();
        })
        .catch(error => {
            console.error('Stop stream error:', error);
            if (cameras[cameraId]) {
                cameras[cameraId].status = 'available';
                displayCameras();
            }
        });
}

// Load the application on page load
window.onload = function() {
    scanCameras();
};

// Update every 10 seconds
setInterval(() => {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            cameras = data.cameras;
            displayCameras();
        })
        .catch(error => console.log('Status update skipped'));
}, 10000);