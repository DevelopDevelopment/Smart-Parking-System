const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const CAM_ID = window.SETUP_CAM_ID || '';

// all saved spots and the points being placed for the current spot
let allSpots = [];
let currentPoints = [];
let backgroundImage = null;

// mode is either 'add' (placing new spots) or 'edit' (moving corners of an existing spot)
let mode = 'add';
let editingIndex = -1;
let editPoints = [];
let draggingPointIdx = -1;

// how close the cursor needs to be (in px) to grab a corner handle
const DRAG_RADIUS = 18;

// load the camera frame as the canvas background
const img = new Image();
img.onload = function () {
    canvas.width = img.width;
    canvas.height = img.height;
    backgroundImage = img;
    redraw();
};
img.src = '/get_frame_for_setup?cam=' + CAM_ID + '&t=' + Date.now();

// converts mouse event coords to canvas-space coords (handles CSS scaling)
function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
}

// start dragging a corner handle if the cursor is close enough to one
canvas.addEventListener('mousedown', function (e) {
    if (mode !== 'edit') return;
    const { x, y } = getCanvasPos(e);
    for (let i = 0; i < editPoints.length; i++) {
        const dx = editPoints[i].x - x;
        const dy = editPoints[i].y - y;
        if (Math.sqrt(dx * dx + dy * dy) < DRAG_RADIUS) {
            draggingPointIdx = i;
            e.preventDefault();
            break;
        }
    }
});

// update cursor style and move the dragged corner
canvas.addEventListener('mousemove', function (e) {
    if (mode === 'edit') {
        const { x, y } = getCanvasPos(e);
        const onHandle = editPoints.some(p => {
            const dx = p.x - x, dy = p.y - y;
            return Math.sqrt(dx * dx + dy * dy) < DRAG_RADIUS;
        });
        canvas.style.cursor = draggingPointIdx !== -1 ? 'grabbing' : (onHandle ? 'grab' : 'default');
        if (draggingPointIdx !== -1) {
            editPoints[draggingPointIdx] = { x, y };
            redraw();
        }
    } else {
        canvas.style.cursor = 'crosshair';
    }
});

canvas.addEventListener('mouseup', function () {
    draggingPointIdx = -1;
});

canvas.addEventListener('mouseleave', function () {
    draggingPointIdx = -1;
});

// place a new point when clicking in add mode (max 4 points per spot)
canvas.addEventListener('click', function (e) {
    if (mode !== 'add') return;
    if (currentPoints.length >= 4) return;
    if (draggingPointIdx !== -1) return;
    const { x, y } = getCanvasPos(e);
    currentPoints.push({ x, y });
    updateUI();
    redraw();
});

// redraws the background, all saved spots, and whatever is currently being placed/edited
function redraw() {
    if (!backgroundImage) return;
    ctx.drawImage(backgroundImage, 0, 0);

    allSpots.forEach((spot, idx) => {
        const isEditing = (mode === 'edit' && idx === editingIndex);
        const pts = isEditing ? editPoints : spot.points;

        // purple when editing, green otherwise
        if (isEditing) {
            ctx.fillStyle = 'rgba(99, 102, 241, 0.22)';
            ctx.strokeStyle = '#818cf8';
            ctx.lineWidth = 3;
        } else {
            ctx.fillStyle = 'rgba(34, 197, 94, 0.28)';
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 3;
        }

        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // draw the spot number label at the center of the polygon
        const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
        const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
        ctx.fillStyle = isEditing ? '#a5b4fc' : '#22c55e';
        ctx.font = 'bold 20px "Open Sans", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`#${idx + 1}`, cx, cy + 6);

        // draw draggable corner handles when editing
        if (isEditing) {
            editPoints.forEach((p, i) => {
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = '#818cf8';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 10, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = '#4f46e5';
                ctx.font = 'bold 11px "Open Sans", sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(i + 1, p.x, p.y + 4);
            });
        }
    });

    // draw the in-progress points and dashed outline while placing a new spot
    if (mode === 'add') {
        currentPoints.forEach((point, idx) => {
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(point.x, point.y, 10, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#0a0a0a';
            ctx.font = 'bold 12px "Open Sans", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(idx + 1, point.x, point.y + 4);
        });

        if (currentPoints.length > 1) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2;
            ctx.setLineDash([8, 4]);
            ctx.beginPath();
            ctx.moveTo(currentPoints[0].x, currentPoints[0].y);
            for (let i = 1; i < currentPoints.length; i++) ctx.lineTo(currentPoints[i].x, currentPoints[i].y);
            if (currentPoints.length === 4) ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
}

// save the current 4 points as a new spot via the API
function addSpot() {
    if (currentPoints.length !== 4) return;
    fetch('/add_parking_spot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: CAM_ID, points: currentPoints })
    }).then(r => r.json()).then(data => {
        if (data.success) {
            allSpots.push({ id: data.spot_id, points: [...currentPoints] });
            currentPoints = [];
            updateUI();
            redraw();
            updateSpotList();
        }
    });
}

// remove the last placed point
function undoPoint() {
    if (currentPoints.length > 0) {
        currentPoints.pop();
        updateUI();
        redraw();
    }
}

// delete all spots from the server and reset local state
function clearAll() {
    if (!confirm('Clear all parking spots? This cannot be undone.')) return;
    fetch('/clear_parking_spots', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cam_id: CAM_ID }) }).then(() => {
        currentPoints = [];
        allSpots = [];
        if (mode === 'edit') cancelEdit();
        updateUI();
        redraw();
        updateSpotList();
    });
}

function finishSetup() {
    if (allSpots.length === 0) {
        alert('Please add at least one parking spot before finishing.');
        return;
    }
    window.location.href = '/';
}

// switch to edit mode for a specific spot, copying its points so edits are non-destructive until saved
function enterEditMode(index) {
    mode = 'edit';
    editingIndex = index;
    editPoints = allSpots[index].points.map(p => ({ ...p }));
    draggingPointIdx = -1;
    canvas.style.cursor = 'grab';
    updateUI();
    redraw();
}

// discard any edits and return to add mode
function cancelEdit() {
    mode = 'add';
    editingIndex = -1;
    editPoints = [];
    draggingPointIdx = -1;
    canvas.style.cursor = 'crosshair';
    updateUI();
    redraw();
    updateSpotList();
}

// send the updated corner positions to the server and exit edit mode
function saveEdit() {
    const spot = allSpots[editingIndex];
    fetch('/update_parking_spot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: CAM_ID, spot_id: spot.id, points: editPoints })
    }).then(r => r.json()).then(data => {
        if (data.success) {
            allSpots[editingIndex].points = editPoints.map(p => ({ ...p }));
            cancelEdit();
        } else {
            alert('Failed to save: ' + (data.error || 'unknown error'));
        }
    });
}

function deleteSpot(index) {
    const spot = allSpots[index];
    if (!confirm(`Delete Spot #${index + 1}? This cannot be undone.`)) return;
    fetch('/delete_parking_spot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: CAM_ID, spot_id: spot.id })
    }).then(r => r.json()).then(data => {
        if (data.success) {
            if (mode === 'edit' && editingIndex === index) cancelEdit();
            allSpots.splice(index, 1);
            updateUI();
            redraw();
            updateSpotList();
        } else {
            alert('Failed to delete: ' + (data.error || 'unknown error'));
        }
    });
}

// sync all UI controls and labels to the current mode and state
function updateUI() {
    document.getElementById('spot-count').textContent = allSpots.length;

    const addControls = document.getElementById('addControls');
    const editControls = document.getElementById('editControls');
    const addInstructions = document.getElementById('addInstructions');
    const editBanner = document.getElementById('editBanner');
    const hint = document.getElementById('canvasHint');
    const modeLabel = document.getElementById('modeLabel');

    if (mode === 'edit') {
        addControls.style.display = 'none';
        editControls.style.display = '';
        addInstructions.style.display = 'none';
        editBanner.classList.add('visible');
        document.getElementById('editBannerText').textContent = `Editing Spot #${editingIndex + 1} — drag the corner handles to reposition them.`;
        hint.textContent = 'Drag corners to adjust the spot shape';
        hint.style.display = 'block';
        modeLabel.innerHTML = `Editing: <strong>Spot #${editingIndex + 1}</strong>`;
    } else {
        addControls.style.display = '';
        editControls.style.display = 'none';
        addInstructions.style.display = '';
        editBanner.classList.remove('visible');
        document.getElementById('btnAddSpot').disabled = currentPoints.length !== 4;

        if (currentPoints.length < 4) {
            hint.textContent = `Click to place point ${currentPoints.length + 1}`;
        } else {
            hint.textContent = 'Click "Add Spot" to save';
        }

        hint.style.display = 'block';
        modeLabel.innerHTML = `Points: <strong id="point-count">${currentPoints.length}</strong>/4`;
    }
}

// re-render the sidebar list of configured spots
function updateSpotList() {
    const empty = document.getElementById('spotsEmpty');
    const items = document.getElementById('spotsItems');

    if (allSpots.length === 0) {
        empty.style.display = 'block';
        items.innerHTML = '';
        return;
    }

    empty.style.display = 'none';
    items.innerHTML = allSpots.map((spot, idx) => `
        <div class="spot-item" id="spot-item-${idx}">
            <span class="spot-number">#${idx + 1}</span>
            <span class="spot-status">Ready</span>
            <div class="spot-actions">
                <button class="spot-btn edit ${mode === 'edit' && editingIndex === idx ? 'active' : ''}"
                        onclick="enterEditMode(${idx})">
                    Edit
                </button>
                <button class="spot-btn del" onclick="deleteSpot(${idx})">
                    Delete
                </button>
            </div>
        </div>
    `).join('');
}

// refresh the background frame every 2s so the image stays current (skipped while editing)
setInterval(() => {
    if (mode === 'edit') return;
    const newImg = new Image();
    newImg.onload = function () {
        backgroundImage = newImg;
        redraw();
    };
    newImg.src = '/get_frame_for_setup?cam=' + CAM_ID + '&t=' + Date.now();
}, 2000);

// load any existing spots from the server on page load
fetch('/get_parking_spots?cam=' + CAM_ID).then(r => r.json()).then(data => {
    if (data.spots && data.spots.length > 0) {
        allSpots = data.spots.map(s => ({
            id: s.id,
            points: s.points.map(p => ({ x: p[0], y: p[1] }))
        }));
        updateUI();
        updateSpotList();
        redraw();
    }
});

canvas.style.cursor = 'crosshair';
updateUI();