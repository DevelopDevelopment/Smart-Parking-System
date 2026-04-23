const CAMERAS = window.CAMERAS_DATA;
let activeCamIds = new Set(window.ACTIVE_CAM_IDS);
let apiKeys = Object.assign({}, window.API_KEYS || {});

const thumbTimers = {};

function loadThumb(camId) {
    if (activeCamIds.has(camId)) return;

    const img = document.getElementById('preview-' + camId);
    const loader = document.getElementById('loader-' + camId);

    function tryLoad() {
        if (activeCamIds.has(camId)) return;
        fetch('/api/camera_thumbnail/' + camId + '?t=' + Date.now())
            .then(res => {
                if (res.status === 202) {
                    setTimeout(tryLoad, 2500);
                    return;
                }
                if (res.ok) {
                    img.onload = () => { loader.style.display = 'none'; };
                    img.src = '/api/camera_thumbnail/' + camId + '?t=' + Date.now();
                    img.style.display = 'block';
                }
            })
            .catch(() => {
                loader.innerHTML = '<span style="font-size:11px;color:var(--gray)">Unavailable</span>';
            });
    }

    tryLoad();

    clearInterval(thumbTimers[camId]);
    thumbTimers[camId] = setInterval(() => {
        if (activeCamIds.has(camId)) {
            clearInterval(thumbTimers[camId]);
            return;
        }
        img.src = '/api/camera_thumbnail/' + camId + '?t=' + Date.now();
    }, 6000);
}

function refreshThumb(camId) {
    const loader = document.getElementById('loader-' + camId);
    const img = document.getElementById('preview-' + camId);
    loader.style.display = 'flex';
    loader.innerHTML = '<div class="spinner"></div><span>Refreshing…</span>';
    img.style.display = 'none';
    fetch('/api/refresh_thumbnail/' + camId, { method: 'POST' })
        .then(() => setTimeout(() => loadThumb(camId), 1800));
}

CAMERAS.forEach((cam, i) => {
    if (!activeCamIds.has(cam.id)) {
        setTimeout(() => loadThumb(cam.id), i * 600);
    }
});

CAMERAS.forEach(cam => {
    const el = document.getElementById('apiUrl-' + cam.id);
    if (el) {
        const key = apiKeys[cam.id] || '';
        const url = window.location.origin + '/api/cameras/' + cam.id + '/status' + (key ? '?key=' + key : '');
        el.textContent = url;
        el.title = url;
    }
});

function updateStats() {
    fetch('/api/camera_stats')
        .then(r => r.json())
        .then(data => {
            let activeCount = 0;
            data.forEach(cam => {
                if (!cam.active) return;
                activeCount++;

                const t = id => document.getElementById(id + cam.id);
                if (t('stat-total-')) t('stat-total-').textContent = cam.total;
                if (t('stat-avail-')) t('stat-avail-').textContent = cam.available ?? '—';
                if (t('stat-occ-')) t('stat-occ-').textContent = cam.occupied ?? '—';

                if (focusedCamId === cam.id) {
                    document.getElementById('focusTotal').textContent = cam.total;
                    document.getElementById('focusAvail').textContent = cam.available ?? '—';
                    document.getElementById('focusOcc').textContent = cam.occupied ?? '—';
                    document.getElementById('focusFps').textContent = cam.fps;
                    document.getElementById('focusStatusText').textContent = cam.status;
                }
            });
            const countEl = document.getElementById('activeCountText');
            if (countEl) countEl.textContent = activeCount + ' Active';
        })
        .catch(() => {});
}

setInterval(updateStats, 2000);
updateStats();

function toggleCamera(camId) {
    const btn = document.getElementById('toggleBtn-' + camId);
    btn.classList.add('loading');

    fetch('/api/toggle_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: camId })
    })
        .then(r => r.json())
        .then(data => {
            btn.classList.remove('loading');

            if (!data.success) {
                showToast('Error: ' + (data.error || 'unknown'));
                return;
            }

            const isNowActive = data.active;
            const wrapper = document.getElementById('wrapper-' + camId);
            const preview = document.getElementById('preview-' + camId);
            const loader = document.getElementById('loader-' + camId);
            const liveBadge = document.getElementById('liveBadge-' + camId);
            const occ = document.getElementById('occupancy-' + camId);
            const cam = CAMERAS.find(c => c.id === camId);

            if (isNowActive) {
                activeCamIds.add(camId);
                wrapper.classList.add('is-active');
                clearInterval(thumbTimers[camId]);

                loader.style.display = 'flex';
                loader.innerHTML = '<div class="spinner"></div><span>Connecting…</span>';
                preview.onload = () => { loader.style.display = 'none'; };
                preview.src = '/video_feed/' + camId + '?t=' + Date.now();
                preview.style.display = 'block';
                liveBadge.style.display = 'flex';

                btn.className = 'btn-toggle on';
                btn.innerHTML = `
                <span class="toggle-on-text">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="m9 12 2 2 4-4"/>
                    </svg>
                    Running
                </span>
                <span class="toggle-off-text">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="m15 9-6 6"/>
                        <path d="m9 9 6 6"/>
                    </svg>
                    Stop
                </span>`;

                if (occ && cam) {
                    occ.innerHTML = `
                    <span class="occ-stat total"><span class="occ-dot"></span><strong id="stat-total-${camId}">${cam.parking_spots.length}</strong> total</span>
                    <span class="occ-stat available"><span class="occ-dot"></span><strong id="stat-avail-${camId}">—</strong> free</span>
                    <span class="occ-stat occupied"><span class="occ-dot"></span><strong id="stat-occ-${camId}">—</strong> taken</span>`;
                }

                showToast('Camera started');
            } else {
                activeCamIds.delete(camId);
                wrapper.classList.remove('is-active');
                preview.src = '';
                preview.style.display = 'none';
                loader.style.display = 'flex';
                loader.innerHTML = '<div class="spinner"></div><span>Loading…</span>';
                liveBadge.style.display = 'none';
                setTimeout(() => loadThumb(camId), 1200);

                btn.className = 'btn-toggle off';
                btn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;">
                    <polygon points="6 3 20 12 6 21 6 3"/>
                </svg>
                Start`;

                if (occ && cam) {
                    occ.innerHTML = `
                    <span class="occ-stat total inactive"><span class="occ-dot"></span><strong>${cam.parking_spots.length}</strong> spot${cam.parking_spots.length !== 1 ? 's' : ''} configured</span>
                    <span class="occ-stat inactive" style="color:var(--gray);border-color:var(--gray-dark);">Enable to track live</span>`;
                }

                showToast('Camera stopped');
            }
        })
        .catch(() => {
            btn.classList.remove('loading');
            showToast('Network error');
        });
}

function toggleSettings(camId) {
    const panel = document.getElementById('settings-' + camId);
    const btn = document.getElementById('settingsBtn-' + camId);
    panel.classList.toggle('open');
    btn.classList.toggle('open');
}

function onInterval(camId, val) {
    document.getElementById('intervalVal-' + camId).textContent = 'Every ' + val + ' second' + (parseInt(val) === 1 ? '' : 's');
    saveSettings(camId);
}

function onConf(camId, val) {
    document.getElementById('confVal-' + camId).textContent = Math.round(val * 100) + '%';
}

function saveSettings(camId) {
    const btn = document.getElementById('saveBtn-' + camId);
    btn.textContent = 'Saving…';
    btn.disabled = true;

    fetch('/api/update_camera_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            camera_id: camId,
            settings: {
                model: document.getElementById('model-' + camId).value,
                fps_limit: parseInt(document.getElementById('fps-' + camId).value),
                detection_interval: parseInt(document.getElementById('interval-' + camId).value),
                confidence_threshold: parseFloat(document.getElementById('conf-' + camId).value),
                show_boxes: document.getElementById('boxes-' + camId).checked,
                show_parking_spots: document.getElementById('parking-spots-' + camId).checked
            }
        })
    })
        .then(r => r.json())
        .then(data => {
            btn.disabled = false;
            if (data.success) {
                btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="width:16px;height:16px;"><polyline points="20 6 9 17 4 12"/></svg> Saved!`;
                btn.classList.add('saved');
                showToast(data.restarted ? 'Settings saved — camera restarting with new model' : 'Settings saved');
                setTimeout(() => {
                    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px;"><path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/></svg> Save Settings`;
                    btn.classList.remove('saved');
                }, 2000);
            } else {
                btn.textContent = 'Save Settings';
                showToast('Error: ' + (data.error || 'unknown'));
            }
        })
        .catch(() => {
            btn.disabled = false;
            btn.textContent = 'Save Settings';
            showToast('Network error');
        });
}

let focusedCamId = null;
let focusOverlayEnabled = true;

function openFocus(camId) {
    focusedCamId = camId;
    const cam = CAMERAS.find(c => c.id === camId);
    const isActive = activeCamIds.has(camId);

    document.getElementById('focusTitle').textContent = cam ? cam.name : camId;
    document.getElementById('focusSource').textContent = cam ? cam.source : '';
    document.getElementById('focusStats').style.display = isActive ? 'flex' : 'none';
    document.getElementById('focusStatusText').textContent = isActive ? 'LIVE' : '';

    const body = document.getElementById('focusBody');
    const feedMode = focusOverlayEnabled ? 'video_feed' : 'raw_feed';
    body.innerHTML = `<img id="focusFeed" src="/${feedMode}/${camId}?t=${Date.now()}" alt="Live feed" style="width:100%;display:block;">`;

    if (!isActive) {
        body.innerHTML += `<div class="focus-inactive-msg">
            <p>Camera is not running</p>
            <span>Start this camera to view the live detection feed</span>
        </div>`;
    }

    const overlayBtn = document.getElementById('focusOverlayBtn');
    if (overlayBtn) {
        if (isActive) {
            overlayBtn.style.display = 'inline-block';
            overlayBtn.textContent = focusOverlayEnabled ? 'Hide overlays' : 'Show overlays';
        } else {
            overlayBtn.style.display = 'none';
        }
    }

    document.getElementById('focusModal').classList.add('active');
}

const overlayBtn = document.getElementById('focusOverlayBtn');
if (overlayBtn) {
    overlayBtn.addEventListener('click', () => {
        focusOverlayEnabled = !focusOverlayEnabled;
        overlayBtn.textContent = focusOverlayEnabled ? 'Hide overlays' : 'Show overlays';
        if (focusedCamId) {
            const feed = document.getElementById('focusFeed');
            if (feed) {
                const feedMode = focusOverlayEnabled ? 'video_feed' : 'raw_feed';
                feed.src = `/${feedMode}/${focusedCamId}?t=${Date.now()}`;
            }
        }
    });
}

function closeFocus() {
    document.getElementById('focusModal').classList.remove('active');
    document.getElementById('focusBody').innerHTML = '';
    focusedCamId = null;
}

function closeFocusOnBackdrop(e) {
    if (e.target === document.getElementById('focusModal')) closeFocus();
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        closeFocus();
        closeAddCamera();
    }
});

function deleteCamera(camId) {
    const cam = CAMERAS.find(c => c.id === camId);
    const name = cam ? cam.name : camId;
    if (!confirm(`Delete camera "${name}"? This cannot be undone.`)) return;

    const btn = document.getElementById('deleteBtn-' + camId);
    btn.disabled = true;

    fetch('/api/delete_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: camId })
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                showToast('Camera deleted');
                setTimeout(() => location.reload(), 600);
            } else {
                showToast('Error: ' + (data.error || 'unknown'));
                btn.disabled = false;
            }
        })
        .catch(() => {
            showToast('Network error');
            btn.disabled = false;
        });
}

function copyApiUrl(camId) {
    const key = apiKeys[camId] || '';
    const url = window.location.origin + '/api/cameras/' + camId + '/status' + (key ? '?key=' + key : '');
    const btn = document.getElementById('apiCopyBtn-' + camId);

    const copyIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px;"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`;

    const doCopy = () => {
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="width:13px;height:13px;"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
        btn.classList.add('copied');
        showToast('API URL copied (includes key)');
        setTimeout(() => {
            btn.innerHTML = copyIcon + ' Copy';
            btn.classList.remove('copied');
        }, 2500);
    };

    if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(url).then(doCopy).catch(() => fallbackCopy(url, doCopy));
    } else {
        fallbackCopy(url, doCopy);
    }
}

function fallbackCopy(text, cb) {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none;';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); cb(); } catch (e) {}
    document.body.removeChild(ta);
}

function regenerateApiKey(camId) {
    if (!confirm('Regenerate API key? Anyone using the old key will lose access.')) return;

    fetch('/api/regenerate_api_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: camId })
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                apiKeys[camId] = data.api_key;
                const el = document.getElementById('apiUrl-' + camId);
                if (el) {
                    const url = window.location.origin + '/api/cameras/' + camId + '/status?key=' + data.api_key;
                    el.textContent = url;
                    el.title = url;
                }
                showToast('API key regenerated');
            } else {
                showToast('Error: ' + (data.error || 'unknown'));
            }
        })
        .catch(() => showToast('Network error'));
}

let toastTimer;

function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}

CAMERAS.forEach(cam => {
    const copyBtn = document.getElementById('apiCopyBtn-' + cam.id);
    if (copyBtn) {
        const regenBtn = document.createElement('button');
        regenBtn.className = 'api-regen-btn';
        regenBtn.title = 'Regenerate API key';
        regenBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px;">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
            <path d="M3 3v5h5"/>
        </svg>`;
        regenBtn.onclick = () => regenerateApiKey(cam.id);
        copyBtn.parentNode.insertBefore(regenBtn, copyBtn.nextSibling);
    }
});

// add camera modal

let acCurrentTab = 'url';

function addCamera() {
    document.getElementById('acModal').classList.add('active');
    acSwitchTab('url');
    acClearError();
    acResetFields();
    acLoadDevices();
}

function closeAddCamera() {
    document.getElementById('acModal').classList.remove('active');
}

function acBackdropClose(e) {
    if (e.target === document.getElementById('acModal')) closeAddCamera();
}

function acSwitchTab(tab) {
    acCurrentTab = tab;
    document.getElementById('panelUrl').style.display = tab === 'url' ? 'flex' : 'none';
    document.getElementById('panelDevice').style.display = tab === 'device' ? 'flex' : 'none';
    document.getElementById('tabUrl').classList.toggle('active', tab === 'url');
    document.getElementById('tabDevice').classList.toggle('active', tab === 'device');
    acClearError();
}

function acSlugify(val) {
    return val.trim().toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
}

function acSyncId(nameInputId) {
    const slug = acSlugify(document.getElementById(nameInputId).value);
    const idField = nameInputId === 'urlName' ? 'urlCamId' : 'devCamId';
    document.getElementById(idField).value = slug;
}

function acResetFields() {
    ['urlName', 'urlSource', 'urlCamId', 'devName', 'devCamId'].forEach(id => {
        document.getElementById(id).value = '';
    });
    const btn = document.getElementById('acSubmit');
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;"><path d="M5 12h14"/><path d="M12 5v14"/></svg> Add Camera`;
}

function acShowError(msg) {
    const el = document.getElementById('acError');
    el.textContent = msg;
    el.style.display = 'block';
}

function acClearError() {
    const el = document.getElementById('acError');
    el.style.display = 'none';
    el.textContent = '';
}

function acLoadDevices() {
    const sel = document.getElementById('devSelect');
    const hint = document.getElementById('devHint');
    sel.innerHTML = '<option value="">Scanning...</option>';
    sel.disabled = true;
    hint.textContent = 'Scanning for connected cameras...';

    fetch('/api/list_local_cameras')
        .then(r => r.json())
        .then(data => {
            sel.disabled = false;
            if (!data.success || !data.cameras || data.cameras.length === 0) {
                sel.innerHTML = '<option value="">No devices found</option>';
                hint.textContent = 'No local cameras detected. Check your connections.';
                return;
            }
            sel.innerHTML = data.cameras.map(idx =>
                `<option value="${idx}">Device ${idx}</option>`
            ).join('');
            hint.textContent = `${data.cameras.length} device${data.cameras.length !== 1 ? 's' : ''} found`;
        })
        .catch(() => {
            sel.disabled = false;
            sel.innerHTML = '<option value="">Failed to load</option>';
            hint.textContent = 'Could not reach the server.';
        });
}

function acSubmit() {
    acClearError();

    let name, source, camId, sourceType;

    if (acCurrentTab === 'url') {
        name = document.getElementById('urlName').value.trim();
        source = document.getElementById('urlSource').value.trim();
        camId = document.getElementById('urlCamId').value.trim();
        sourceType = 'url';

        if (!name) return acShowError('Camera name is required.');
        if (!source) return acShowError('Stream URL is required.');
        if (!camId) return acShowError('Camera ID is required.');
    } else {
        name = document.getElementById('devName').value.trim();
        const devIdx = document.getElementById('devSelect').value;
        camId = document.getElementById('devCamId').value.trim();
        sourceType = 'device';

        if (!name) return acShowError('Camera name is required.');
        if (devIdx === '') return acShowError('Select a device.');
        if (!camId) return acShowError('Camera ID is required.');

        source = devIdx;
    }

    const btn = document.getElementById('acSubmit');
    btn.disabled = true;
    btn.textContent = 'Adding...';

    fetch('/api/add_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: camId, name, source, source_type: sourceType })
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                closeAddCamera();
                showToast('Camera added');
                setTimeout(() => location.reload(), 600);
            } else {
                acShowError(data.error || 'Failed to add camera.');
                btn.disabled = false;
                btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;"><path d="M5 12h14"/><path d="M12 5v14"/></svg> Add Camera`;
            }
        })
        .catch(() => {
            acShowError('Network error. Is the server running?');
            btn.disabled = false;
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;"><path d="M5 12h14"/><path d="M12 5v14"/></svg> Add Camera`;
        });
}