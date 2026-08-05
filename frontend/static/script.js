document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const localVideo = document.getElementById('local-video');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const octx = overlayCanvas.getContext('2d');
    const videoPlaceholder = document.getElementById('video-placeholder');
    
    // Status elements
    const currentState = document.getElementById('current-state');
    const alertBox = document.getElementById('alert-box');
    
    // Metrics elements
    const metricEyes = document.getElementById('metric-eyes');
    const metricMar = document.getElementById('metric-mar');
    const metricPerclos = document.getElementById('metric-perclos');
    const metricYawns = document.getElementById('metric-yawns');
    const metricSession = document.getElementById('metric-session');
    const metricMl = document.getElementById('metric-ml');
    const wsStatus  = document.getElementById('ws-status');

    const icons = {
        "ALERT": "🟢",
        "DROWSY": "🟡",
        "VERY_DROWSY": "🔴",
        "AWAY": "🔵",
        "NO_FACE": "⚫",
        "STOPPED": "⚪"
    };

    // Web Audio API Alarm System
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    let alarmInterval = null;

    function playAlarmSound() {
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        // Siren effect
        oscillator.type = 'square';
        oscillator.frequency.setValueAtTime(800, audioCtx.currentTime); 
        oscillator.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.2); 
        
        gainNode.gain.setValueAtTime(0.5, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 0.3);
    }

    function startAlarm() {
        if (!alarmInterval) {
            alarmInterval = setInterval(playAlarmSound, 400); 
        }
    }

    function stopAlarm() {
        if (alarmInterval) {
            clearInterval(alarmInterval);
            alarmInterval = null;
        }
    }

    // History tracking
    let sessionHistory = [];
    let previousState = "STOPPED";
    const historyList = document.getElementById('history-list');
    const historyPlaceholder = document.getElementById('history-placeholder');
    const btnExport = document.getElementById('btn-export');

    let ws = null;
    let isMonitoring = false;
    let localStream = null;
    let captureInterval = null;
    
    // Create an invisible video element to capture the webcam feed
    const hiddenVideo = document.createElement('video');
    hiddenVideo.autoplay = true;
    hiddenVideo.playsInline = true;
    hiddenVideo.muted = true;
    
    // Create an offscreen canvas to extract frames
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Make sure to set the correct backend URL when deployed!
    // In production, change this to your Railway app URL (e.g. wss://your-app.up.railway.app/ws)
    const backendHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
                        ? `${window.location.host}` 
                        : "drowsiness-detection-production-98d6.up.railway.app";
    
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${backendHost}/ws`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected to', wsUrl);
            if (wsStatus) { wsStatus.textContent = '🟢 Connected to backend'; wsStatus.className = 'ws-status connected'; }
            
            // If the user already clicked start while connecting, kickstart the loop now!
            if (isMonitoring && localStream) {
                ws.send(JSON.stringify({ type: 'start' }));
                sendNextFrame();
            }
        };

        ws.onmessage = (event) => {
            isSending = false; // UNLOCK! Server finished processing the frame!
            const response = JSON.parse(event.data);
            
            if (response.data) {
                updateDashboard(response.data);
                drawOverlay(response.data);
                
                // Ask for the next frame ONLY after we finished rendering this one!
                if (isMonitoring) {
                    requestAnimationFrame(sendNextFrame);
                }
            } else if (response.error) {
                // If the server rejected it (e.g. not running yet), keep trying
                if (isMonitoring) {
                    setTimeout(sendNextFrame, 50);
                }
            }
        };

        ws.onclose = () => {
            isSending = false;
            console.log('WebSocket disconnected. Reconnecting...');
            if (wsStatus) { wsStatus.textContent = '🔴 Disconnected — reconnecting…'; wsStatus.className = 'ws-status error'; }
            setTimeout(connectWebSocket, 2000);
        };
    }

    function updateDashboard(data) {
        // Update state badge
        const state = data.state || 'STOPPED';
        const icon = icons[state] || '⚪';
        currentState.innerHTML = `${icon} ${state.replace('_', ' ')}`;

        // Update alert message
        const alertMsg = data.alert_msg || '';
        if (state === "VERY_DROWSY") {
            alertBox.className = 'alert alert-danger';
            alertBox.innerHTML = `🚨 ${alertMsg}`;
            startAlarm();
        } else if (["DROWSY", "NO_FACE", "AWAY"].includes(state) && alertMsg) {
            alertBox.className = 'alert alert-warn';
            alertBox.innerHTML = `⚠️ ${alertMsg}`;
            if (state === "DROWSY" || state === "AWAY") {
                startAlarm();
            } else {
                stopAlarm();
            }
        } else if (state === "STOPPED") {
            alertBox.className = 'alert alert-ok';
            alertBox.innerHTML = `ℹ️ System stopped.`;
            stopAlarm();
        } else {
            alertBox.className = 'alert alert-ok';
            alertBox.innerHTML = `✅ Driver Alert — All Good`;
            stopAlarm();
        }

        // Log state changes to history
        if (state !== previousState && state !== "STOPPED") {
            logHistoryEvent(state, alertMsg || state.replace('_', ' '));
            previousState = state;
        }

        // Update metrics
        if (data.eyes !== undefined) metricEyes.innerText = data.eyes.toFixed(3);
        if (data.mar !== undefined) metricMar.innerText = data.mar.toFixed(3);
        if (data.perclos !== undefined) metricPerclos.innerText = `${Math.round(data.perclos * 100)}%`;
        if (data.yawn_count !== undefined) metricYawns.innerText = data.yawn_count;
        if (data.session_duration !== undefined) metricSession.innerText = `${data.session_duration}s`;
        if (data.ml_label !== undefined) metricMl.innerText = `${data.ml_label} ${Math.round((data.confidence || 0) * 100)}%`;
    }

    let isSending = false;

    function drawOverlay(data) {
        // Clear previous overlay
        octx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (data.state === "VERY_DROWSY") {
            // Draw red screen alert!
            octx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            octx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            octx.fillStyle = 'red';
            octx.font = 'bold 40px Arial';
            octx.fillText("WAKE UP!", overlayCanvas.width/2 - 100, overlayCanvas.height/2);
        }
        
        if (data.face_box) {
            const [x, y, w, h] = data.face_box;
            
            // Draw green bounding box for the face
            octx.strokeStyle = data.state === "ALERT" ? '#10b981' : (data.state === "DROWSY" ? '#f59e0b' : '#ef4444');
            octx.lineWidth = 4;
            octx.strokeRect(x, y, w, h);
            
            // Draw text tag
            octx.fillStyle = octx.strokeStyle;
            octx.fillRect(x, y - 30, w, 30);
            octx.fillStyle = '#fff';
            octx.font = 'bold 20px Arial';
            octx.fillText(data.state, x + 10, y - 8);
        }

        // Draw left eye box
        if (data.left_eye_box) {
            const [x, y, w, h] = data.left_eye_box;
            octx.strokeStyle = data.eyes_closed ? '#ef4444' : '#10b981';
            octx.lineWidth = 2;
            octx.strokeRect(x, y, w, h);
            octx.fillStyle = octx.strokeStyle;
            octx.font = 'bold 11px Arial';
            octx.fillText(`EYE: ${data.eyes}`, x, y - 4);
        }

        // Draw right eye box
        if (data.right_eye_box) {
            const [x, y, w, h] = data.right_eye_box;
            octx.strokeStyle = data.eyes_closed ? '#ef4444' : '#10b981';
            octx.lineWidth = 2;
            octx.strokeRect(x, y, w, h);
            octx.fillStyle = octx.strokeStyle;
            octx.font = 'bold 11px Arial';
            octx.fillText(`EYE: ${data.eyes}`, x, y - 4);
        }

        // Draw mouth box
        if (data.mouth_box) {
            const [x, y, w, h] = data.mouth_box;
            octx.strokeStyle = data.yawning_now ? '#ef4444' : '#f59e0b';
            octx.lineWidth = 2;
            octx.strokeRect(x, y, w, h);
            octx.fillStyle = octx.strokeStyle;
            octx.font = 'bold 11px Arial';
            octx.fillText(data.yawning_now ? "YAWNING!" : `MAR: ${data.mar}`, x, y - 4);
        }
    }

    function sendNextFrame() {
        if (isSending) return; // STRICT CONCURRENCY LOCK!
        if (isMonitoring && ws && ws.readyState === WebSocket.OPEN && localStream) {
            isSending = true;
            
            // Send to backend via invisible canvas
            // Mirror it so Python sees the same thing the user sees (which is mirrored by CSS)
            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
            ctx.restore();
            
            const base64Image = canvas.toDataURL('image/jpeg', 0.5); 
            ws.send(JSON.stringify({ type: "frame", image: base64Image }));
        }
    }

    async function startCamera() {
        try {
            // Request High-Definition (HD) video
            localStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } 
            });
            localVideo.srcObject = localStream;
            
            // Wait for the video to start playing before sending the first frame
            localVideo.onplaying = () => {
                // Get native webcam resolution to prevent stretching
                const vw = localVideo.videoWidth || 640;
                const vh = localVideo.videoHeight || 480;
                
                // Cap the maximum width for the ML processing canvas to ensure FAST network
                const maxW = 480; // Small payload for max speed!
                if (vw > maxW) {
                    canvas.width = maxW;
                    canvas.height = Math.floor(vh * (maxW / vw));
                } else {
                    canvas.width = vw;
                    canvas.height = vh;
                }
                
                // Setup the overlay canvas to MATCH the local video perfectly
                overlayCanvas.width = maxW; 
                overlayCanvas.height = Math.floor(vh * (maxW / vw)); 
                
                if (isMonitoring) sendNextFrame();
            };
            
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access the webcam. Please grant permissions.");
        }
    }

    function stopCamera() {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
    }

    btnStart.addEventListener('click', () => {
        // CRITICAL FIX: Modern browsers block audio unless it's started by a user gesture.
        // We must resume the AudioContext right when the user clicks 'Start Monitoring'!
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'start' }));
        }
        
        localVideo.classList.remove('hidden');
        overlayCanvas.classList.remove('hidden');
        videoPlaceholder.classList.add('hidden');
        
        isMonitoring = true;
        startCamera();
    });

    btnStop.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stop' }));
        }
        
        localVideo.classList.add('hidden');
        overlayCanvas.classList.add('hidden');
        videoPlaceholder.classList.remove('hidden');
        
        isMonitoring = false;
        stopCamera();
        
        updateDashboard({
            state: "STOPPED",
            alert_msg: "Monitoring Stopped",
            eyes: 0, mar: 0, perclos: 0, yawn_count: 0, session_duration: 0, ml_label: "-", confidence: 0
        });
    });

    function logHistoryEvent(state, message) {
        if (historyPlaceholder) historyPlaceholder.remove();
        
        const timestamp = new Date().toLocaleTimeString();
        sessionHistory.push({ time: timestamp, state: state, message: message });
        
        const li = document.createElement('li');
        li.className = 'history-item';
        if (state === "VERY_DROWSY") li.classList.add('danger');
        else if (["DROWSY", "AWAY", "NO_FACE"].includes(state)) li.classList.add('warn');
        else li.classList.add('info');
        
        li.innerHTML = `<span class="history-time">${timestamp}</span> <strong>${state.replace('_', ' ')}</strong> - ${message}`;
        historyList.prepend(li);
    }

    btnExport.addEventListener('click', () => {
        if (sessionHistory.length === 0) {
            alert("No events to export yet!");
            return;
        }
        
        const csvRows = ['Time,State,Message'];
        for (const row of sessionHistory) {
            const escapedMsg = row.message.replace(/"/g, '""');
            csvRows.push(`"${row.time}","${row.state}","${escapedMsg}"`);
        }
        
        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', 'drowsyguard_session_history.csv');
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // Initialize
    connectWebSocket();
});
