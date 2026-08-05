document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const videoFeed = document.getElementById('video-feed');
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
    
    // Create an offscreen canvas to extract frames
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
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
        };

        ws.onmessage = (event) => {
            const response = JSON.parse(event.data);
            
            if (response.data) {
                updateDashboard(response.data);
            }
            if (response.image) {
                // Update the video feed with the processed image from the backend
                videoFeed.src = response.image;
                
                // Ask for the next frame ONLY after we finished rendering this one!
                if (isMonitoring) {
                    requestAnimationFrame(sendNextFrame);
                }
            }
        };

        ws.onclose = () => {
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
        } else if (["DROWSY", "NO_FACE", "AWAY"].includes(state) && alertMsg) {
            alertBox.className = 'alert alert-warn';
            alertBox.innerHTML = `⚠️ ${alertMsg}`;
        } else if (state === "STOPPED") {
            alertBox.className = 'alert alert-ok';
            alertBox.innerHTML = `ℹ️ System stopped.`;
        } else {
            alertBox.className = 'alert alert-ok';
            alertBox.innerHTML = `✅ Driver Alert — All Good`;
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

    function sendNextFrame() {
        if (isMonitoring && ws && ws.readyState === WebSocket.OPEN && localStream) {
            ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);
            // Lower quality slightly to improve latency
            const base64Image = canvas.toDataURL('image/jpeg', 0.5); 
            ws.send(JSON.stringify({ type: "frame", image: base64Image }));
        }
    }

    async function startCamera() {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } } 
            });
            hiddenVideo.srcObject = localStream;
            
            // Wait for the video to start playing before sending the first frame
            hiddenVideo.onplaying = () => {
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
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'start' }));
        }
        
        videoFeed.classList.remove('hidden');
        videoPlaceholder.classList.add('hidden');
        
        isMonitoring = true;
        startCamera();
    });

    btnStop.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'stop' }));
        }
        
        videoFeed.src = '';
        videoFeed.classList.add('hidden');
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
