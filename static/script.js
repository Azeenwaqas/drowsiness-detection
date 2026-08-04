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

    const icons = {
        "ALERT": "🟢",
        "DROWSY": "🟡",
        "VERY_DROWSY": "🔴",
        "AWAY": "🔵",
        "NO_FACE": "⚫",
        "STOPPED": "⚪"
    };

    let ws = null;
    let isMonitoring = false;

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected. Reconnecting...');
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
            alertBox.className = 'alert-box alert-danger';
            alertBox.innerHTML = `🚨 ${alertMsg}`;
        } else if (["DROWSY", "NO_FACE", "AWAY"].includes(state) && alertMsg) {
            alertBox.className = 'alert-box alert-warn';
            alertBox.innerHTML = `⚠️ ${alertMsg}`;
        } else if (state === "STOPPED") {
            alertBox.className = 'alert-box alert-ok';
            alertBox.innerHTML = `ℹ️ System stopped.`;
        } else {
            alertBox.className = 'alert-box alert-ok';
            alertBox.innerHTML = `✅ Driver Alert — All Good`;
        }

        // Update metrics
        if (data.eyes !== undefined) metricEyes.innerText = data.eyes.toFixed(3);
        if (data.mar !== undefined) metricMar.innerText = data.mar.toFixed(3);
        if (data.perclos !== undefined) metricPerclos.innerText = `${Math.round(data.perclos * 100)}%`;
        if (data.yawn_count !== undefined) metricYawns.innerText = data.yawn_count;
        if (data.session_duration !== undefined) metricSession.innerText = `${data.session_duration}s`;
        if (data.ml_label !== undefined) metricMl.innerText = `${data.ml_label} ${Math.round((data.confidence || 0) * 100)}%`;
    }

    btnStart.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send('start');
        }
        
        // Add random query param to bypass cache
        videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
        videoFeed.classList.remove('hidden');
        videoPlaceholder.classList.add('hidden');
        
        isMonitoring = true;
    });

    btnStop.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send('stop');
        }
        
        videoFeed.src = '';
        videoFeed.classList.add('hidden');
        videoPlaceholder.classList.remove('hidden');
        
        isMonitoring = false;
        
        updateDashboard({
            state: "STOPPED",
            alert_msg: "Monitoring Stopped",
            eyes: 0, mar: 0, perclos: 0, yawn_count: 0, session_duration: 0, ml_label: "-", confidence: 0
        });
    });

    // Initialize
    connectWebSocket();
});
