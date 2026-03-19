#!/usr/bin/env bash
# Usage: ./app.sh {start|stop|restart|status}

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_CMD="uv run streamlit run src/dashboard/app.py --server.port 8501"
PID_FILE="$APP_DIR/.streamlit.pid"
LOG_FILE="$APP_DIR/.streamlit.log"

start() {
    if is_running; then
        echo "Already running (PID $(cat "$PID_FILE"))"
        return 1
    fi
    cd "$APP_DIR"
    nohup $APP_CMD > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if is_running; then
        echo "Started (PID $(cat "$PID_FILE"))"
        echo "URL: http://localhost:8501"
    else
        echo "Failed to start — check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if ! is_running; then
        echo "Not running"
        # Clean up any orphaned streamlit processes
        pkill -f "streamlit run src/dashboard/app.py" 2>/dev/null
        rm -f "$PID_FILE"
        return 0
    fi
    local pid=$(cat "$PID_FILE")
    kill "$pid" 2>/dev/null
    # Also kill child python process
    pkill -P "$pid" 2>/dev/null
    sleep 1
    # Force kill if still alive
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null
        pkill -9 -f "streamlit run src/dashboard/app.py" 2>/dev/null
        sleep 1
    fi
    rm -f "$PID_FILE"
    echo "Stopped"
}

restart() {
    stop
    start
}

status() {
    if is_running; then
        echo "Running (PID $(cat "$PID_FILE"))"
        echo "URL: http://localhost:8501"
    else
        echo "Not running"
    fi
}

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
