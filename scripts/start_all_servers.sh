#!/bin/bash

# Start all servers (multi-server setup)
echo "🎨 Starting Multi-Server Dreamspace Setup..."

cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting Stable Diffusion 1.5 server on port 8001..."
python scripts/start_server.py --backend sd15_server --host 0.0.0.0 --port 8001 --workers 1 &
SD15_PID=$!

echo "Starting Stable Diffusion 2.1 server on port 8002..."  
python scripts/start_server.py --backend sd21_server --host 0.0.0.0 --port 8002 --workers 1 &
SD21_PID=$!

echo "Starting Kandinsky 2.1 server on port 8003..."
python scripts/start_server.py --backend kandinsky21_server --host 0.0.0.0 --port 8003 --workers 1 &
KANDINSKY_PID=$!

echo ""
echo "🚀 All servers started!"
echo "   SD 1.5:    http://localhost:8001"
echo "   SD 2.1:    http://localhost:8002" 
echo "   Kandinsky: http://localhost:8003"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all servers..."
    kill $SD15_PID $SD21_PID $KANDINSKY_PID 2>/dev/null
    wait $SD15_PID $SD21_PID $KANDINSKY_PID 2>/dev/null
    echo "All servers stopped."
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
