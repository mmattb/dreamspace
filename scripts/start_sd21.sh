#!/bin/bash

# Start Stable Diffusion 2.1 server
echo "🎨 Starting Stable Diffusion 2.1 Server on port 8002..."

cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python scripts/start_server.py \
    --backend sd21_server \
    --host 0.0.0.0 \
    --port 8002 \
    --workers 1
