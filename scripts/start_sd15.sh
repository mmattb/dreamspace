#!/bin/bash

# Start Stable Diffusion 1.5 server
echo "🎨 Starting Stable Diffusion 1.5 Server on port 8001..."

cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python scripts/start_server.py \
    --backend sd15_server \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1
