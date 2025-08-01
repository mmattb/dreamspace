#!/bin/bash

# Start Kandinsky 2.1 server (recommended for smooth interpolation)
echo "🔮 Starting Kandinsky 2.1 Server on port 8000..."

cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python scripts/start_server.py \
    --backend kandinsky21_server \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
