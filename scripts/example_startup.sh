#!/bin/bash

# Example startup script for Dreamspace Co-Pilot Server
# This script shows how to start the server with common configurations

echo "🚀 Starting Dreamspace Co-Pilot Server..."

# Configuration - modify these as needed
BACKEND="kandinsky21_server"  # Options: kandinsky21_server, sd15_server, sd21_server
HOST="0.0.0.0"               # Use 0.0.0.0 for external access, localhost for local only
PORT="8000"                  # Port to run on
WORKERS="1"                  # Number of workers (usually 1 for GPU models)

# Optional authentication - uncomment to enable
# AUTH_ENABLED="--auth"
# API_KEY="--api-key your-secret-key-here"

# Get the script directory to ensure we're in the right place
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📁 Project directory: $PROJECT_DIR"
echo "🔧 Backend: $BACKEND"
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"

# Change to project directory
cd "$PROJECT_DIR" || {
    echo "❌ Could not change to project directory: $PROJECT_DIR"
    exit 1
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Python package is installed
python -c "import dreamspace" 2>/dev/null || {
    echo "❌ Dreamspace package not found. Installing..."
    pip install -e .
}

# Start the server
echo "🚀 Starting server..."
python scripts/start_server.py \
    --backend "$BACKEND" \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    $AUTH_ENABLED \
    $API_KEY

# Note: To run this script:
# 1. Make it executable: chmod +x scripts/example_startup.sh
# 2. Run it: ./scripts/example_startup.sh
#
# To run different models on different ports:
# - Kandinsky 2.1: ./scripts/example_startup.sh (default)
# - Edit the script and change BACKEND and PORT variables
# - Or run multiple instances with different configs
