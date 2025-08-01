# 🚀 Dreamspace Co-Pilot Server Setup Guide

This guide explains how to set up and run the Dreamspace Co-Pilot API server on your console-only server.

## 📋 Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for models
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Dependencies
```bash
# Install Python 3.8+ and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Install CUDA (if not already installed)
# Follow NVIDIA's official guide for your system
```

## 🛠️ Installation

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url> dreamspace-copilot
cd dreamspace-copilot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Verify GPU Setup
```bash
# Check if PyTorch can see your GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 🎯 Available Models

You can run servers for different models:

### **Kandinsky 2.1** (Recommended)
- **Backend**: `kandinsky21_server`
- **Model**: `kandinsky-community/kandinsky-2-1`
- **Best for**: Semantic interpolation, smooth transitions
- **VRAM**: ~6-8GB

### **Stable Diffusion 1.5**
- **Backend**: `sd15_server` 
- **Model**: `runwayml/stable-diffusion-v1-5`
- **Best for**: General purpose, wide compatibility
- **VRAM**: ~4-6GB

### **Stable Diffusion 2.1**
- **Backend**: `sd21_server`
- **Model**: `stabilityai/stable-diffusion-2-1` 
- **Best for**: Higher quality, 768x768 images
- **VRAM**: ~6-8GB

## 🚀 Starting the Server

### Basic Server (Kandinsky 2.1)
```bash
# Activate virtual environment
source venv/bin/activate

# Start server on localhost
python scripts/start_server.py --backend kandinsky21_server --port 8000

# Or run directly
cd src && python -m dreamspace.servers.api_server --backend kandinsky21_server --port 8000
```

### Production Server (External Access)
```bash
# Start server accessible from outside
python scripts/start_server.py \
  --backend kandinsky21_server \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1

# With authentication
python scripts/start_server.py \
  --backend kandinsky21_server \
  --host 0.0.0.0 \
  --port 8000 \
  --auth \
  --api-key "your-secret-api-key-here"
```

### Different Models
```bash
# Stable Diffusion 1.5
python scripts/start_server.py --backend sd15_server --port 8001

# Stable Diffusion 2.1  
python scripts/start_server.py --backend sd21_server --port 8002

# Run multiple models on different ports
python scripts/start_server.py --backend kandinsky21_server --port 8000 &
python scripts/start_server.py --backend sd15_server --port 8001 &
python scripts/start_server.py --backend sd21_server --port 8002 &
```

## 🔧 Configuration Options

### Command Line Arguments
```bash
python scripts/start_server.py --help

# Key options:
--backend         # Model backend (kandinsky21_server, sd15_server, sd21_server)
--host           # Host to bind to (localhost, 0.0.0.0)  
--port           # Port number (8000, 8001, etc.)
--workers        # Number of worker processes (1 recommended for GPU)
--auth           # Enable API key authentication
--api-key        # API key for authentication
--device         # Force device (cuda, cpu)
--config         # Path to config file
```

### Environment Variables
```bash
# Set environment variables
export DREAMSPACE_DEVICE=cuda
export DREAMSPACE_HOST=0.0.0.0
export DREAMSPACE_PORT=8000
export DREAMSPACE_API_KEY=your-secret-key

# Then start server
python scripts/start_server.py --backend kandinsky21_server
```

## 🌐 Testing the Server

### Health Check
```bash
# Test if server is running
curl http://localhost:8000/health

# With authentication
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/health
```

### Generate Test Image
```bash
# Simple text-to-image
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a magical forest with glowing mushrooms"}' \
  | jq '.image' | sed 's/"//g' | base64 -d > test_image.png

# With authentication
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"prompt": "a magical forest with glowing mushrooms"}' \
  | jq '.image' | sed 's/"//g' | base64 -d > test_image.png
```

## 🔍 Monitoring and Logs

### View Server Logs
```bash
# Server logs will show:
# ✅ Model loaded successfully: kandinsky21_server
# 🚀 Starting Dreamspace Co-Pilot Server
# INFO: Started server process
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### Monitor GPU Usage
```bash
# Install nvidia-ml-py if not available
pip install nvidia-ml-py3

# Monitor GPU usage
watch -n 1 nvidia-smi

# Or use htop for CPU/RAM
htop
```

### Check Server Status
```bash
# Check if port is listening
netstat -tulpn | grep 8000

# Check process
ps aux | grep api_server
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find and kill process using port
sudo lsof -i :8000
sudo kill -9 <PID>
```

**Out of Memory**
```bash
# Reduce batch size or use CPU
python scripts/start_server.py --backend kandinsky21_server --device cpu
```

**Model Download Issues**
```bash
# Pre-download models
python -c "from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('kandinsky-community/kandinsky-2-1')"
```

**Permission Denied**
```bash
# For port 80/443, use sudo or higher port
sudo python scripts/start_server.py --backend kandinsky21_server --port 80
# Or use port > 1024
python scripts/start_server.py --backend kandinsky21_server --port 8000
```

### Debugging
```bash
# Run with debug logging
python scripts/start_server.py --backend kandinsky21_server --log-level debug

# Check Python import issues
python -c "from dreamspace import ImgGen; print('Import successful')"
```

## 🔒 Security

### Firewall Setup
```bash
# Allow specific port through firewall
sudo ufw allow 8000/tcp

# Or allow from specific IP
sudo ufw allow from YOUR_CLIENT_IP to any port 8000
```

### HTTPS (Optional)
```bash
# Use nginx as reverse proxy for HTTPS
sudo apt install nginx
# Configure nginx to proxy to your app on localhost:8000
```

## 📱 Client Usage

Once your server is running, you can connect from client code:

```python
from dreamspace import ImgGen
from dreamspace.config.settings import Config, RemoteConfig

# Configure client
config = Config()
config.remote = RemoteConfig(
    api_url="http://your-server-ip:8000",
    api_key="your-api-key"  # If authentication enabled
)

# Use remote backend
img_gen = ImgGen("remote", config=config)
image = img_gen.gen("a surreal dreamscape")
```

## 🔄 Process Management

### Using systemd (Recommended)
```bash
# Create service file
sudo tee /etc/systemd/system/dreamspace.service > /dev/null <<EOF
[Unit]
Description=Dreamspace Co-Pilot API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/dreamspace-copilot
Environment=PATH=/path/to/dreamspace-copilot/venv/bin
ExecStart=/path/to/dreamspace-copilot/venv/bin/python scripts/start_server.py --backend kandinsky21_server --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable dreamspace
sudo systemctl start dreamspace
sudo systemctl status dreamspace
```

### Using screen/tmux
```bash
# Using screen
screen -S dreamspace
python scripts/start_server.py --backend kandinsky21_server --port 8000
# Ctrl+A, D to detach

# Using tmux
tmux new-session -d -s dreamspace 'python scripts/start_server.py --backend kandinsky21_server --port 8000'
```

This should get your server up and running! The models will auto-download on first use, so the initial startup may take 5-10 minutes.
