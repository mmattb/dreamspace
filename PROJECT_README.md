# Dreamspace Co-Pilot: AI + BCI Co-Piloted Visual Dreamspace

A hybrid Brain-Computer Interface (BCI) and Artificial Intelligence (AI) system that enables users to co-pilot and explore imaginative visual experiences in a way that feels like a dream: semi-intentional, emotionally responsive, yet partially out of one's control.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dreamspaceCopilot

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from dreamspace import ImgGen

# Initialize with Kandinsky backend (recommended for smooth interpolation)
img_gen = ImgGen("kandinsky_local", prompt="a surreal dreamlike forest")

# Generate initial image
image = img_gen.gen()

# Evolve the image with visual continuity
evolved_image = img_gen.gen_img2img(strength=0.3, prompt="a mystical forest with glowing lights")
```

### Run Examples

```bash
# Keyboard navigation example (requires pygame)
python examples/keyboard_navigation.py

# Remote API example
python examples/remote_api_example.py
```

## 📁 Project Structure

```
dreamspaceCopilot/
├── src/dreamspace/              # Main package
│   ├── core/                    # Core components
│   │   ├── base.py             # Abstract backend interface
│   │   └── image_gen.py        # Main ImgGen class
│   ├── backends/                # Model backends
│   │   ├── stable_diffusion/   # Stable Diffusion backend
│   │   ├── kandinsky/          # Kandinsky backend
│   │   └── remote/             # Remote API backend
│   ├── servers/                 # Server implementations
│   │   └── api_server.py       # FastAPI server
│   └── config/                  # Configuration management
│       └── settings.py         # Config classes
├── examples/                    # Usage examples
│   ├── keyboard_navigation.py  # Interactive keyboard control
│   └── remote_api_example.py   # Remote API usage
├── scripts/                     # Utility scripts
│   └── start_server.py         # Server startup script
├── config/                      # Configuration files
│   └── default.yaml            # Default configuration
├── requirements.txt             # Python dependencies
└── setup.py                    # Package setup
```

## 🎯 Key Features

### 🔮 Latent-Space Continuity
Images evolve gradually to maintain narrative and visual cohesion through img2img generation with configurable strength parameters.

### 🧠 Multiple Backends
- **Kandinsky 2.2**: Best for semantic interpolation and smooth transitions
- **Stable Diffusion**: Wide model variety and community support  
- **Remote API**: Scalable server deployment

### ✨ Semantic Interpolation
Smooth transitions between concepts using spherical linear interpolation (slerp) in embedding space.

### 🔧 Configuration Management
Flexible YAML/JSON configuration with environment variable overrides.

### 🌐 Server Deployment
Production-ready FastAPI server with authentication, CORS, and batch processing.

## 🎮 Interactive Navigation

The keyboard navigation example demonstrates smooth image space exploration:

```bash
python examples/keyboard_navigation.py
```

**Controls:**
- `←→` Adjust transformation strength
- `↑↓` Navigate through variations
- `Space` Add random effects
- `R` Reset effects
- `Escape` Exit

## 🖥️ Server Deployment

### Start Local Server

```bash
# Basic server
python scripts/start_server.py --backend kandinsky_local --port 8000

# Production server with authentication
python scripts/start_server.py \
  --backend kandinsky_local \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --auth \
  --api-key your-secret-key
```

### API Endpoints

- `POST /generate` - Text-to-image generation
- `POST /img2img` - Image-to-image transformation  
- `POST /interpolate` - Embedding interpolation
- `GET /health` - Health check
- `GET /models` - Model information

### Using Remote API

```python
from dreamspace import ImgGen
from dreamspace.config.settings import Config, RemoteConfig

config = Config()
config.remote = RemoteConfig(
    api_url="http://your-server:8000",
    api_key="your-api-key"
)

img_gen = ImgGen("remote", config=config)
image = img_gen.gen("a magical landscape")
```

## ⚙️ Configuration

### YAML Configuration

```yaml
# config/custom.yaml
models:
  kandinsky_2_2:
    model_id: "kandinsky-community/kandinsky-2-2-decoder"
    device: "cuda"
    torch_dtype: "float16"

generation:
  guidance_scale: 7.5
  num_inference_steps: 50
  width: 512
  height: 512

server:
  host: "0.0.0.0"
  port: 8000
  workers: 2
```

### Environment Variables

```bash
export DREAMSPACE_DEVICE=cuda
export DREAMSPACE_API_URL=https://your-api.com
export DREAMSPACE_API_KEY=your-key
```

### Programmatic Configuration

```python
from dreamspace.config.settings import Config, ModelConfig

config = Config()
config.add_model_config("custom_sd", ModelConfig(
    model_id="your-custom/stable-diffusion-model",
    device="cuda",
    custom_params={"scheduler": "DPMSolverMultistepScheduler"}
))
```

## 🔧 Development

### Adding New Backends

1. Create backend class inheriting from `ImgGenBackend`
2. Implement required methods: `generate()`, `img2img()`, `interpolate_embeddings()`
3. Register in `ImgGen._create_backend()`

```python
from dreamspace.core.base import ImgGenBackend

class CustomBackend(ImgGenBackend):
    def generate(self, prompt: str, **kwargs):
        # Implementation
        pass
    
    def img2img(self, image, prompt: str, strength: float = 0.5, **kwargs):
        # Implementation  
        pass
    
    def interpolate_embeddings(self, emb1, emb2, alpha: float):
        # Implementation
        pass
```

### Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/
isort src/

# Type checking
mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Diffusers team for excellent model implementations
- Kandinsky and Stable Diffusion model creators
- The open-source AI community

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Make sure to install in development mode: `pip install -e .`

**CUDA Memory**: Reduce batch size or use CPU: `--device cpu`

**Server Not Starting**: Check port availability: `netstat -tulpn | grep 8000`

**Model Loading Fails**: Verify internet connection and disk space for model downloads

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review example code in `examples/`
- Read configuration documentation in `config/`
