"""Configuration management for Dreamspace Co-Pilot."""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    device: str = "cuda"
    torch_dtype: str = "float16"
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class GenerationConfig:
    """Default generation parameters."""
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    width: int = 512
    height: int = 512
    scheduler: Optional[str] = None


@dataclass
class ServerConfig:
    """Configuration for server deployment."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 4
    timeout: int = 300
    enable_cors: bool = True


@dataclass
class RemoteConfig:
    """Configuration for remote API access."""
    api_url: str
    api_key: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


class Config:
    """Main configuration class for Dreamspace Co-Pilot."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.models: Dict[str, ModelConfig] = {}
        self.generation = GenerationConfig()
        self.server = ServerConfig()
        self.remote: Optional[RemoteConfig] = None
        
        # Load default configurations
        self._load_defaults()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_defaults(self):
        """Load default model configurations."""
        self.models = {
            "stable_diffusion_v1_5": ModelConfig(
                model_id="runwayml/stable-diffusion-v1-5",
                device="cuda",
                torch_dtype="float16"
            ),
            "kandinsky_2_2": ModelConfig(
                model_id="kandinsky-community/kandinsky-2-2-decoder",
                device="cuda", 
                torch_dtype="float16"
            )
        }
    
    def load_from_file(self, config_path: Union[str, Path]):
        """Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        self._update_from_dict(data)
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary."""
        if 'models' in data:
            for name, model_data in data['models'].items():
                self.models[name] = ModelConfig(**model_data)
        
        if 'generation' in data:
            self.generation = GenerationConfig(**data['generation'])
        
        if 'server' in data:
            self.server = ServerConfig(**data['server'])
        
        if 'remote' in data:
            self.remote = RemoteConfig(**data['remote'])
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Server configuration
        if os.getenv('DREAMSPACE_HOST'):
            self.server.host = os.getenv('DREAMSPACE_HOST')
        if os.getenv('DREAMSPACE_PORT'):
            self.server.port = int(os.getenv('DREAMSPACE_PORT'))
        
        # Remote API configuration
        if os.getenv('DREAMSPACE_API_URL'):
            if not self.remote:
                self.remote = RemoteConfig(api_url=os.getenv('DREAMSPACE_API_URL'))
            else:
                self.remote.api_url = os.getenv('DREAMSPACE_API_URL')
        
        if os.getenv('DREAMSPACE_API_KEY'):
            if not self.remote:
                self.remote = RemoteConfig(
                    api_url=os.getenv('DREAMSPACE_API_URL', ''),
                    api_key=os.getenv('DREAMSPACE_API_KEY')
                )
            else:
                self.remote.api_key = os.getenv('DREAMSPACE_API_KEY')
        
        # Device configuration
        if os.getenv('DREAMSPACE_DEVICE'):
            device = os.getenv('DREAMSPACE_DEVICE')
            for model_config in self.models.values():
                model_config.device = device
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model.
        
        Args:
            model_name: Name of the model configuration
            
        Returns:
            ModelConfig instance
            
        Raises:
            KeyError: If model configuration not found
        """
        if model_name not in self.models:
            raise KeyError(f"Model configuration '{model_name}' not found")
        return self.models[model_name]
    
    def add_model_config(self, name: str, config: ModelConfig):
        """Add a new model configuration.
        
        Args:
            name: Name for the model configuration
            config: ModelConfig instance
        """
        self.models[name] = config
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save current configuration to file.
        
        Args:
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        
        data = {
            'models': {name: asdict(config) for name, config in self.models.items()},
            'generation': asdict(self.generation),
            'server': asdict(self.server)
        }
        
        if self.remote:
            data['remote'] = asdict(self.remote)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        data = {
            'models': {name: asdict(config) for name, config in self.models.items()},
            'generation': asdict(self.generation),
            'server': asdict(self.server)
        }
        
        if self.remote:
            data['remote'] = asdict(self.remote)
        
        return data
