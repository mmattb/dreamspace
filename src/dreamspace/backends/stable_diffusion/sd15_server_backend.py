"""Stable Diffusion 1.5 server backend."""

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config, ModelConfig


class StableDiffusion15ServerBackend(ImgGenBackend):
    """Stable Diffusion 1.5 server backend using AutoPipeline.
    
    Optimized for server deployment with CPU offloading and memory efficiency.
    Uses the AutoPipeline approach for simplified model loading.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 device: Optional[str] = None):
        """Initialize the SD 1.5 server backend.
        
        Args:
            config: Configuration instance
            device: Device to run on (overrides config)
        """
        self.config = config or Config()
        self.device = device or "cuda"
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self._load_pipelines()
    
    def _load_pipelines(self):
        """Load the diffusion pipelines using AutoPipeline."""
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        
        print(f"🔮 Loading Stable Diffusion 1.5 from {self.model_id}...")
        
        # Load text-to-image pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        self.pipe.enable_model_cpu_offload()
        
        # Load image-to-image pipeline
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        self.img2img_pipe.enable_model_cpu_offload()
        
        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory optimization enabled")
        except Exception:
            print("⚠️ XFormers not available, using default attention")
        
        print("✅ Stable Diffusion 1.5 loaded successfully!")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        # Set default generator for reproducibility
        if 'generator' not in kwargs and 'seed' in kwargs:
            kwargs['generator'] = torch.Generator().manual_seed(kwargs.pop('seed'))
        
        # Check if batch generation is requested
        num_images = kwargs.get('num_images_per_prompt', 1)
        print(f"🎯 SD15 server backend: num_images_per_prompt = {num_images}")
        
        result = self.pipe(prompt, return_dict=True, **kwargs)
        
        # Handle single or multiple images
        images = result.images
        print(f"🖼️ Generated {len(images)} images")
        
        return_image = images[0] if num_images == 1 else images
        
        return {
            'image': return_image,
            'latents': getattr(result, 'latents', None),
            'embeddings': self._extract_text_embeddings(prompt)
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Transform an existing image using a text prompt."""
        # Set default generator for reproducibility
        if 'generator' not in kwargs and 'seed' in kwargs:
            kwargs['generator'] = torch.Generator().manual_seed(kwargs.pop('seed'))
        
        result = self.img2img_pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            return_dict=True,
            **kwargs
        )
        
        return {
            'image': result.images[0],
            'latents': getattr(result, 'latents', None),
            'embeddings': self._extract_text_embeddings(prompt)
        }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        """Interpolate between two text embeddings."""
        if embedding1 is None or embedding2 is None:
            return None
        return self._slerp(embedding1, embedding2, alpha)
    
    def _extract_text_embeddings(self, prompt: str) -> torch.Tensor:
        """Extract text embeddings from prompt."""
        try:
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                text_embeddings = self.pipe.text_encoder(
                    text_inputs.input_ids.to(self.pipe.device)
                )[0]
            
            return text_embeddings
        except Exception:
            return None
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'img2img_pipe'):
            del self.img2img_pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
