"""Stable Diffusion 2.1 server backend."""

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config, ModelConfig


class StableDiffusion21ServerBackend(ImgGenBackend):
    """Stable Diffusion 2.1 server backend using AutoPipeline.
    
    Optimized for server deployment with CPU offloading and memory efficiency.
    Uses the AutoPipeline approach for simplified model loading.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 device: Optional[str] = None):
        """Initialize the SD 2.1 server backend.
        
        Args:
            config: Configuration instance
            device: Device to run on (overrides config)
        """
        self.config = config or Config()
        self.device = device or "cuda"
        self.model_id = "stabilityai/stable-diffusion-2-1"
        self._load_pipelines()
    
    def _load_pipelines(self):
        """Load the diffusion pipelines using AutoPipeline."""
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        
        print(f"🔮 Loading Stable Diffusion 2.1 from {self.model_id} on device {self.device}...")
        
        # Load text-to-image pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        # Only enable CPU offloading for the primary GPU (cuda:0)
        # For multi-GPU setups, we want models to stay on their assigned GPU
        if self.device == "cuda:0" or self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            print(f"  🔄 CPU offloading enabled for primary GPU: {self.device}")
        else:
            print(f"  🎯 CPU offloading disabled for multi-GPU device: {self.device}")
        
        # Load image-to-image pipeline
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        # Same CPU offloading logic for img2img pipeline
        if self.device == "cuda:0" or self.device == "cuda":
            self.img2img_pipe.enable_model_cpu_offload()
        else:
            print(f"  🎯 CPU offloading disabled for img2img on multi-GPU device: {self.device}")
        
        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory optimization enabled")
        except Exception:
            print("⚠️ XFormers not available, using default attention")
        
        print("✅ Stable Diffusion 2.1 loaded successfully!")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        # Set default generator for reproducibility - use device-specific generator
        if 'generator' not in kwargs and 'seed' in kwargs:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(kwargs.pop('seed'))
            kwargs['generator'] = generator
        
        # Check if batch generation is requested
        num_images = kwargs.get('num_images_per_prompt', 1)
        print(f"🎯 SD21 server backend on {self.device}: generating {num_images} images in parallel from same generator state")
        
        result = self.pipe(prompt, return_dict=True, **kwargs)
        
        # Handle single or multiple images correctly
        images = result.images
        print(f"🖼️ Generated {len(images)} images in parallel on device {self.device}")
        
        # Always return the full list of images for batch processing
        return {
            'image': images,  # Return the full list, not just the first image
            'latents': getattr(result, 'latents', None),
            'embeddings': self._extract_text_embeddings(prompt)
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Transform an existing image using a text prompt."""
        # Set default generator for reproducibility - use device-specific generator
        if 'generator' not in kwargs and 'seed' in kwargs:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(kwargs.pop('seed'))
            kwargs['generator'] = generator
        
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
