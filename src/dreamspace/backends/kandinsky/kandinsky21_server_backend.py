"""Kandinsky 2.1 server backend."""

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config, ModelConfig


class Kandinsky21ServerBackend(ImgGenBackend):
    """Kandinsky 2.1 server backend using AutoPipeline.
    
    Optimized for server deployment with CPU offloading and memory efficiency.
    Uses the AutoPipeline approach for simplified model loading.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 device: Optional[str] = None):
        """Initialize the Kandinsky 2.1 server backend.
        
        Args:
            config: Configuration instance
            device: Device to run on (overrides config)
        """
        self.config = config or Config()
        self.device = device or "cuda"
        self.model_id = "kandinsky-community/kandinsky-2-1"
        self._load_pipelines()
    
    def _load_pipelines(self):
        """Load the Kandinsky pipelines using AutoPipeline."""
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        
        print(f"🔮 Loading Kandinsky 2.1 from {self.model_id} on {self.device}...")
        
        # Load text-to-image pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        
        # Move pipeline to specified device
        self.pipe = self.pipe.to(self.device)
        print(f"  📍 Text2Image pipeline moved to {self.device}")
        
        # Only enable CPU offload for single GPU setups
        # For multi-GPU, keep models on their assigned GPUs
        if self.device == "cuda" or self.device == "cuda:0":
            self.pipe.enable_model_cpu_offload()
            print(f"  💾 CPU offload enabled for {self.device}")
        else:
            print(f"  🎯 Multi-GPU mode: keeping pipeline on {self.device} (no CPU offload)")
        
        # Load image-to-image pipeline  
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        
        # Move pipeline to specified device
        self.img2img_pipe = self.img2img_pipe.to(self.device)
        print(f"  📍 Image2Image pipeline moved to {self.device}")
        
        if self.device == "cuda" or self.device == "cuda:0":
            self.img2img_pipe.enable_model_cpu_offload()
            print(f"  💾 CPU offload enabled for {self.device}")
        else:
            print(f"  🎯 Multi-GPU mode: keeping pipeline on {self.device} (no CPU offload)")
        
        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory optimization enabled")
        except Exception:
            print("⚠️ XFormers not available, using default attention")
        
        print(f"✅ Kandinsky 2.1 loaded successfully on {self.device}!")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        # Set default generator for reproducibility on the correct device
        if 'generator' not in kwargs and 'seed' in kwargs:
            seed = kwargs.pop('seed')
            # Create generator on the same device as the pipeline
            device = self.device if hasattr(self, 'device') else 'cuda'
            kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)
        
        # Check if batch generation is requested
        num_images = kwargs.get('num_images_per_prompt', 1)
        print(f"🎯 Kandinsky21 server backend on {self.device}: generating {num_images} images in parallel from same generator state")
        
        result = self.pipe(prompt, return_dict=True, **kwargs)
        
        # Handle single or multiple images correctly
        images = result.images
        print(f"🖼️ Generated {len(images)} images in parallel on {self.device}")
        
        # Always return the full list of images for batch processing
        return {
            'image': images,  # Return the full list, not just the first image
            'latents': getattr(result, 'latents', None),
            'embeddings': self._extract_embeddings(prompt)
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
            'embeddings': self._extract_embeddings(prompt)
        }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        """Interpolate between two embeddings.
        
        Kandinsky supports better semantic interpolation than SD models.
        """
        if embedding1 is None or embedding2 is None:
            return None
        return self._slerp(embedding1, embedding2, alpha)
    
    def _extract_embeddings(self, prompt: str) -> torch.Tensor:
        """Extract embeddings from prompt.
        
        For Kandinsky 2.1, this extracts the image embeddings from the prior.
        """
        try:
            # Kandinsky 2.1 uses a prior model for text-to-image embeddings
            if hasattr(self.pipe, 'prior'):
                with torch.no_grad():
                    # Get image embeddings from the prior
                    prior_output = self.pipe.prior(prompt)
                    if hasattr(prior_output, 'image_embeds'):
                        return prior_output.image_embeds
                    elif isinstance(prior_output, tuple):
                        return prior_output[0]  # First element is usually image_embeds
            
            # Fallback to text encoder if available
            if hasattr(self.pipe, 'text_encoder') and hasattr(self.pipe, 'tokenizer'):
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
            
        except Exception as e:
            print(f"Warning: Could not extract embeddings: {e}")
            return None
        
        return None
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'img2img_pipe'):
            del self.img2img_pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
