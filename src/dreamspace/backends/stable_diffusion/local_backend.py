"""Local Stable Diffusion backend implementation."""

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config, ModelConfig


class LocalStableDiffusionBackend(ImgGenBackend):
    """Local Stable Diffusion backend using Hugging Face diffusers.
    
    This backend provides local inference for Stable Diffusion models with
    support for text-to-image and image-to-image generation. It handles
    model loading, pipeline management, and CUDA optimization.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 model_name: str = "stable_diffusion_v1_5",
                 device: Optional[str] = None):
        """Initialize the Stable Diffusion backend.
        
        Args:
            config: Configuration instance
            model_name: Name of model configuration to use
            device: Device to run on (overrides config)
        """
        self.config = config or Config()
        
        # Get model configuration
        try:
            self.model_config = self.config.get_model_config(model_name)
        except KeyError:
            # Fallback to default if not found
            self.model_config = ModelConfig(
                model_id="runwayml/stable-diffusion-v1-5",
                device=device or "cuda",
                torch_dtype="float16"
            )
        
        if device:
            self.model_config.device = device
        
        self.device = self.model_config.device
        self._load_pipelines()
    
    def _load_pipelines(self):
        """Load the diffusion pipelines."""
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        
        # Determine torch dtype
        torch_dtype = getattr(torch, self.model_config.torch_dtype)
        
        # Load text-to-image pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_config.model_id,
            torch_dtype=torch_dtype,
            **self.model_config.custom_params
        ).to(self.device)
        
        # Load image-to-image pipeline (shares components with text2img)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_config.model_id,
            torch_dtype=torch_dtype,
            **self.model_config.custom_params
        ).to(self.device)
        
        # Enable memory efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers not available, use regular attention
            pass
        
        # Enable CPU offload for memory efficiency
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.pipe.enable_model_cpu_offload()
                self.img2img_pipe.enable_model_cpu_offload()
            except Exception:
                pass
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            **kwargs: Generation parameters
            
        Returns:
            Dict with 'image', 'latents', and 'embeddings'
        """
        # Set default generator for reproducibility
        if 'generator' not in kwargs and 'seed' in kwargs:
            kwargs['generator'] = torch.Generator(self.device).manual_seed(kwargs.pop('seed'))
        
        # Check if batch generation is requested
        num_images = kwargs.get('num_images_per_prompt', 1)
        print(f"🎯 SD backend: num_images_per_prompt = {num_images}")
        
        result = self.pipe(prompt, return_dict=True, **kwargs)
        
        # Handle single or multiple images
        images = result.images
        print(f"🖼️ Generated {len(images)} images")
        
        return_image = images[0] if num_images == 1 else images
        
        # Debugging: Log the type and structure of latents
        latents = getattr(result, 'latents', None)
        if latents is not None:
            print(f"🔍 Latents type: {type(latents)}, Latents shape: {getattr(latents, 'shape', 'N/A')}")

        return {
            'image': return_image,
            'latents': latents,
            'embeddings': self._extract_text_embeddings(prompt)
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Transform an existing image using a text prompt.
        
        Args:
            image: Source image to transform
            prompt: Text description guiding the transformation
            strength: How much to change the image (0.0-1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with 'image', 'latents', and 'embeddings'
        """
        # Set default generator for reproducibility
        if 'generator' not in kwargs and 'seed' in kwargs:
            kwargs['generator'] = torch.Generator(self.device).manual_seed(kwargs.pop('seed'))
        
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
        """Interpolate between two text embeddings.
        
        Args:
            embedding1: First text embedding
            embedding2: Second text embedding
            alpha: Interpolation factor (0.0-1.0)
            
        Returns:
            Interpolated embedding
        """
        if embedding1 is None or embedding2 is None:
            return None
        
        return self._slerp(embedding1, embedding2, alpha)
    
    def _extract_text_embeddings(self, prompt: str) -> torch.Tensor:
        """Extract text embeddings from prompt.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Text embeddings tensor
        """
        try:
            # Tokenize and encode the prompt
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                text_embeddings = self.pipe.text_encoder(
                    text_inputs.input_ids.to(self.device)
                )[0]
            
            return text_embeddings
        except Exception:
            # If extraction fails, return None
            return None
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'img2img_pipe'):
            del self.img2img_pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
