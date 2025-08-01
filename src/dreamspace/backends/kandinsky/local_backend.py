"""Local Kandinsky backend implementation."""

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config, ModelConfig


class LocalKandinskyBackend(ImgGenBackend):
    """Local Kandinsky backend using Hugging Face diffusers.
    
    This backend provides local inference for Kandinsky models with support
    for text-to-image and image-to-image generation. Kandinsky uses a 
    separate prior model for better semantic understanding and interpolation.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 model_name: str = "kandinsky_2_2",
                 device: Optional[str] = None):
        """Initialize the Kandinsky backend.
        
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
                model_id="kandinsky-community/kandinsky-2-2-decoder",
                device=device or "cuda",
                torch_dtype="float16"
            )
        
        if device:
            self.model_config.device = device
        
        self.device = self.model_config.device
        self._load_pipelines()
    
    def _load_pipelines(self):
        """Load the Kandinsky pipelines."""
        from diffusers import (
            KandinskyV22Pipeline, 
            KandinskyV22Img2ImgPipeline, 
            KandinskyV22PriorPipeline
        )
        
        # Determine torch dtype
        torch_dtype = getattr(torch, self.model_config.torch_dtype)
        
        # Load prior pipeline for text-to-image embeddings
        self.prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch_dtype,
            **self.model_config.custom_params
        ).to(self.device)
        
        # Load text-to-image pipeline
        self.pipe = KandinskyV22Pipeline.from_pretrained(
            self.model_config.model_id,
            torch_dtype=torch_dtype,
            **self.model_config.custom_params
        ).to(self.device)
        
        # Load image-to-image pipeline
        self.img2img_pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
            self.model_config.model_id,
            torch_dtype=torch_dtype,
            **self.model_config.custom_params
        ).to(self.device)
        
        # Enable memory optimizations
        try:
            self.prior_pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.prior_pipe.enable_model_cpu_offload()
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
        print(f"🎯 Kandinsky backend: num_images_per_prompt = {num_images}")
        
        # Generate image embeddings from text
        prior_result = self.prior_pipe(prompt, return_dict=True)
        image_embeds = prior_result.image_embeds
        negative_embeds = prior_result.negative_image_embeds
        
        print(f"📐 Original embedding shape: {image_embeds.shape}")
        
        # For batch generation, repeat embeddings
        if num_images > 1:
            image_embeds = image_embeds.repeat(num_images, 1)
            negative_embeds = negative_embeds.repeat(num_images, 1)
            print(f"🔄 Repeated embedding shape: {image_embeds.shape}")
        
        # Filter out parameters that Kandinsky pipeline doesn't accept
        pipe_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_images_per_prompt']}
        
        # Generate image from embeddings
        result = self.pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_embeds,
            return_dict=True,
            **pipe_kwargs
        )
        
        # Return single image or list based on num_images_per_prompt
        images = result.images
        print(f"🖼️ Generated {len(images)} images")
        return_image = images[0] if num_images == 1 else images
        
        return {
            'image': return_image,
            'latents': None,  # Kandinsky doesn't typically expose latents
            'embeddings': image_embeds
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
        
        # Generate image embeddings from text
        prior_result = self.prior_pipe(prompt, return_dict=True)
        image_embeds = prior_result.image_embeds
        negative_embeds = prior_result.negative_image_embeds
        
        # Transform image using embeddings
        result = self.img2img_pipe(
            image=image,
            image_embeds=image_embeds,
            negative_image_embeds=negative_embeds,
            strength=strength,
            return_dict=True,
            **kwargs
        )
        
        return {
            'image': result.images[0],
            'latents': None,
            'embeddings': image_embeds
        }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        """Interpolate between two image embeddings.
        
        Args:
            embedding1: First image embedding
            embedding2: Second image embedding
            alpha: Interpolation factor (0.0-1.0)
            
        Returns:
            Interpolated embedding
        """
        if embedding1 is None or embedding2 is None:
            return None
        
        return self._slerp(embedding1, embedding2, alpha)
    
    def generate_from_embeddings(self, image_embeds: torch.Tensor, 
                                negative_embeds: Optional[torch.Tensor] = None,
                                **kwargs) -> Image.Image:
        """Generate image directly from embeddings.
        
        This method allows generation from interpolated embeddings without
        going through the prior pipeline again.
        
        Args:
            image_embeds: Image embeddings tensor
            negative_embeds: Optional negative embeddings
            **kwargs: Generation parameters
            
        Returns:
            Generated PIL Image
        """
        if negative_embeds is None:
            # Create zero negative embeddings if not provided
            negative_embeds = torch.zeros_like(image_embeds)
        
        result = self.pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_embeds,
            return_dict=True,
            **kwargs
        )
        
        return result.images[0]
    
    def get_text_embeddings(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get text embeddings for a prompt.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Tuple of (image_embeds, negative_embeds)
        """
        prior_result = self.prior_pipe(prompt, return_dict=True)
        return prior_result.image_embeds, prior_result.negative_image_embeds
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'prior_pipe'):
            del self.prior_pipe
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'img2img_pipe'):
            del self.img2img_pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
