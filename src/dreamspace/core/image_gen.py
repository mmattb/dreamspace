"""High-level interface for image generation with continuity and interpolation support."""

import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any, Union
from collections import deque

from .base import ImgGenBackend
from ..config.settings import Config


class ImgGen:
    """High-level interface for image generation with continuity and interpolation support.
    
    ImgGen provides a unified API for different image generation models (Stable Diffusion,
    Kandinsky, remote APIs) with built-in support for visual continuity, semantic 
    interpolation, and generation history management. It's designed for applications
    that need smooth navigation through image space, such as the dreamspace co-pilot.
    
    Key Features:
        - Backend abstraction: Works with local models or remote APIs
        - Visual continuity: img2img generation maintains visual coherence
        - Semantic interpolation: Smooth transitions between concepts
        - History management: Automatic tracking of generations for continuity
        - Parameter management: Flexible generation parameter handling
    
    Example:
        >>> # Initialize with Kandinsky backend
        >>> img_gen = ImgGen("kandinsky_local", prompt="a surreal forest")
        >>> 
        >>> # Generate initial image
        >>> image1 = img_gen.gen()
        >>> 
        >>> # Evolve the image while maintaining visual continuity
        >>> image2 = img_gen.gen_img2img(strength=0.3, prompt="a mystical forest")
        >>> 
        >>> # Set up semantic interpolation
        >>> img_gen.set_interpolation_targets("forest", "ocean")
        >>> interpolated = img_gen.gen_interpolated(0.5)  # Halfway between concepts
    
    Attributes:
        prompt (str): Default prompt for generation
        backend (ImgGenBackend): The underlying generation backend
        generation_params (dict): Default parameters for all generations
        history_size (int): Maximum number of generations to keep in history
    """
    
    def __init__(self, 
                 backend: Union[str, ImgGenBackend] = "kandinsky_local",
                 prompt: Optional[str] = None,
                 history_size: int = 10,
                 config: Optional[Config] = None,
                 **backend_kwargs):
        """Initialize the ImgGen instance.
        
        Args:
            backend: Either a backend type string ("sd_local", "kandinsky_local", "remote")
                    or a pre-configured ImgGenBackend instance
            prompt: Default prompt to use for generations when none is specified
            history_size: Maximum number of recent generations to keep in memory
            config: Configuration instance, if None will create default
            **backend_kwargs: Additional arguments passed to backend constructor
                            (e.g., model_id, device, api_url, api_key)
        
        Raises:
            ValueError: If backend type string is not recognized
        """
        
        # Initialize configuration
        self.config = config or Config()
        
        # Initialize backend
        if isinstance(backend, str):
            self.backend = self._create_backend(backend, **backend_kwargs)
        else:
            self.backend = backend
        
        # Core state
        self.prompt = prompt
        self.history_size = history_size
        
        # History storage using deques for efficiency
        self._recent_images = deque(maxlen=history_size)
        self._recent_embeddings = deque(maxlen=history_size)
        self._recent_latents = deque(maxlen=history_size)
        self._recent_prompts = deque(maxlen=history_size)
        
        # Generation parameters from config
        gen_config = self.config.generation
        self.generation_params = {
            'guidance_scale': gen_config.guidance_scale,
            'num_inference_steps': gen_config.num_inference_steps,
            'width': gen_config.width,
            'height': gen_config.height
        }
        
        # Interpolation state
        self._interpolation_source = None
        self._interpolation_target = None
    
    def _create_backend(self, backend_type: str, **kwargs) -> ImgGenBackend:
        """Create a backend instance based on type string.
        
        Args:
            backend_type: Type of backend to create
            **kwargs: Additional arguments for backend
            
        Returns:
            Configured backend instance
            
        Raises:
            ValueError: If backend type is not recognized
        """
        if backend_type == "sd_local":
            # Redirect to server backend (local backend deprecated)
            print("⚠️  sd_local backend deprecated, using sd15_server instead")
            from ..backends.stable_diffusion.sd15_server_backend import StableDiffusion15ServerBackend
            return StableDiffusion15ServerBackend(config=self.config, **kwargs)
        elif backend_type == "sd15_server":
            from ..backends.stable_diffusion.sd15_server_backend import StableDiffusion15ServerBackend
            return StableDiffusion15ServerBackend(config=self.config, **kwargs)
        elif backend_type == "sd21_server":
            from ..backends.stable_diffusion.sd21_server_backend import StableDiffusion21ServerBackend
            return StableDiffusion21ServerBackend(config=self.config, **kwargs)
        elif backend_type == "kandinsky_local":
            # Redirect to server backend (local backend deprecated)
            print("⚠️  kandinsky_local backend deprecated, using kandinsky21_server instead")
            from ..backends.kandinsky.kandinsky21_server_backend import Kandinsky21ServerBackend
            return Kandinsky21ServerBackend(config=self.config, **kwargs)
        elif backend_type == "kandinsky21_server":
            from ..backends.kandinsky.kandinsky21_server_backend import Kandinsky21ServerBackend
            return Kandinsky21ServerBackend(config=self.config, **kwargs)
        elif backend_type == "remote":
            from ..backends.remote.api_backend import RemoteBackend
            return RemoteBackend(config=self.config, **kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def register(self, image: Image.Image, embedding: Any = None, latent: Any = None, prompt: str = None):
        """Register the latest generation result for history and interpolation.
        
        Args:
            image: Generated PIL Image
            embedding: Optional embedding tensor
            latent: Optional latent tensor
            prompt: Prompt used for generation
        """
        self._recent_images.append(image)
        self._recent_embeddings.append(embedding)
        self._recent_latents.append(latent)
        self._recent_prompts.append(prompt or self.prompt)
    
    def gen(self, prompt: Optional[str] = None, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """Generate a new image from text prompt.
        
        Args:
            prompt: Text description of image to generate
            **kwargs: Generation parameters to override defaults
            
        Returns:
            Generated PIL Image or list of images if num_images_per_prompt > 1
            
        Raises:
            ValueError: If no prompt is provided
        """
        use_prompt = prompt or self.prompt
        if not use_prompt:
            raise ValueError("No prompt provided")
        
        # Merge generation parameters
        params = {**self.generation_params, **kwargs}
        
        # Check if batch generation is requested
        num_images = params.get('num_images_per_prompt', 1)
        
        # Generate
        result = self.backend.generate(use_prompt, **params)
        
        # Handle single or multiple images
        if num_images > 1:
            # Multiple images - result should contain list
            images = result['image'] if isinstance(result['image'], list) else [result['image']]
            
            # Register the first image for continuity
            if images:
                self.register(
                    images[0], 
                    result.get('embeddings'), 
                    result.get('latents'),
                    use_prompt
                )
            
            return images
        else:
            # Single image
            image = result['image'] if not isinstance(result['image'], list) else result['image'][0]
            
            # Register result
            self.register(
                image, 
                result.get('embeddings'), 
                result.get('latents'),
                use_prompt
            )
            
            return image
    
    def set_interpolation_targets(self, prompt1: str, prompt2: str):
        """Set up interpolation between two prompts.
        
        Args:
            prompt1: First prompt (source)
            prompt2: Second prompt (target)
        """
        result1 = self.backend.generate(prompt1, **self.generation_params)
        result2 = self.backend.generate(prompt2, **self.generation_params)
        
        self._interpolation_source = {
            'prompt': prompt1,
            'embedding': result1.get('embeddings'),
            'image': result1['image']
        }
        self._interpolation_target = {
            'prompt': prompt2,
            'embedding': result2.get('embeddings'),
            'image': result2['image']
        }
    
    def gen_interpolated(self, alpha: float) -> Image.Image:
        """Generate an image interpolated between source and target.
        
        Args:
            alpha: Interpolation factor (0.0=source, 1.0=target)
            
        Returns:
            Interpolated PIL Image
            
        Raises:
            ValueError: If interpolation targets are not set
        """
        if not self._interpolation_source or not self._interpolation_target:
            raise ValueError("Interpolation targets not set. Call set_interpolation_targets() first.")
        
        if not (self._interpolation_source['embedding'] is not None and 
                self._interpolation_target['embedding'] is not None):
            # Fall back to img2img interpolation if no embeddings
            alpha = max(0.0, min(1.0, alpha))
            strength = 0.3 + alpha * 0.4  # Vary strength based on alpha
            return self.gen_img2img(strength=strength)
        
        # Interpolate embeddings
        interp_embedding = self.backend.interpolate_embeddings(
            self._interpolation_source['embedding'],
            self._interpolation_target['embedding'],
            alpha
        )
        
        # Generate with interpolated embedding
        # Note: This would need backend-specific implementation
        # For now, fall back to img2img
        return self.gen_img2img(strength=0.3 + alpha * 0.4)
    
    def get_recent_image(self, index: int = -1) -> Optional[Image.Image]:
        """Get a recent image by index.
        
        Args:
            index: Index of image to retrieve (-1 for most recent)
            
        Returns:
            PIL Image or None if index is out of range
        """
        try:
            return list(self._recent_images)[index]
        except IndexError:
            return None
    
    def update_params(self, **kwargs):
        """Update generation parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.generation_params.update(kwargs)
    
    def gen_interpolated_embeddings(self, prompt1: str, prompt2: str, batch_size: int, **kwargs) -> List[Image.Image]:
        """Generate a batch of images using interpolated embeddings between two prompts.
        
        Args:
            prompt1: The starting text prompt
            prompt2: The ending text prompt  
            batch_size: Number of interpolation steps (including start and end)
            **kwargs: Additional generation parameters
            
        Returns:
            List of PIL Images representing the interpolation sequence
        """
        # Merge generation parameters
        params = {**self.generation_params, **kwargs}
        
        # Call backend method
        result = self.backend.generate_interpolated_embeddings(
            prompt1=prompt1,
            prompt2=prompt2, 
            batch_size=batch_size,
            **params
        )
        
        # Extract images from result
        images = result['images'] if isinstance(result['images'], list) else [result['images']]
        
        # Register the first image for continuity
        if images:
            self.register(
                images[0],
                result.get('embeddings'),
                result.get('latents'), 
                prompt1
            )
        
        return images
    
    def clear_history(self):
        """Clear generation history."""
        self._recent_images.clear()
        self._recent_embeddings.clear()
        self._recent_latents.clear()
        self._recent_prompts.clear()
