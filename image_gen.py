
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
import requests
import json
from collections import deque


class ImgGenBackend(ABC):
    """Abstract backend for different image generation models.
    
    This abstract base class defines the interface that all image generation backends
    must implement. It supports text-to-image generation, image-to-image transformation,
    and semantic embedding interpolation for smooth transitions between concepts.
    
    Implementations should handle model-specific details like loading pipelines,
    managing embeddings, and optimizing performance for their respective models
    (e.g., Stable Diffusion, Kandinsky, or remote API services).
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt.
        
        Args:
            prompt: The text description of the image to generate
            **kwargs: Model-specific generation parameters (guidance_scale, 
                     num_inference_steps, width, height, etc.)
        
        Returns:
            Dict containing:
                - 'image': PIL.Image.Image - The generated image
                - 'latents': Optional tensor - Latent space representation
                - 'embeddings': Optional tensor - Text/image embeddings for interpolation
        """
        pass
    
    @abstractmethod
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Transform an existing image using a text prompt.
        
        This method enables visual continuity by using an existing image as a starting
        point and applying transformations based on the prompt. Lower strength values
        preserve more of the original image structure.
        
        Args:
            image: Source PIL Image to transform
            prompt: Text description guiding the transformation
            strength: Float 0.0-1.0, how much to change the image (0.0=no change, 1.0=complete transformation)
            **kwargs: Model-specific parameters
        
        Returns:
            Dict containing:
                - 'image': PIL.Image.Image - The transformed image
                - 'latents': Optional tensor - Latent space representation  
                - 'embeddings': Optional tensor - Text/image embeddings
        """
        pass
    
    @abstractmethod
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        """Interpolate between two semantic embeddings.
        
        Creates smooth transitions between different concepts by blending their
        semantic representations in embedding space. Uses spherical linear 
        interpolation (slerp) for better results than linear interpolation.
        
        Args:
            embedding1: First embedding (source concept)
            embedding2: Second embedding (target concept)  
            alpha: Float 0.0-1.0, interpolation factor (0.0=embedding1, 1.0=embedding2)
        
        Returns:
            Interpolated embedding that can be used for generation
        """
        pass


class LocalStableDiffusionBackend(ImgGenBackend):
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        self.device = device
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        result = self.pipe(prompt, **kwargs)
        return {
            'image': result.images[0],
            'latents': getattr(result, 'latents', None),
            'embeddings': None  # Would need to extract from pipe internals
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        result = self.img2img_pipe(prompt=prompt, image=image, strength=strength, **kwargs)
        return {
            'image': result.images[0],
            'latents': getattr(result, 'latents', None),
            'embeddings': None
        }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        # Spherical linear interpolation for better results
        return self._slerp(embedding1, embedding2, alpha)
    
    def _slerp(self, v1, v2, alpha):
        """Spherical linear interpolation"""
        if torch.is_tensor(v1):
            dot = torch.sum(v1 * v2)
            omega = torch.acos(torch.clamp(dot, -1, 1))
            sin_omega = torch.sin(omega)
            return (torch.sin((1 - alpha) * omega) * v1 + torch.sin(alpha * omega) * v2) / sin_omega
        return v1 * (1 - alpha) + v2 * alpha


class LocalKandinskyBackend(ImgGenBackend):
    def __init__(self, device: str = "cuda"):
        from diffusers import KandinskyV22Pipeline, KandinskyV22Img2ImgPipeline, KandinskyV22PriorPipeline
        self.prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ).to(device)
        self.pipe = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to(device)
        self.img2img_pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to(device)
        self.device = device
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        image_embeds, negative_embeds = self.prior_pipe(prompt).to_tuple()
        result = self.pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_embeds,
            **kwargs
        )
        return {
            'image': result.images[0],
            'latents': None,
            'embeddings': image_embeds
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        image_embeds, negative_embeds = self.prior_pipe(prompt).to_tuple()
        result = self.img2img_pipe(
            image=image,
            image_embeds=image_embeds,
            negative_image_embeds=negative_embeds,
            strength=strength,
            **kwargs
        )
        return {
            'image': result.images[0],
            'latents': None,
            'embeddings': image_embeds
        }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        return self._slerp(embedding1, embedding2, alpha)
    
    def _slerp(self, v1, v2, alpha):
        if torch.is_tensor(v1):
            dot = torch.sum(v1 * v2)
            omega = torch.acos(torch.clamp(dot, -1, 1))
            sin_omega = torch.sin(omega)
            return (torch.sin((1 - alpha) * omega) * v1 + torch.sin(alpha * omega) * v2) / sin_omega
        return v1 * (1 - alpha) + v2 * alpha


class RemoteBackend(ImgGenBackend):
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        payload = {"prompt": prompt, **kwargs}
        response = self.session.post(f"{self.api_url}/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Convert base64 image back to PIL Image
        import base64
        from io import BytesIO
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        
        return {
            'image': image,
            'latents': data.get('latents'),
            'embeddings': data.get('embeddings')
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        # Convert PIL Image to base64 for API
        import base64
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        payload = {
            "prompt": prompt,
            "image": image_b64,
            "strength": strength,
            **kwargs
        }
        response = self.session.post(f"{self.api_url}/img2img", json=payload)
        response.raise_for_status()
        data = response.json()
        
        image_data = base64.b64decode(data['image'])
        result_image = Image.open(BytesIO(image_data))
        
        return {
            'image': result_image,
            'latents': data.get('latents'),
            'embeddings': data.get('embeddings')
        }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        payload = {
            "embedding1": embedding1,
            "embedding2": embedding2,
            "alpha": alpha
        }
        response = self.session.post(f"{self.api_url}/interpolate", json=payload)
        response.raise_for_status()
        return response.json()['result']


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
                 **backend_kwargs):
        """Initialize the ImgGen instance.
        
        Args:
            backend: Either a backend type string ("sd_local", "kandinsky_local", "remote")
                    or a pre-configured ImgGenBackend instance
            prompt: Default prompt to use for generations when none is specified
            history_size: Maximum number of recent generations to keep in memory
            **backend_kwargs: Additional arguments passed to backend constructor
                            (e.g., model_id, device, api_url, api_key)
        
        Raises:
            ValueError: If backend type string is not recognized
        """
        
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
        
        # Generation parameters
        self.generation_params = {
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'width': 512,
            'height': 512
        }
        
        # Interpolation state
        self._interpolation_source = None
        self._interpolation_target = None
    
    def _create_backend(self, backend_type: str, **kwargs) -> ImgGenBackend:
        if backend_type == "sd_local":
            return LocalStableDiffusionBackend(**kwargs)
        elif backend_type == "kandinsky_local":
            return LocalKandinskyBackend(**kwargs)
        elif backend_type == "remote":
            return RemoteBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def register(self, image: Image.Image, embedding: Any = None, latent: Any = None, prompt: str = None):
        """Register the latest generation result for history and interpolation"""
        self._recent_images.append(image)
        self._recent_embeddings.append(embedding)
        self._recent_latents.append(latent)
        self._recent_prompts.append(prompt or self.prompt)
    
    def gen(self, prompt: Optional[str] = None, **kwargs) -> Image.Image:
        """Generate a new image"""
        use_prompt = prompt or self.prompt
        if not use_prompt:
            raise ValueError("No prompt provided")
        
        # Merge generation parameters
        params = {**self.generation_params, **kwargs}
        
        # Generate
        result = self.backend.generate(use_prompt, **params)
        
        # Register result
        self.register(
            result['image'], 
            result.get('embeddings'), 
            result.get('latents'),
            use_prompt
        )
        
        return result['image']
    
    def gen_img2img(self, strength: float = 0.5, prompt: Optional[str] = None, **kwargs) -> Image.Image:
        """Generate using img2img from the most recent image"""
        if not self._recent_images:
            return self.gen(prompt, **kwargs)  # Fall back to text2img
        
        use_prompt = prompt or self.prompt
        if not use_prompt:
            raise ValueError("No prompt provided")
        
        # Use most recent image as source
        source_image = self._recent_images[-1]
        
        params = {**self.generation_params, **kwargs}
        result = self.backend.img2img(source_image, use_prompt, strength, **params)
        
        # Register result
        self.register(
            result['image'], 
            result.get('embeddings'), 
            result.get('latents'),
            use_prompt
        )
        
        return result['image']
    
    def set_interpolation_targets(self, prompt1: str, prompt2: str):
        """Set up interpolation between two prompts"""
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
        """Generate an image interpolated between source and target (alpha: 0.0 to 1.0)"""
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
        """Get a recent image by index (-1 for most recent)"""
        try:
            return list(self._recent_images)[index]
        except IndexError:
            return None
    
    def update_params(self, **kwargs):
        """Update generation parameters"""
        self.generation_params.update(kwargs)
    
    def clear_history(self):
        """Clear generation history"""
        self._recent_images.clear()
        self._recent_embeddings.clear()
        self._recent_latents.clear()
        self._recent_prompts.clear()
        
