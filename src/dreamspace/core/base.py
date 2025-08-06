"""Base abstract interface for image generation backends."""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any


class ImgGenBackend(ABC):
    """Abstract backend for different image generation models.
    
    This abstract base class defines the interface that all image generation backends
    must implement. It supports text-to-image generation with latent space wiggle
    variations and semantic embedding interpolation for smooth transitions between concepts.
    
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
                - 'image': PIL.Image.Image or PyTorch tensor - The generated image
                - 'latents': Optional tensor - Latent space representation
                - 'embeddings': Optional tensor - Text/image embeddings for interpolation
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

    def _slerp(self, v1, v2, alpha):
        """Spherical linear interpolation utility method.
        
        Args:
            v1: First vector/tensor
            v2: Second vector/tensor
            alpha: Interpolation factor (0.0 to 1.0)
        
        Returns:
            Interpolated vector/tensor
        """
        if torch.is_tensor(v1):
            dot = torch.sum(v1 * v2)
            omega = torch.acos(torch.clamp(dot, -1, 1))
            sin_omega = torch.sin(omega)
            if sin_omega.abs() < 1e-6:  # Vectors are nearly parallel
                return v1 * (1 - alpha) + v2 * alpha
            return (torch.sin((1 - alpha) * omega) * v1 + torch.sin(alpha * omega) * v2) / sin_omega
        return v1 * (1 - alpha) + v2 * alpha
