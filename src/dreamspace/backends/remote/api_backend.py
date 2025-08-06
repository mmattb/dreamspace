"""Remote API backend implementation."""

import requests
import base64
import json
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config


class RemoteBackend(ImgGenBackend):
    """Remote API backend for accessing hosted image generation services.
    
    This backend communicates with remote servers that expose image generation
    APIs. It handles authentication, request/response serialization, and
    error handling for network operations.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: Optional[int] = None,
                 max_retries: Optional[int] = None):
        """Initialize the remote backend.
        
        Args:
            config: Configuration instance
            api_url: Base URL for the API (overrides config)
            api_key: API authentication key (overrides config)
            timeout: Request timeout in seconds (overrides config)
            max_retries: Maximum number of retry attempts (overrides config)
        """
        self.config = config or Config()
        
        # Use provided values or fall back to config
        if self.config.remote:
            self.api_url = (api_url or self.config.remote.api_url).rstrip('/')
            self.api_key = api_key or self.config.remote.api_key
            self.timeout = timeout or self.config.remote.timeout
            self.max_retries = max_retries or self.config.remote.max_retries
        else:
            if not api_url:
                raise ValueError("No API URL provided and no remote config found")
            self.api_url = api_url.rstrip('/')
            self.api_key = api_key
            self.timeout = timeout or 60
            self.max_retries = max_retries or 3
        
        # Set up HTTP session
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "dreamspace-copilot/0.1.0"
        })
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt via API.
        
        Args:
            prompt: Text description of the image
            **kwargs: Generation parameters
            
        Returns:
            Dict with 'image', 'latents', and 'embeddings'
        """
        payload = {"prompt": prompt, **kwargs}
        
        response = self._make_request("POST", "/generate", payload)
        return self._process_image_response(response)
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Transform an existing image using a text prompt via API.
        
        Args:
            image: Source image to transform
            prompt: Text description guiding the transformation
            strength: How much to change the image (0.0-1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with 'image', 'latents', and 'embeddings'
        """
        # Convert PIL Image to base64 for API
        image_b64 = self._image_to_base64(image)
        
        payload = {
            "prompt": prompt,
            "image": image_b64,
            "strength": strength,
            **kwargs
        }
        
        response = self._make_request("POST", "/img2img", payload)
        return self._process_image_response(response)
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        """Interpolate between two embeddings via API.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            alpha: Interpolation factor (0.0-1.0)
            
        Returns:
            Interpolated embedding
        """
        payload = {
            "embedding1": embedding1,
            "embedding2": embedding2,
            "alpha": alpha
        }
        
        response = self._make_request("POST", "/interpolate", payload)
        return response.json().get('result')
    
    def generate_interpolated_embeddings(self, prompt1: str, prompt2: str, batch_size: int, **kwargs) -> Dict[str, Any]:
        """Generate a batch of images using interpolated embeddings via API.
        
        Args:
            prompt1: The starting text prompt
            prompt2: The ending text prompt
            batch_size: Number of interpolation steps (including start and end)
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with 'images', 'latents', and 'embeddings'
        """
        payload = {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "batch_size": batch_size,
            **kwargs
        }
        
        response = self._make_request("POST", "/generate_interpolated_embeddings", payload)
        return self._process_batch_image_response(response)
    
    def _process_batch_image_response(self, response: requests.Response) -> Dict[str, Any]:
        """Process API response containing multiple images.
        
        Args:
            response: HTTP response object
            
        Returns:
            Dict with processed images and metadata
        """
        data = response.json()
        
        # Convert base64 images back to PIL Images
        images = []
        for image_data in data['images']:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
        
        return {
            'images': images,
            'latents': data.get('metadata', {}).get('latents'),
            'embeddings': data.get('metadata', {}).get('embeddings')
        }

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> requests.Response:
        """Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request payload data
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails after all retries
        """
        url = f"{self.api_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=self.timeout, params=data)
                else:
                    response = self.session.request(
                        method, url, json=data, timeout=self.timeout
                    )
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)
        
        # This shouldn't be reached, but just in case
        raise requests.RequestException("Max retries exceeded")
    
    def _process_image_response(self, response: requests.Response) -> Dict[str, Any]:
        """Process API response containing image data.
        
        Args:
            response: HTTP response object
            
        Returns:
            Dict with processed image and metadata
        """
        data = response.json()
        
        # Convert base64 image back to PIL Image
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        
        return {
            'image': image,
            'latents': data.get('latents'),
            'embeddings': data.get('embeddings')
        }
    
    def _image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image to convert
            format: Image format for encoding
            
        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_b64
    
    def health_check(self) -> bool:
        """Check if the remote API is available.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = self._make_request("GET", "/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models from the API.
        
        Returns:
            Dict containing model information
        """
        response = self._make_request("GET", "/models")
        return response.json()
