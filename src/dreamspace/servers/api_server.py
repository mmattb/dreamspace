"""FastAPI server for hosting image generation models."""

import asyncio
import base64
import time
from io import BytesIO
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

try:
    # Try relative imports first (when used as a package)
    from ...core.image_gen import ImgGen
    from ...config.settings import Config
except ImportError:
    # Fall back to absolute imports (when used as a script)
    from dreamspace.core.image_gen import ImgGen
    from dreamspace.config.settings import Config


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(512, description="Image width")
    height: Optional[int] = Field(512, description="Image height")
    seed: Optional[int] = Field(None, description="Seed for reproducibility")
    batch_size: Optional[int] = Field(1, description="Number of images to generate (1-32)")


class GenerateBatchRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    batch_size: int = Field(..., description="Number of variations to generate (1-32)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(512, description="Image width")
    height: Optional[int] = Field(512, description="Image height")
    seed: Optional[int] = Field(None, description="Base seed for variations")


class BatchImageResponse(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded generated images")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class Img2ImgRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image transformation")
    image: str = Field(..., description="Base64 encoded source image")
    strength: float = Field(0.5, description="Transformation strength (0.0-1.0)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(512, description="Image width")
    height: Optional[int] = Field(512, description="Image height")
    seed: Optional[int] = Field(None, description="Seed for reproducibility")


class InterpolateRequest(BaseModel):
    embedding1: List[float] = Field(..., description="First embedding")
    embedding2: List[float] = Field(..., description="Second embedding")
    alpha: float = Field(..., description="Interpolation factor (0.0-1.0)")


class ImageResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded generated image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool


class ModelInfo(BaseModel):
    name: str
    backend_type: str
    device: str
    memory_usage: Optional[str] = None


# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = Config()
    backend_type = app_state.get("backend_type", "kandinsky_local")
    
    try:
        app_state["img_gen"] = ImgGen(backend=backend_type, config=config)
        app_state["config"] = config
        print(f"✅ Model loaded successfully: {backend_type}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        app_state["img_gen"] = None
    
    yield
    
    # Shutdown
    if "img_gen" in app_state and app_state["img_gen"]:
        if hasattr(app_state["img_gen"].backend, 'cleanup'):
            app_state["img_gen"].backend.cleanup()
        print("🧹 Cleaned up resources")


def create_app(backend_type: str = "kandinsky_local", 
               enable_auth: bool = False,
               api_key: Optional[str] = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        backend_type: Type of backend to use
        enable_auth: Whether to enable API key authentication
        api_key: API key for authentication (if enabled)
        
    Returns:
        Configured FastAPI application
    """
    app_state["backend_type"] = backend_type
    app_state["enable_auth"] = enable_auth
    app_state["api_key"] = api_key
    
    app = FastAPI(
        title="Dreamspace Co-Pilot API",
        description="Image generation API for the Dreamspace Co-Pilot project",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Authentication
    security = HTTPBearer() if enable_auth else None
    
    def verify_token_with_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify API token when auth is enabled."""
        if not credentials or credentials.credentials != api_key:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return True
    
    def verify_token_no_auth():
        """No-op verification when auth is disabled."""
        return True
    
    # Choose the right dependency based on auth setting
    auth_dependency = verify_token_with_auth if enable_auth else verify_token_no_auth
    
    def get_img_gen():
        """Get ImgGen instance."""
        img_gen = app_state.get("img_gen")
        if not img_gen:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return img_gen
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        import torch
        return HealthResponse(
            status="healthy",
            model_loaded=app_state.get("img_gen") is not None,
            gpu_available=torch.cuda.is_available()
        )
    
    @app.get("/models", response_model=List[ModelInfo])
    async def get_models(authenticated: bool = Depends(auth_dependency)):
        """Get available model information."""
        img_gen = get_img_gen()
        backend = img_gen.backend
        
        return [ModelInfo(
            name=app_state.get("backend_type", "unknown"),
            backend_type=type(backend).__name__,
            device=getattr(backend, 'device', 'unknown')
        )]
    
    @app.post("/generate", response_model=ImageResponse)
    async def generate_image(
        request: GenerateRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Generate an image from text prompt."""
        try:
            img_gen = get_img_gen()
            
            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'batch_size']
            }
            
            # Handle batch generation
            batch_size = request.batch_size or 1
            if batch_size > 1:
                # For batch generation, return first image but could extend this
                images = []
                base_seed = request.seed
                for i in range(batch_size):
                    if base_seed is not None:
                        gen_params['seed'] = base_seed + i
                    image = img_gen.gen(prompt=request.prompt, **gen_params)
                    images.append(image)
                
                # Return first image (could extend API to return all)
                image = images[0]
            else:
                # Single image generation
                image = img_gen.gen(prompt=request.prompt, **gen_params)
            
            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return ImageResponse(
                image=image_b64,
                metadata={
                    "prompt": request.prompt,
                    "parameters": gen_params,
                    "batch_size": batch_size
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate_batch", response_model=BatchImageResponse)
    async def generate_batch(
        request: GenerateBatchRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Generate a batch of image variations for animation."""
        try:
            img_gen = get_img_gen()
            
            # Limit batch size for server stability
            batch_size = min(request.batch_size, 32)
            
            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'batch_size']
            }
            
            print(f"🎬 Generating batch of {batch_size} subtle variations...")
            start_time = time.time()
            
            # Generate variations using img2img for consistency
            images = []
            
            # First, generate a base image
            base_seed = request.seed or 42
            base_params = {**gen_params, 'seed': base_seed}
            # Remove batch parameter for single generation
            base_params.pop('num_images_per_prompt', None)
            
            print(f"  � Generating base image with seed {base_seed}...")
            base_image = img_gen.gen(prompt=request.prompt, **base_params)
            images.append(base_image)
            
            # Generate variations using img2img with low strength
            print(f"  🔄 Generating {batch_size-1} variations using img2img...")
            for i in range(1, batch_size):
                # Use very low strength for subtle variations
                variation_seed = base_seed + i
                
                # Use backend img2img directly since we have a specific source image
                variation_params = {k: v for k, v in gen_params.items() if k not in ['seed', 'num_images_per_prompt']}
                if variation_seed:
                    variation_params['seed'] = variation_seed
                
                # Call backend img2img directly
                result = img_gen.backend.img2img(
                    image=base_image,
                    prompt=request.prompt,
                    strength=0.15,  # Very low strength for subtle changes
                    **variation_params
                )
                
                # Extract the image from the result
                variation = result['image'] if isinstance(result, dict) else result
                images.append(variation)
                print(f"    Generated variation {i+1}/{batch_size-1}")
            
            # Convert all images to base64
            image_b64_list = []
            for i, image in enumerate(images):
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_b64_list.append(image_b64)
                print(f"  Encoded image {i+1}/{len(images)}")
            
            elapsed = time.time() - start_time
            print(f"✅ Variation batch complete in {elapsed:.1f}s ({elapsed/batch_size:.2f}s per image)")
            
            return BatchImageResponse(
                images=image_b64_list,
                metadata={
                    "prompt": request.prompt,
                    "parameters": gen_params,
                    "batch_size": len(image_b64_list),
                    "animation_ready": True,
                    "generation_time": elapsed,
                    "variation_method": "img2img_low_strength",
                    "base_seed": base_seed
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/img2img", response_model=ImageResponse)
    async def image_to_image(
        request: Img2ImgRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Transform an image using text prompt."""
        try:
            img_gen = get_img_gen()
            
            # Decode source image
            image_data = base64.b64decode(request.image)
            source_image = Image.open(BytesIO(image_data))
            
            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'image', 'strength']
            }
            
            # Transform image
            result_image = img_gen.gen_img2img(
                strength=request.strength,
                prompt=request.prompt,
                **gen_params
            )
            
            # Convert to base64
            buffer = BytesIO()
            result_image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return ImageResponse(
                image=image_b64,
                metadata={
                    "prompt": request.prompt,
                    "strength": request.strength,
                    "parameters": gen_params
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/interpolate")
    async def interpolate_embeddings(
        request: InterpolateRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Interpolate between embeddings."""
        try:
            img_gen = get_img_gen()
            backend = img_gen.backend
            
            import torch
            embedding1 = torch.tensor(request.embedding1)
            embedding2 = torch.tensor(request.embedding2)
            
            result = backend.interpolate_embeddings(embedding1, embedding2, request.alpha)
            
            return {"result": result.tolist() if torch.is_tensor(result) else result}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def run_server(backend_type: str = "kandinsky_local",
               host: str = "localhost",
               port: int = 8000,
               workers: int = 1,
               enable_auth: bool = False,
               api_key: Optional[str] = None):
    """Run the image generation server.
    
    Args:
        backend_type: Type of backend to use
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        enable_auth: Whether to enable authentication
        api_key: API key for authentication
    """
    app = create_app(backend_type, enable_auth, api_key)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dreamspace Co-Pilot API Server")
    parser.add_argument("--backend", default="kandinsky21_server", 
                       choices=[
                           "kandinsky_local", "kandinsky21_server",
                           "sd_local", "sd15_server", "sd21_server",
                           "remote"
                       ],
                       help="Backend type to use")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--auth", action="store_true", help="Enable authentication")
    parser.add_argument("--api-key", help="API key for authentication")
    
    args = parser.parse_args()
    
    run_server(
        backend_type=args.backend,
        host=args.host,
        port=args.port,
        workers=args.workers,
        enable_auth=args.auth,
        api_key=args.api_key
    )
