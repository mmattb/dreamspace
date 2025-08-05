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
import torch

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
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
    seed: Optional[int] = Field(None, description="Seed for reproducibility")
    batch_size: Optional[int] = Field(1, description="Number of images to generate (1-32)")


class GenerateBatchRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    batch_size: int = Field(..., description="Number of variations to generate (1-32)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
    seed: Optional[int] = Field(None, description="Base seed for variations")
    noise_magnitude: Optional[float] = Field(0.3, description="Magnitude of noise for latent variations")
    bifurcation_step: Optional[int] = Field(3, description="Number of steps from end to bifurcate in bifurcated wiggle")
    output_format: Optional[str] = Field("jpeg", description="Output format: 'jpeg' (base64), 'tensor' (numpy), or 'png' (base64)")


class BatchImageResponse(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images or serialized tensors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class Img2ImgRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image transformation")
    image: str = Field(..., description="Base64 encoded source image")
    strength: float = Field(0.5, description="Transformation strength (0.0-1.0)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
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
    gpu_count: Optional[int] = None
    current_device: Optional[int] = None
    selected_gpus: Optional[List[int]] = None
    selection: Optional[str] = None


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
    gpu_selection = app_state.get("gpu_selection", "auto")
    disable_safety_checker = app_state.get("disable_safety_checker", False)
    
    try:
        # Prepare backend kwargs with GPU configuration
        backend_kwargs = {}
        
        # Add safety checker configuration for SD 1.5
        if backend_type == "sd15_server" and disable_safety_checker:
            backend_kwargs["disable_safety_checker"] = True
        
        # Add GPU configuration if specified
        if gpu_selection != "auto":
            # For single GPU, set device parameter
            first_gpu = gpu_selection.split(',')[0]
            backend_kwargs["device"] = f"cuda:{first_gpu}"
            
            if "," in gpu_selection:
                # Multi-GPU configuration - create multiple backend instances
                gpu_list = [gpu.strip() for gpu in gpu_selection.split(",")]
                print(f"🎮 Multi-GPU setup: Creating backends on GPUs {gpu_list}")
                
                # Create primary backend on first GPU
                app_state["img_gen"] = ImgGen(backend=backend_type, config=config, **backend_kwargs)
                
                # Create additional backends for other GPUs
                multi_backends = {"0": app_state["img_gen"]}  # GPU 0 (first GPU)
                
                for i, gpu_id in enumerate(gpu_list[1:], 1):  # Start from second GPU
                    try:
                        gpu_backend_kwargs = {**backend_kwargs}
                        gpu_backend_kwargs["device"] = f"cuda:{gpu_id}"
                        gpu_backend = ImgGen(backend=backend_type, config=config, **gpu_backend_kwargs)
                        multi_backends[gpu_id] = gpu_backend
                        print(f"  ✅ Backend {i+1} loaded on GPU {gpu_id}")
                    except Exception as e:
                        print(f"  ❌ Failed to load backend on GPU {gpu_id}: {e}")
                
                app_state["multi_backends"] = multi_backends
                app_state["gpu_list"] = gpu_list
                print(f"🎯 Multi-GPU setup complete: {len(multi_backends)} backends ready")
            else:
                print(f"🎮 Single GPU setup: GPU {gpu_selection}")
                app_state["img_gen"] = ImgGen(backend=backend_type, config=config, **backend_kwargs)
        else:
            app_state["img_gen"] = ImgGen(backend=backend_type, config=config, **backend_kwargs)
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
        print("🧹 Cleaned up primary backend")
    
    # Clean up additional GPU backends
    if "multi_backends" in app_state:
        multi_backends = app_state["multi_backends"]
        for gpu_id, backend in multi_backends.items():
            if backend and backend != app_state.get("img_gen"):  # Don't double-cleanup primary
                if hasattr(backend.backend, 'cleanup'):
                    backend.backend.cleanup()
        print(f"🧹 Cleaned up {len(multi_backends)} GPU backends")


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
    async def health():
        """Health check endpoint."""
        try:
            img_gen = get_img_gen()
            model_loaded = img_gen is not None
            
            # Check GPU availability
            gpu_available = False
            gpu_info = {}
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    gpu_selection = app_state.get("gpu_selection", "auto")
                    if gpu_selection == "auto":
                        gpu_count = torch.cuda.device_count()
                        gpu_info = {
                            "gpu_count": gpu_count,
                            "current_device": torch.cuda.current_device(),
                            "selection": "auto (all GPUs)"
                        }
                    else:
                        gpu_ids = [int(x.strip()) for x in gpu_selection.split(",")]
                        gpu_info = {
                            "gpu_count": len(gpu_ids),
                            "selected_gpus": gpu_ids,
                            "selection": gpu_selection
                        }
            except ImportError:
                pass
            
            response_data = {
                "status": "healthy",
                "model_loaded": model_loaded,
                "gpu_available": gpu_available
            }
            response_data.update(gpu_info)
            
            return HealthResponse(**response_data)
            
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                gpu_available=False
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
                base_seed = request.seed or 42  # Default seed if not provided
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
            
            # Convert to base64 using JPEG for smaller file size
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=90, optimize=True)
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
        """Generate a batch of image variations for animation with smart chunking for multi-GPU."""
        import traceback
        import time

        try:
            img_gen = get_img_gen()

            # Limit batch size for server stability
            batch_size = min(request.batch_size, 32)

            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'batch_size', 'noise_magnitude', 'bifurcation_step']
            }

            # Delegate batch generation with latent wiggle to the backend
            print(f"🎬 Generating batch of {batch_size} images with latent wiggle...")
            start_time = time.time()
            base_seed = request.seed or 42  # Default seed if not provided

            # Choose between original and bifurcated wiggle methods
            # Bifurcated wiggle is now the default method
            # Set bifurcation_step to 0 to use original method (for backwards compatibility)
            use_original = request.bifurcation_step == 0
            
            if use_original:
                print("🎯 Using original latent wiggle method (bifurcation_step=0)")
                # Call the original wiggle method
                result = img_gen.backend.generate_batch_with_latent_wiggle(
                    prompt=request.prompt,
                    batch_size=batch_size,
                    noise_magnitude=request.noise_magnitude,
                    **gen_params
                )
            else:
                print(f"🔀 Using bifurcated wiggle with {request.bifurcation_step} refinement steps (default method)")
                # Call the bifurcated wiggle method with output format
                result = img_gen.backend.generate_batch_with_bifurcated_wiggle(
                    prompt=request.prompt,
                    batch_size=batch_size,
                    noise_magnitude=request.noise_magnitude,
                    bifurcation_step=request.bifurcation_step,
                    output_format="tensor" if request.output_format == "tensor" else "pil",
                    **gen_params
                )

            all_images = result['images']
            result_format = result.get('format', 'pil')

            print(f"✅ Batch generation complete: {len(all_images)} images in {result_format} format")

            # Handle different output formats
            if request.output_format == "tensor" and result_format == "numpy_float32":
                # Ultra-fast tensor serialization for local clients
                print("🚀 Serializing tensors for high-speed local transfer...")
                encoding_start = time.time()
                
                import numpy as np
                import pickle
                import gzip
                
                # Compress numpy array for network transfer
                serialized_data = pickle.dumps({
                    'images': all_images,  # numpy array shape (batch, h, w, 3) 
                    'shape': all_images.shape,
                    'dtype': str(all_images.dtype),
                    'format': 'numpy_compressed'
                })
                
                # Optional compression for network transfer
                compressed_data = gzip.compress(serialized_data)
                tensor_b64 = base64.b64encode(compressed_data).decode('utf-8')
                
                encoding_time = time.time() - encoding_start
                print(f"🎯 Tensor serialization complete in {encoding_time:.3f}s")
                print(f"   Original size: {all_images.nbytes/1024/1024:.1f}MB")
                print(f"   Compressed size: {len(compressed_data)/1024/1024:.1f}MB")
                print(f"   Compression ratio: {all_images.nbytes/len(compressed_data):.1f}x")
                
                return BatchImageResponse(
                    images=[tensor_b64],  # Single serialized tensor containing all images
                    metadata={
                        "prompt": request.prompt,
                        "parameters": gen_params,
                        "batch_size": len(all_images),
                        "animation_ready": True,
                        "generation_time": time.time() - start_time,
                        "generation_method": "latent_wiggle" if use_original else "bifurcated_wiggle",
                        "output_format": "tensor",
                        "tensor_shape": list(all_images.shape),
                        "tensor_dtype": str(all_images.dtype),
                        "compression": "gzip",
                        "noise_magnitude": request.noise_magnitude,
                        "bifurcation_step": None if use_original else request.bifurcation_step
                    }
                )
            
            else:
                # Traditional PIL → JPEG → Base64 pipeline
                print("📦 Encoding images to JPEG...")
                encoding_start = time.time()

            def encode_single_image(args):
                from io import BytesIO  # Import inside function to avoid scope issues
                import base64
                import time
                i, image = args
                
                # Time the JPEG encoding
                jpeg_start = time.time()
                
                if request.output_format == "jpeg_optimized" and hasattr(image, 'dtype'):
                    # Direct numpy → JPEG encoding (skip PIL)
                    import cv2
                    # Convert from float [0,1] to uint8 [0,255] if needed
                    if image.dtype == 'float32' or image.dtype == 'float64':
                        image_array = (image * 255).astype('uint8')
                    else:
                        image_array = image
                    
                    # Direct JPEG encoding
                    _, jpeg_bytes = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    buffer_size = len(jpeg_bytes)
                    jpeg_time = time.time() - jpeg_start
                    
                    # Time the base64 encoding
                    b64_start = time.time()
                    image_b64 = base64.b64encode(jpeg_bytes.tobytes()).decode('utf-8')
                    b64_time = time.time() - b64_start
                else:
                    # Traditional PIL → JPEG encoding
                    buffer = BytesIO()
                    # Use JPEG with high quality for much smaller file sizes
                    image.save(buffer, format='JPEG', quality=90, optimize=True)
                    jpeg_time = time.time() - jpeg_start
                    
                    # Time the base64 encoding
                    b64_start = time.time()
                    buffer_size = len(buffer.getvalue())
                    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    b64_time = time.time() - b64_start
                
                return i, image_b64, buffer_size, jpeg_time, b64_time

            # Use ThreadPoolExecutor for parallel encoding if we have many images
            if len(all_images) > 8:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(encode_single_image, enumerate(all_images)))

                # Sort results by original index and extract data
                results.sort(key=lambda x: x[0])
                image_b64_list = [r[1] for r in results]
                total_size = sum(r[2] for r in results)
                total_jpeg_time = sum(r[3] for r in results)
                total_b64_time = sum(r[4] for r in results)

                print(f"  Parallel encoded {len(all_images)} images")
                print(f"  📸 JPEG encoding time: {total_jpeg_time:.3f}s total, {total_jpeg_time/len(all_images):.3f}s avg")
                print(f"  🔤 Base64 encoding time: {total_b64_time:.3f}s total, {total_b64_time/len(all_images):.3f}s avg")
            else:
                # Sequential encoding for smaller batches
                image_b64_list = []
                total_size = 0
                total_jpeg_time = 0
                total_b64_time = 0

                for i, image in enumerate(all_images):
                    _, image_b64, buffer_size, jpeg_time, b64_time = encode_single_image((i, image))
                    image_b64_list.append(image_b64)
                    total_size += buffer_size
                    total_jpeg_time += jpeg_time
                    total_b64_time += b64_time

                print(f"  📸 JPEG encoding time: {total_jpeg_time:.3f}s total, {total_jpeg_time/len(all_images):.3f}s avg")
                print(f"  🔤 Base64 encoding time: {total_b64_time:.3f}s total, {total_b64_time/len(all_images):.3f}s avg")

            encoding_time = time.time() - encoding_start
            avg_size = total_size / len(all_images) if all_images else 0
            print(f"📦 Encoding complete in {encoding_time:.1f}s - Total: {total_size/1024/1024:.1f}MB, Avg: {avg_size/1024:.1f}KB per image")

            elapsed = time.time() - start_time
            avg_time = elapsed / len(all_images) if all_images else 0

            return BatchImageResponse(
                images=image_b64_list,
                metadata={
                    "prompt": request.prompt,
                    "parameters": gen_params,
                    "batch_size": len(image_b64_list),
                    "animation_ready": True,
                    "generation_time": time.time() - start_time,
                    "generation_method": "latent_wiggle" if use_original else "bifurcated_wiggle",
                    "multi_gpu": False,  # Using single GPU with batch generation
                    "gpu_count": 1,
                    "seed": request.seed,
                    "base_seed": base_seed,
                    "variation_method": "latent_noise",
                    "output_format": request.output_format or "jpeg",
                    "encoding_time": encoding_time,
                    "noise_magnitude": request.noise_magnitude,
                    "bifurcation_step": None if use_original else request.bifurcation_step
                }
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ CRITICAL ERROR in generate_batch main logic: {e}")
            print(f"📋 Full traceback: {error_details}")
            raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")
    
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
            
            # Convert to base64 using JPEG for smaller file size
            buffer = BytesIO()
            result_image.save(buffer, format='JPEG', quality=90, optimize=True)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Log encoding info
            buffer_size = len(buffer.getvalue())
            print(f"📦 Encoded img2img result: {buffer_size/1024:.1f}KB")
            
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
               api_key: Optional[str] = None,
               gpus: Optional[str] = None,
               disable_safety_checker: bool = False):
    """Run the image generation server.
    
    Args:
        backend_type: Type of backend to use
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        enable_auth: Whether to enable authentication
        api_key: API key for authentication
        gpus: GPU selection - 'auto', '0', '1', '0,1', or specific GPU IDs
        disable_safety_checker: Disable NSFW safety checker for SD 1.5
    """
    # Store configuration for backend creation
    app_state["gpu_selection"] = gpus
    app_state["disable_safety_checker"] = disable_safety_checker
    
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
    import os
    
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
    
    # GPU selection arguments
    parser.add_argument("--gpus", type=str, default="auto",
                       help="GPU selection: 'auto' (use all), '0' (first GPU), '1' (second GPU), '0,1' (both GPUs), or specific GPU IDs")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9,
                       help="Fraction of GPU memory to use (0.1-1.0)")
    
    # Safety checker arguments
    parser.add_argument("--disable-safety-checker", action="store_true",
                       help="Disable NSFW safety checker for SD 1.5 (fixes false positives)")
    
    args = parser.parse_args()
    
    # Set GPU environment based on selection
    if args.gpus != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"🎯 GPU Selection: Using GPU(s) {args.gpus}")
    else:
        print("🎯 GPU Selection: Auto (using all available GPUs)")
    
    print(f"🚀 Starting server with backend: {args.backend}")
    print(f"🌐 Binding to: {args.host}:{args.port}")
    if args.gpus != "auto":
        print(f"🎮 GPU Configuration: {args.gpus} (memory fraction: {args.gpu_memory_fraction})")
    if args.disable_safety_checker:
        print("🚫 NSFW safety checker disabled (fixes false positives)")
    
    run_server(
        backend_type=args.backend,
        host=args.host,
        port=args.port,
        workers=args.workers,
        enable_auth=args.auth,
        api_key=args.api_key,
        gpus=args.gpus,
        disable_safety_checker=args.disable_safety_checker
    )
