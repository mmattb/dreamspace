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


class BatchImageResponse(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded generated images")
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
    
    try:
        # Prepare backend kwargs with GPU configuration
        backend_kwargs = {}
        
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
            
            # Convert to base64 using JPEG for smaller file size
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=90, optimize=True)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Log encoding info
            buffer_size = len(buffer.getvalue())
            print(f"📦 Encoded image: {buffer_size/1024:.1f}KB")
            
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
            print(f"🔍 DEBUG: Starting generate_batch endpoint")
            print(f"🔍 DEBUG: Request data: {request.dict()}")
            
            img_gen = get_img_gen()
            print(f"🔍 DEBUG: Got img_gen: {type(img_gen)}")
            
            # Limit batch size for server stability
            batch_size = min(request.batch_size, 32)
            print(f"🔍 DEBUG: Batch size: {batch_size}")
            
            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'batch_size']
            }
            print(f"🔍 DEBUG: Generation parameters: {gen_params}")
            
            print(f"🎬 Generating batch of {batch_size} subtle variations...")
            start_time = time.time()
            
            # Check for multi-GPU setup
            gpu_selection = app_state.get("gpu_selection", "auto")
            multi_gpu = gpu_selection != "auto" and "," in gpu_selection
            has_multi_backends = "multi_backends" in app_state
            
            # Generate variations using img2img with low strength for consistency
            all_images = []
            base_seed = request.seed or 42
            
            print(f"  🎲 Using base seed: {base_seed}")
            
            # First, generate a base image (always on primary GPU)
            base_params = {**gen_params, 'seed': base_seed}
            # Remove batch parameter for single generation
            base_params.pop('num_images_per_prompt', None)
            
            print(f"  🖼️ Generating base image with seed {base_seed}...")
            base_image = img_gen.gen(prompt=request.prompt, **base_params)
            all_images.append(base_image)
            
            if batch_size == 1:
                print(f"✅ Single image generated")
            elif multi_gpu and has_multi_backends and batch_size > 4:
                # Multi-GPU parallel img2img variations for larger batches
                import threading
                
                multi_backends = app_state["multi_backends"]
                gpu_list = app_state["gpu_list"]
                variations_needed = batch_size - 1  # Exclude base image
                
                print(f"  🎮 Distributing {variations_needed} img2img variations across {len(gpu_list)} GPUs: {gpu_list}")
                
                # Thread-safe results collection
                results = {}
                results_lock = threading.Lock()
                
                # Convert base image to bytes for thread-safe sharing
                from io import BytesIO
                base_image_buffer = BytesIO()
                base_image.save(base_image_buffer, format='PNG')
                base_image_bytes = base_image_buffer.getvalue()
                
                def generate_variation_parallel(var_idx, gpu_id):
                    """Generate a single img2img variation on a specific GPU backend."""
                    try:
                        # Set CUDA device context at the start of each thread
                        import torch
                        torch.cuda.set_device(gpu_id)
                        
                        img_gen_instance = multi_backends.get(gpu_id)
                        if img_gen_instance is None:
                            print(f"  ⚠️ No img_gen instance available for GPU {gpu_id}, skipping variation {var_idx+1}")
                            with results_lock:
                                results[var_idx] = None
                            return
                        
                        # Access the actual backend through the ImgGen wrapper
                        backend = img_gen_instance.backend
                        
                        # Recreate base image from bytes in this thread's context
                        base_image_local = Image.open(BytesIO(base_image_bytes))
                        
                        variation_seed = base_seed + var_idx + 1
                        
                        # Use backend img2img directly with device-specific generator
                        variation_params = {k: v for k, v in gen_params.items() if k not in ['seed', 'num_images_per_prompt']}
                        if variation_seed:
                            variation_params['seed'] = variation_seed
                        
                        print(f"  🔧 GPU {gpu_id}: Generating img2img variation {var_idx+1} with seed {variation_seed}")
                        
                        # Clear CUDA cache before processing
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Call backend img2img directly with local image copy
                        result = backend.img2img(
                            image=base_image_local,
                            prompt=request.prompt,
                            strength=0.15,  # Very low strength for subtle changes
                            **variation_params
                        )
                        
                        # Extract the image from the result
                        variation = result['image'] if isinstance(result, dict) else result
                        
                        # Store results thread-safely
                        with results_lock:
                            results[var_idx] = variation
                        
                        print(f"  ✅ GPU {gpu_id}: Variation {var_idx+1} complete")
                        
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        print(f"  ❌ GPU {gpu_id}: Variation {var_idx+1} failed with exception: {e}")
                        print(f"  📋 GPU {gpu_id}: Full traceback: {error_details}")
                        with results_lock:
                            results[var_idx] = None
                        
                        # Clear CUDA cache on error
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                
                # Create and start threads for parallel img2img variations
                threads = []
                parallel_start_time = time.time()
                
                for i in range(variations_needed):
                    gpu_id = gpu_list[i % len(gpu_list)]  # Round-robin GPU assignment
                    thread = threading.Thread(
                        target=generate_variation_parallel,
                        args=(i, gpu_id),
                        name=f"GPU-{gpu_id}-Img2Img-{i+1}"
                    )
                    threads.append(thread)
                    thread.start()
                    print(f"  � Started parallel img2img thread for GPU {gpu_id}, variation {i+1}")
                
                # Wait for all parallel threads to complete
                print(f"  ⏳ Waiting for {len(threads)} parallel img2img threads to complete...")
                for i, thread in enumerate(threads):
                    thread.join()
                    print(f"  📋 Img2Img thread {i+1}/{len(threads)} completed ({thread.name})")
                
                parallel_elapsed = time.time() - parallel_start_time
                
                # Collect results in order
                for i in range(variations_needed):
                    if i in results and results[i] is not None:
                        all_images.append(results[i])
                    else:
                        print(f"    ⚠️ Missing or failed variation {i+1}, using base image as fallback")
                        all_images.append(base_image)  # Fallback to base image
                
                print(f"  🎯 Parallel multi-GPU img2img processing complete in {parallel_elapsed:.1f}s: {len(all_images)} total images")
                print(f"  ⚡ Parallel speedup: {variations_needed} variations processed simultaneously")
                
            else:
                # Sequential img2img variations (single GPU or small batches)
                print(f"  �🔄 Generating {batch_size-1} variations using sequential img2img...")
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
                    all_images.append(variation)
                    print(f"    Generated variation {i}/{batch_size-1}")
            
            print(f"✅ Generated {len(all_images)} images using img2img variations")
            
            # Convert all images to base64 using JPEG for much smaller file sizes
            print("📦 Encoding images to JPEG...")
            encoding_start = time.time()
            
            def encode_single_image(args):
                i, image = args
                buffer = BytesIO()
                # Use JPEG with high quality for much smaller file sizes
                image.save(buffer, format='JPEG', quality=90, optimize=True)
                buffer_size = len(buffer.getvalue())
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return i, image_b64, buffer_size
            
            # Use ThreadPoolExecutor for parallel encoding if we have many images
            if len(all_images) > 8:
                from concurrent.futures import ThreadPoolExecutor
                import threading
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(encode_single_image, enumerate(all_images)))
                
                # Sort results by original index and extract data
                results.sort(key=lambda x: x[0])
                image_b64_list = [r[1] for r in results]
                total_size = sum(r[2] for r in results)
                
                print(f"  Parallel encoded {len(all_images)} images")
            else:
                # Sequential encoding for smaller batches
                image_b64_list = []
                total_size = 0
                
                for i, image in enumerate(all_images):
                    _, image_b64, buffer_size = encode_single_image((i, image))
                    image_b64_list.append(image_b64)
                    total_size += buffer_size
                    
                    if i % 4 == 0:  # Print every 4th image to reduce spam
                        print(f"  Encoded image {i+1}/{len(all_images)} ({buffer_size/1024:.1f}KB)")
            
            encoding_time = time.time() - encoding_start
            avg_size = total_size / len(all_images) if all_images else 0
            print(f"📦 Encoding complete in {encoding_time:.1f}s - Total: {total_size/1024/1024:.1f}MB, Avg: {avg_size/1024:.1f}KB per image")
            
            elapsed = time.time() - start_time
            avg_time = elapsed / len(all_images) if all_images else 0
            
            # Determine generation method
            if multi_gpu and has_multi_backends and batch_size > 4:
                method = "multi_gpu_img2img_variations"
            else:
                method = "sequential_img2img_variations"
            
            print(f"✅ {method.replace('_', ' ').title()} complete in {elapsed:.1f}s ({avg_time:.2f}s per image)")
            
            # Create list of seeds that were used for debugging
            seeds_used = [base_seed]  # Base image seed
            for i in range(1, len(all_images)):
                seeds_used.append(base_seed + i)  # Variation seeds
            
            return BatchImageResponse(
                images=image_b64_list,
                metadata={
                    "prompt": request.prompt,
                    "parameters": gen_params,
                    "batch_size": len(image_b64_list),
                    "animation_ready": True,
                    "generation_time": elapsed,
                    "generation_method": method,
                    "multi_gpu": multi_gpu and has_multi_backends and batch_size > 4,
                    "gpu_count": len(app_state.get("gpu_list", [])) if multi_gpu and has_multi_backends else 1,
                    "seed": request.seed,
                    "base_seed": base_seed,
                    "seeds_used": seeds_used,
                    "variation_method": "img2img_low_strength",
                    "variation_strength": 0.15
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
               gpus: Optional[str] = None):
    """Run the image generation server.
    
    Args:
        backend_type: Type of backend to use
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        enable_auth: Whether to enable authentication
        api_key: API key for authentication
        gpus: GPU selection - 'auto', '0', '1', '0,1', or specific GPU IDs
    """
    # Store GPU selection for backend creation
    app_state["gpu_selection"] = gpus
    
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
    
    run_server(
        backend_type=args.backend,
        host=args.host,
        port=args.port,
        workers=args.workers,
        enable_auth=args.auth,
        api_key=args.api_key,
        gpus=args.gpus
    )
