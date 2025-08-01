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
        """Generate a batch of image variations for animation with smart chunking for multi-GPU."""
        try:
            img_gen = get_img_gen()
            
            # Limit batch size for server stability
            batch_size = min(request.batch_size, 32)
            
            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'batch_size']
            }
            
            print(f"🎬 Generating batch of {batch_size} variations...")
            start_time = time.time()
            
            # Smart chunking for memory management and multi-GPU utilization
            # Calculate optimal chunk size based on image dimensions and available memory
            image_pixels = gen_params.get('width', 768) * gen_params.get('height', 768)
            gpu_selection = app_state.get("gpu_selection", "auto")
            
            # Determine if we have multi-GPU setup
            multi_gpu = gpu_selection != "auto" and "," in gpu_selection
            gpu_count = len(gpu_selection.split(",")) if multi_gpu else 1
            
            if image_pixels >= 768 * 768:
                # High-res images: smaller chunks to avoid OOM
                base_chunk_size = 4
            elif image_pixels >= 512 * 512:
                # Medium-res images: moderate chunks
                base_chunk_size = 6
            else:
                # Low-res images: larger chunks
                base_chunk_size = 8
            
            # Adjust chunk size for multi-GPU (can handle larger chunks)
            max_chunk_size = base_chunk_size * gpu_count if multi_gpu else base_chunk_size
            
            # Split batch into chunks
            if batch_size <= max_chunk_size:
                # Small batch - generate all at once
                chunks = [batch_size]
                if multi_gpu:
                    print(f"  🚀 Generating {batch_size} images in single batch across {gpu_count} GPUs...")
                else:
                    print(f"  🚀 Generating {batch_size} images in single batch...")
            else:
                # Large batch - split into chunks
                chunks = []
                remaining = batch_size
                while remaining > 0:
                    chunk_size = min(max_chunk_size, remaining)
                    chunks.append(chunk_size)
                    remaining -= chunk_size
                
                gpu_info = f" across {gpu_count} GPUs" if multi_gpu else ""
                print(f"  🧩 Splitting into {len(chunks)} chunks{gpu_info}: {chunks}")
            
            # Generate images in chunks with multi-GPU distribution
            all_images = []
            base_seed = request.seed
            
            if multi_gpu and len(chunks) > 1 and "multi_backends" in app_state:
                # True parallel multi-GPU processing with separate backend instances
                import threading
                import time
                import traceback
                
                try:
                    multi_backends = app_state["multi_backends"]
                    gpu_list = app_state["gpu_list"]
                    print(f"  🎮 Processing {len(chunks)} chunks in PARALLEL across {len(gpu_list)} GPU backends: {gpu_list}")
                    
                    # Thread-safe results collection
                    results = {}
                    results_lock = threading.Lock()
                    
                    def generate_chunk_parallel(chunk_idx, chunk_size, gpu_id):
                        """Generate a chunk in parallel on a specific GPU backend."""
                        try:
                            backend = multi_backends.get(gpu_id)
                            if backend is None:
                                print(f"  ⚠️ No backend available for GPU {gpu_id}, skipping chunk {chunk_idx+1}")
                                with results_lock:
                                    results[chunk_idx] = []
                                return
                            
                            print(f"  🚀 GPU {gpu_id}: Starting parallel chunk {chunk_idx+1}/{len(chunks)} ({chunk_size} images)...")
                            chunk_start_time = time.time()
                            
                            # Use different seeds for each chunk to ensure variety
                            batch_params = {**gen_params}
                            batch_params['num_images_per_prompt'] = chunk_size
                            
                            if base_seed is not None:
                                # Offset seed for each chunk to get variations
                                batch_params['seed'] = base_seed + chunk_idx * 1000
                            
                            print(f"  🔧 GPU {gpu_id}: Calling backend.gen() with params: {batch_params}")
                            
                            # Generate chunk using the specific GPU backend
                            chunk_images = backend.gen(prompt=request.prompt, **batch_params)
                            
                            print(f"  📦 GPU {gpu_id}: Backend returned {type(chunk_images)} with length/content: {len(chunk_images) if isinstance(chunk_images, list) else 'single item'}")
                            
                            # Ensure we have a list
                            if not isinstance(chunk_images, list):
                                chunk_images = [chunk_images]
                            
                            # Store results thread-safely
                            with results_lock:
                                results[chunk_idx] = chunk_images
                            
                            chunk_elapsed = time.time() - chunk_start_time
                            print(f"  ✅ GPU {gpu_id}: Parallel chunk {chunk_idx+1} complete in {chunk_elapsed:.1f}s ({len(chunk_images)} images)")
                            
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            print(f"  ❌ GPU {gpu_id}: Parallel chunk {chunk_idx+1} failed with exception: {e}")
                            print(f"  📋 GPU {gpu_id}: Full traceback: {error_details}")
                            with results_lock:
                                results[chunk_idx] = []
                    
                    # Create and start threads for parallel execution
                    threads = []
                    parallel_start_time = time.time()
                    
                    for i, chunk_size in enumerate(chunks):
                        gpu_id = gpu_list[i % len(gpu_list)]  # Round-robin GPU assignment
                        thread = threading.Thread(
                            target=generate_chunk_parallel,
                            args=(i, chunk_size, gpu_id),
                            name=f"GPU-{gpu_id}-Chunk-{i+1}"
                        )
                        threads.append(thread)
                        thread.start()
                        print(f"  🔥 Started parallel thread for GPU {gpu_id}, chunk {i+1}")
                    
                    # Wait for all parallel threads to complete
                    print(f"  ⏳ Waiting for {len(threads)} parallel GPU threads to complete...")
                    for i, thread in enumerate(threads):
                        thread.join()
                        print(f"  📋 Thread {i+1}/{len(threads)} completed ({thread.name})")
                    
                    parallel_elapsed = time.time() - parallel_start_time
                    
                    # Collect results in order
                    for i in range(len(chunks)):
                        if i in results:
                            all_images.extend(results[i])
                        else:
                            print(f"    ⚠️ Missing results for chunk {i+1}")
                    
                    print(f"  🎯 Parallel multi-GPU processing complete in {parallel_elapsed:.1f}s: {len(all_images)} total images")
                    print(f"  ⚡ Parallel speedup: {len(chunks)} chunks processed simultaneously instead of sequentially")
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"  ❌ CRITICAL: Parallel multi-GPU processing failed: {e}")
                    print(f"  📋 Full traceback: {error_details}")
                    # Fall back to sequential processing
                    print(f"  🔄 Falling back to sequential single-GPU processing...")
                    multi_gpu = False
                
            if not multi_gpu or len(chunks) <= 1 or "multi_backends" not in app_state:
                # Single GPU or single chunk processing
                for i, chunk_size in enumerate(chunks):
                    print(f"  🚀 Generating chunk {i+1}/{len(chunks)}: {chunk_size} images...")
                    
                    # Use different seeds for each chunk to ensure variety
                    batch_params = {**gen_params}
                    batch_params['num_images_per_prompt'] = chunk_size
                    
                    if base_seed is not None:
                        # Offset seed for each chunk to get variations
                        batch_params['seed'] = base_seed + i * 1000
                    
                    # Generate chunk
                    chunk_images = img_gen.gen(prompt=request.prompt, **batch_params)
                    
                    # Ensure we have a list
                    if not isinstance(chunk_images, list):
                        chunk_images = [chunk_images]
                    
                    all_images.extend(chunk_images)
                    print(f"  ✅ Chunk {i+1} complete: {len(chunk_images)} images")
            
            print(f"✅ Generated {len(all_images)} images total using chunked batching")
            
            # Convert all images to base64
            image_b64_list = []
            for i, image in enumerate(all_images):
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_b64_list.append(image_b64)
                if i % 4 == 0:  # Print every 4th image to reduce spam
                    print(f"  Encoded image {i+1}/{len(all_images)}")
            
            elapsed = time.time() - start_time
            avg_time = elapsed / len(all_images) if all_images else 0
            method = "multi_gpu_backend_batch" if multi_gpu and len(chunks) > 1 and "multi_backends" in app_state else "chunked_batch"
            print(f"✅ {method.replace('_', ' ').title()} complete in {elapsed:.1f}s ({avg_time:.2f}s per image)")
            
            return BatchImageResponse(
                images=image_b64_list,
                metadata={
                    "prompt": request.prompt,
                    "parameters": gen_params,
                    "batch_size": len(image_b64_list),
                    "animation_ready": True,
                    "generation_time": elapsed,
                    "generation_method": method,
                    "chunks": len(chunks),
                    "chunk_sizes": chunks,
                    "gpu_count": gpu_count,
                    "multi_gpu": multi_gpu,
                    "seed": request.seed
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
