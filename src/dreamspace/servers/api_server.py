"""FastAPI server for hosting image generation models."""

import asyncio
import base64
import time
from io import BytesIO
import random
import shutil
import traceback
import time
import torch
import threading
import os
import uuid
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

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
    model: Optional[str] = Field("sd15_server", description="Model to use: 'sd15_server', 'sd21_server', or 'kandinsky21_server'")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
    seed: Optional[int] = Field(None, description="Seed for reproducibility")
    batch_size: Optional[int] = Field(1, description="Number of images to generate (1-32)")


class GenerateBatchRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    model: Optional[str] = Field("sd15_server", description="Model to use: 'sd15_server', 'sd21_server', or 'kandinsky21_server'")
    batch_size: int = Field(..., description="Number of variations to generate (1-32)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
    seed: Optional[int] = Field(None, description="Base seed for variations")
    noise_magnitude: Optional[float] = Field(0.3, description="Magnitude of noise for latent variations")
    bifurcation_step: Optional[int] = Field(3, description="Number of steps from end to bifurcate in bifurcated wiggle")
    output_format: Optional[str] = Field("png", description="Output format: 'png' (base64), 'jpeg' (base64), or 'tensor' (numpy)")
    latent_cookie: Optional[int] = Field(None, description="Cookie for shared latent across batches")


class GenerateInterpolatedEmbeddingsRequest(BaseModel):
    prompt1: str = Field(..., description="Starting text prompt for interpolation")
    prompt2: str = Field(..., description="Ending text prompt for interpolation")
    model: Optional[str] = Field("sd15_server", description="Model to use: 'sd15_server', 'sd21_server', or 'kandinsky21_server'")
    batch_size: int = Field(..., description="Number of interpolation steps (including start and end)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
    seed: Optional[int] = Field(None, description="Base seed for variations")
    output_format: Optional[str] = Field("png", description="Output format: 'png' (base64), 'jpeg' (base64), or 'tensor' (numpy)")
    latent_cookie: Optional[int] = Field(None, description="Cookie for shared latent across batches")


class AsyncMultiPromptRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts for multi-prompt interpolation sequence (minimum 2 prompts)")
    output_dir: str = Field(..., description="Directory where PNG files will be saved")
    model: Optional[str] = Field("sd15_server", description="Model to use: 'sd15_server', 'sd21_server', or 'kandinsky21_server'")
    batch_size: int = Field(8, description="Number of interpolation steps per prompt segment (1-32)")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Image width")
    height: Optional[int] = Field(768, description="Image height")
    seed: Optional[int] = Field(None, description="Random seed for consistent generation")
    latent_cookie: Optional[int] = Field(None, description="Cookie for shared latent across all segments")


class AsyncAdaptiveMultiPromptRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts for multi-prompt interpolation sequence (minimum 2 prompts)")
    output_dir: str = Field(..., description="Directory where PNG files will be saved")
    model: Optional[str] = Field("sd15_server", description="Model to use: 'sd15_server', 'sd21_server', or 'kandinsky21_server'")
    base_batch_size: int = Field(100, description="Initial number of uniform interpolation steps per prompt segment")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    width: Optional[int] = Field(768, description="Final image width")
    height: Optional[int] = Field(768, description="Final image height")
    seed: Optional[int] = Field(None, description="Random seed for consistent generation")
    latent_cookie: Optional[int] = Field(None, description="Cookie for shared latent across all segments")
    metric: Optional[str] = Field("mse", description="Perceptual metric: 'lpips', 'ssim', or 'mse'")
    threshold: Optional[float] = Field(None, description="If set, subdivide until adjacent distances <= threshold")
    target_frames_per_segment: Optional[int] = Field(None, description="If set, resample to exactly K frames per segment using importance resampling")
    preview_size: int = Field(256, description="Preview resolution for metric computation")
    max_depth: int = Field(5, description="Maximum refinement rounds for threshold mode")
    save_intermediate: bool = Field(False, description="If true, save preview frames to _preview for debugging")
    max_frames_total: Optional[int] = Field(None, description="Optional global cap on frames across all segments")


class BatchImageResponse(BaseModel):
    images: List[str] = Field(..., description="List of base64 encoded images or serialized tensors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")


class AsyncResponse(BaseModel):
    success: bool = Field(..., description="Whether the async job was started successfully")
    job_id: str = Field(..., description="Unique identifier for the background job")
    message: str = Field(..., description="Human-readable status message")
    estimated_frames: Optional[int] = Field(None, description="Estimated total number of frames to be generated")
    estimated_duration: Optional[str] = Field(None, description="Estimated time to completion")
    output_dir: str = Field(..., description="Directory where PNG files will be saved")


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
    available_models: Optional[List[str]] = None


class ModelInfo(BaseModel):
    name: str
    backend_type: str
    device: str
    memory_usage: Optional[str] = None


class SwitchModelRequest(BaseModel):
    model: str = Field(..., description="Model to switch to: 'sd15_server', 'sd21_server', or 'kandinsky21_server'")


class SwitchModelResponse(BaseModel):
    success: bool
    message: str
    current_model: str


# Global state
app_state = {}


def get_available_models():
    """Get list of available models."""
    return ["sd15_server", "sd21_server", "kandinsky21_server"]


def get_model_backend(model: str = None):
    """Get the backend for the specified model, creating it if necessary."""
    if model is None:
        model = app_state.get("current_model", "sd15_server")
    
    # Validate model
    if model not in get_available_models():
        raise ValueError(f"Unknown model: {model}. Available models: {get_available_models()}")
    
    # Check if model is already loaded
    model_cache = app_state.get("model_cache", {})
    if model in model_cache:
        print(f"🔄 Using cached model: {model}")
        return model_cache[model]
    
    # Create new model backend
    try:
        config = app_state.get("config") or Config()
        gpu_selection = app_state.get("gpu_selection", "auto")
        disable_safety_checker = app_state.get("disable_safety_checker", False)
        
        # Prepare backend kwargs
        backend_kwargs = {}
        
        # Add safety checker configuration for SD models
        if model in ["sd15_server", "sd21_server"] and disable_safety_checker:
            backend_kwargs["disable_safety_checker"] = True
        
        # Add GPU configuration if specified
        if gpu_selection != "auto":
            first_gpu = gpu_selection.split(',')[0]
            backend_kwargs["device"] = f"cuda:{first_gpu}"
        
        print(f"🔮 Loading model: {model}")
        img_gen = ImgGen(backend=model, config=config, **backend_kwargs)
        
        # Cache the model
        if "model_cache" not in app_state:
            app_state["model_cache"] = {}
        app_state["model_cache"][model] = img_gen
        app_state["current_model"] = model
        
        print(f"✅ Model loaded successfully: {model}")
        return img_gen
        
    except Exception as e:
        print(f"❌ Failed to load model {model}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model}: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = Config()
    backend_type = app_state.get("backend_type", "sd15_server")  # Default to SD 1.5
    gpu_selection = app_state.get("gpu_selection", "auto")
    disable_safety_checker = app_state.get("disable_safety_checker", False)
    
    try:
        # Initialize with default model
        app_state["config"] = config
        app_state["current_model"] = backend_type
        
        # Load default model
        get_model_backend(backend_type)
        
        print(f"✅ API server initialized with default model: {backend_type}")
    except Exception as e:
        print(f"❌ Failed to initialize API server: {e}")
        app_state["img_gen"] = None
    
    yield
    
    # Shutdown - Clean up all cached models
    model_cache = app_state.get("model_cache", {})
    for model_name, img_gen in model_cache.items():
        if img_gen and hasattr(img_gen.backend, 'cleanup'):
            img_gen.backend.cleanup()
            print(f"🧹 Cleaned up model: {model_name}")
    
    # Clean up additional GPU backends (if any)
    if "multi_backends" in app_state:
        multi_backends = app_state["multi_backends"]
        for gpu_id, backend in multi_backends.items():
            if backend and hasattr(backend.backend, 'cleanup'):
                backend.backend.cleanup()
        print(f"🧹 Cleaned up {len(multi_backends)} GPU backends")


def _async_multi_prompt_worker(job_id: str, request: AsyncMultiPromptRequest):
    """Background worker function for async multi-prompt generation.
    
    This function handles the entire multi-prompt interpolation sequence,
    saving PNG files directly to disk as they are generated.
    """
    try:
        print(f"🌈 [Job {job_id[:8]}] Starting async multi-prompt generation")
        print(f"📝 [Job {job_id[:8]}] Prompts: {request.prompts}")
        print(f"📁 [Job {job_id[:8]}] Output directory: {request.output_dir}")
        print(f"🎯 [Job {job_id[:8]}] Requested model: {request.model}")
        
        start_time = time.time()
        
        # Get the backend
        img_gen = get_model_backend(request.model)
        
        # Prepare output directory
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Clear directory contents
        for filename in os.listdir(request.output_dir):
            file_path = os.path.join(request.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️ [Job {job_id[:8]}] Failed to delete {file_path}: {e}")
        
        print(f"🗑️ [Job {job_id[:8]}] Cleared output directory")
        
        # Prepare generation parameters
        gen_params = {
            'guidance_scale': request.guidance_scale,
            'num_inference_steps': request.num_inference_steps,
            'width': request.width,
            'height': request.height,
            'output_format': 'pil'  # Always use PIL for PNG saving
        }
        
        # Set up shared latent cookie and seed
        if request.latent_cookie is not None:
            latent_cookie = request.latent_cookie
            print(f"🍪 [Job {job_id[:8]}] Using user-specified latent cookie {latent_cookie}")
        else:
            latent_cookie = random.randint(1, 1000000)
            print(f"🍪 [Job {job_id[:8]}] Generated latent cookie {latent_cookie}")
        
        if request.seed is not None:
            shared_seed = request.seed
            print(f"🎲 [Job {job_id[:8]}] Using user-specified seed {shared_seed}")
        else:
            shared_seed = random.randint(0, 2**32 - 1)
            print(f"🎲 [Job {job_id[:8]}] Generated shared seed {shared_seed}")
        
        gen_params['latent_cookie'] = latent_cookie
        gen_params['seed'] = shared_seed
        
        # Create prompt pairs including loop back to start
        prompts = request.prompts
        num_prompts = len(prompts)
        prompt_pairs = []
        for i in range(num_prompts):
            next_i = (i + 1) % num_prompts
            prompt_pairs.append((prompts[i], prompts[next_i]))
        
        print(f"🔄 [Job {job_id[:8]}] Created {len(prompt_pairs)} interpolation segments (including loop)")
        
        total_frames_saved = 0
        
        # Generate each interpolation segment
        for i, (prompt1, prompt2) in enumerate(prompt_pairs):
            segment_start_time = time.time()
            
            print(f"🔄 [Job {job_id[:8]}] Segment {i+1}/{len(prompt_pairs)}: '{prompt1[:30]}...' → '{prompt2[:30]}...'")
            
            try:
                # Generate interpolated embeddings for this segment
                result = img_gen.backend.generate_interpolated_embeddings(
                    prompt1=prompt1,
                    prompt2=prompt2,
                    batch_size=request.batch_size,
                    **gen_params
                )
                
                frames = result.get('images', [])
                
                if frames:
                    # Save each frame as PNG immediately
                    for frame_idx, frame in enumerate(frames):
                        global_frame_number = total_frames_saved + frame_idx
                        filename = f"frame_{global_frame_number:06d}.png"
                        filepath = os.path.join(request.output_dir, filename)
                        
                        frame.save(filepath, format='PNG', optimize=True)
                        
                        # Print progress every 4 frames or for first/last frames
                        if frame_idx == 0 or frame_idx == len(frames) - 1 or (frame_idx + 1) % 4 == 0:
                            print(f"💾 [Job {job_id[:8]}] Saved {filename}")
                    
                    total_frames_saved += len(frames)
                    
                    segment_duration = time.time() - segment_start_time
                    print(f"✅ [Job {job_id[:8]}] Segment {i+1} complete: {len(frames)} frames in {segment_duration:.1f}s")
                    
                else:
                    print(f"❌ [Job {job_id[:8]}] No frames generated for segment {i+1}")
                    
            except Exception as e:
                print(f"❌ [Job {job_id[:8]}] Error in segment {i+1}: {e}")
                traceback.print_exc()
        
        total_duration = time.time() - start_time
        
        print(f"✅ [Job {job_id[:8]}] Multi-prompt sequence complete!")
        print(f"📊 [Job {job_id[:8]}] Generated {total_frames_saved} total frames in {total_duration:.1f}s")
        print(f"📁 [Job {job_id[:8]}] All frames saved to: {request.output_dir}")
        print(f"🎬 [Job {job_id[:8]}] Ready for video creation: frame_000000.png to frame_{total_frames_saved-1:06d}.png")
        
    except Exception as e:
        print(f"❌ [Job {job_id[:8]}] Fatal error in async multi-prompt worker: {e}")
        traceback.print_exc()


def _downscale_image(img: Image.Image, size: int) -> Image.Image:
    try:
        return img.resize((size, size), Image.LANCZOS)
    except Exception:
        return img.resize((size, size))


def _to_float_np(img: Image.Image):
    try:
        import numpy as _np  # type: ignore
    except Exception:
        return None
    arr = _np.asarray(img.convert('RGB'), dtype=_np.float32) / 255.0
    return arr


def _compute_metric(img_a: Image.Image, img_b: Image.Image, metric: str) -> float:
    metric = (metric or 'mse').lower()
    # LPIPS if available
    if metric == 'lpips':
        try:
            import torch as _torch  # type: ignore
            import lpips  # type: ignore
            loss_fn = lpips.LPIPS(net='vgg')
            def pil_to_t(img):
                import numpy as _np  # type: ignore
                arr = _np.asarray(img.convert('RGB'), dtype=_np.float32) / 255.0
                t = _torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
                t = t * 2.0 - 1.0
                return t
            d = loss_fn(pil_to_t(img_a), pil_to_t(img_b))
            return float(d.item())
        except Exception:
            metric = 'ssim'
    if metric == 'ssim':
        try:
            from skimage.metrics import structural_similarity as _ssim  # type: ignore
            a = _to_float_np(img_a)
            b = _to_float_np(img_b)
            if a is None or b is None:
                raise RuntimeError('numpy not available')
            # new skimage uses channel_axis
            try:
                val = _ssim(a, b, channel_axis=2)
            except TypeError:
                val = _ssim(a, b, multichannel=True)
            return float(1.0 - val)
        except Exception:
            metric = 'mse'
    # Fallback MSE (pure PIL, no numpy required)
    from PIL import ImageChops as _ImageChops  # type: ignore
    a = img_a.convert('RGB')
    b = img_b.convert('RGB')
    w, h = a.size
    total = max(1, w * h * 3)
    sq_sum = 0
    for ch in range(3):
        diff = _ImageChops.difference(a.getchannel(ch), b.getchannel(ch))
        hist = diff.histogram()
        sq_sum += sum((i * i) * c for i, c in enumerate(hist))
    return float(sq_sum / (total * 255.0 * 255.0))


def _importance_resample(alphas: List[float], images: List[Image.Image], target_k: int, metric: str) -> List[float]:
    if target_k <= 2 or len(alphas) <= 2:
        return [alphas[0], alphas[-1]] if target_k == 2 else alphas[:target_k]
    dists = []
    for i in range(len(images) - 1):
        d = _compute_metric(images[i], images[i+1], metric)
        dists.append(max(0.0, d))
    cum = [0.0]
    for d in dists:
        cum.append(cum[-1] + d)
    total = cum[-1] if cum[-1] > 0 else 1e-6
    target = [i * (total / (target_k - 1)) for i in range(target_k)]
    out = []
    j = 0
    for t in target:
        while j < len(cum) - 1 and cum[j+1] < t:
            j += 1
        if j >= len(cum) - 1:
            out.append(alphas[-1])
        else:
            c0, c1 = cum[j], cum[j+1]
            a0, a1 = alphas[j], alphas[j+1]
            if c1 == c0:
                out.append(a0)
            else:
                w = (t - c0) / (c1 - c0)
                out.append(a0 + w * (a1 - a0))
    out[0] = alphas[0]
    out[-1] = alphas[-1]
    # dedup tiny deltas
    dedup = []
    for a in out:
        if not dedup or abs(a - dedup[-1]) > 1e-6:
            dedup.append(a)
    return dedup


def _async_adaptive_multi_prompt_worker(job_id: str, request: AsyncAdaptiveMultiPromptRequest):
    import math
    try:
        print(f"🌈 [Job {job_id[:8]}] Starting adaptive async multi-prompt generation")
        print(f"📝 [Job {job_id[:8]}] Prompts: {request.prompts}")
        print(f"📁 [Job {job_id[:8]}] Output directory: {request.output_dir}")
        print(f"🎯 [Job {job_id[:8]}] Requested model: {request.model}")
        mode_msg = 'threshold=' + str(request.threshold) if request.threshold is not None else ('target_frames=' + str(request.target_frames_per_segment) if request.target_frames_per_segment else 'uniform')
        print(f"🧮 [Job {job_id[:8]}] Mode: {mode_msg} | base_batch_size={request.base_batch_size}")
        start_time = time.time()

        img_gen = get_model_backend(request.model)
        backend = img_gen.backend

        os.makedirs(request.output_dir, exist_ok=True)
        preview_dir = os.path.join(request.output_dir, "_preview") if request.save_intermediate else None
        if preview_dir:
            os.makedirs(preview_dir, exist_ok=True)

        # Clear directory (keep _preview if saving intermediates)
        for filename in os.listdir(request.output_dir):
            if request.save_intermediate and filename == "_preview":
                continue
            path = os.path.join(request.output_dir, filename)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"⚠️ [Job {job_id[:8]}] Failed to delete {path}: {e}")

        print(f"🗑️ [Job {job_id[:8]}] Cleared output directory")

        gen_params = {
            'guidance_scale': request.guidance_scale,
            'num_inference_steps': request.num_inference_steps,
            'output_format': 'pil',
            'latent_cookie': request.latent_cookie if request.latent_cookie is not None else random.randint(1, 1000000),
            'seed': request.seed if request.seed is not None else random.randint(0, 2**32 - 1),
        }
        if request.latent_cookie is None:
            print(f"🍪 [Job {job_id[:8]}] Generated latent cookie {gen_params['latent_cookie']}")
        else:
            print(f"🍪 [Job {job_id[:8]}] Using user-specified latent cookie {gen_params['latent_cookie']}")
        if request.seed is None:
            print(f"🎲 [Job {job_id[:8]}] Generated shared seed {gen_params['seed']}")
        else:
            print(f"🎲 [Job {job_id[:8]}] Using user-specified seed {gen_params['seed']}")

        prompts = request.prompts
        num_prompts = len(prompts)
        prompt_pairs = [(prompts[i], prompts[(i+1) % num_prompts]) for i in range(num_prompts)]
        print(f"🔄 [Job {job_id[:8]}] Created {len(prompt_pairs)} interpolation segments (including loop)")

        metric_name = (request.metric or 'mse').lower()
        base_n = max(2, int(request.base_batch_size))
        preview_size = int(request.preview_size)
        target_K = int(request.target_frames_per_segment) if request.target_frames_per_segment else None
        threshold = float(request.threshold) if request.threshold is not None else None
        max_depth = int(request.max_depth)

        total_frames_saved = 0
        global_frame_number = 0

        def render_alphas(prompt1, prompt2, alphas: List[float], width: int, height: int):
            images = []
            if hasattr(backend, 'generate_interpolated_embeddings_at_alphas'):
                res = backend.generate_interpolated_embeddings_at_alphas(
                    prompt1=prompt1,
                    prompt2=prompt2,
                    alphas=alphas,
                    width=width,
                    height=height,
                    **gen_params
                )
                images = res.get('images', [])
            else:
                bs = max(2, len(alphas))
                res = backend.generate_interpolated_embeddings(
                    prompt1=prompt1,
                    prompt2=prompt2,
                    batch_size=bs,
                    width=width,
                    height=height,
                    **gen_params
                )
                cand = res.get('images', [])
                if bs > 1 and cand:
                    grid = [i/(bs-1) for i in range(bs)]
                    for a in alphas:
                        idx = min(range(bs), key=lambda k: abs(grid[k]-a))
                        images.append(cand[idx])
                elif cand:
                    images.append(cand[0])
            return images

        for seg_idx, (p1, p2) in enumerate(prompt_pairs):
            seg_start = time.time()
            print(f"🔄 [Job {job_id[:8]}] Segment {seg_idx+1}/{len(prompt_pairs)}: '{p1[:30]}...' → '{p2[:30]}...'")

            alphas = [i/(base_n-1) for i in range(base_n)] if base_n > 1 else [0.0]
            preview_imgs = render_alphas(p1, p2, alphas, preview_size, preview_size)
            if request.save_intermediate and preview_dir:
                for idx, img in enumerate(preview_imgs):
                    img.save(os.path.join(preview_dir, f"seg{seg_idx:02d}_pre_{idx:03d}.png"), format='PNG', optimize=True)

            if target_K:
                final_alphas = _importance_resample(alphas, preview_imgs, target_K, metric_name)
                final_alphas = sorted(set(round(a,6) for a in final_alphas))
            elif threshold is not None:
                def pair_dists(imgs):
                    return [_compute_metric(imgs[i], imgs[i+1], metric_name) for i in range(len(imgs)-1)]
                dists = pair_dists(preview_imgs)
                depth = 0
                while depth < max_depth:
                    mids = []
                    for i, dv in enumerate(dists):
                        print("ddddd", dv, threshold)
                        if dv > threshold:
                            mids.append((alphas[i] + alphas[i+1]) / 2.0)
                    mids = sorted(set(round(m,6) for m in mids))
                    if not mids:
                        break
                    mid_imgs = render_alphas(p1, p2, mids, preview_size, preview_size)
                    # merge
                    new_a, new_i = [], []
                    m_idx = 0
                    for i in range(len(preview_imgs)-1):
                        new_a.append(alphas[i]); new_i.append(preview_imgs[i])
                        mid_val = (alphas[i] + alphas[i+1]) / 2.0
                        while m_idx < len(mids) and abs(mids[m_idx] - mid_val) <= 1e-6:
                            new_a.append(mids[m_idx]); new_i.append(mid_imgs[m_idx]); m_idx += 1
                    new_a.append(alphas[-1]); new_i.append(preview_imgs[-1])
                    alphas, preview_imgs = new_a, new_i
                    dists = pair_dists(preview_imgs)
                    depth += 1
                    print(f"  🧪 [Job {job_id[:8]}] Refinement round {depth}: frames={len(alphas)} (violations={sum(1 for x in dists if x>threshold)})")
                    if request.max_frames_total is not None and (total_frames_saved + len(alphas)) > int(request.max_frames_total):
                        print(f"  ⛔ [Job {job_id[:8]}] Max frames total reached during refinement")
                        break
                final_alphas = alphas
            else:
                final_alphas = alphas

            # Full-res render
            full_imgs = []
            CHUNK = 64
            for s in range(0, len(final_alphas), CHUNK):
                sub = final_alphas[s:s+CHUNK]
                imgs = render_alphas(p1, p2, sub, request.width, request.height)
                full_imgs.extend(imgs)

            for frame in full_imgs:
                filename = f"frame_{global_frame_number:06d}.png"
                frame.save(os.path.join(request.output_dir, filename), format='PNG', optimize=True)
                global_frame_number += 1
                total_frames_saved += 1
                if global_frame_number % 8 == 0 or global_frame_number < 4:
                    print(f"💾 [Job {job_id[:8]}] Saved {filename}")

            seg_dur = time.time() - seg_start
            print(f"✅ [Job {job_id[:8]}] Segment {seg_idx+1} complete: {len(full_imgs)} frames in {seg_dur:.1f}s")

        total_duration = time.time() - start_time
        print(f"✅ [Job {job_id[:8]}] Adaptive multi-prompt sequence complete!")
        print(f"📊 [Job {job_id[:8]}] Generated {total_frames_saved} total frames in {total_duration:.1f}s")
        print(f"📁 [Job {job_id[:8]}] All frames saved to: {request.output_dir}")
    except Exception as e:
        print(f"❌ [Job {job_id[:8]}] Fatal error in adaptive async worker: {e}")
        traceback.print_exc()


def create_app(backend_type: str = "sd15_server", 
               enable_auth: bool = False,
               api_key: Optional[str] = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        backend_type: Default backend type to use ("sd15_server", "sd21_server", etc.)
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
    
    def get_img_gen(model: str = None):
        """Get ImgGen instance for the specified model."""
        try:
            return get_model_backend(model)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        try:
            current_model = app_state.get("current_model", "sd15_server")
            img_gen = get_img_gen(current_model)
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
                "gpu_available": gpu_available,
                "available_models": ["sd15_server", "sd21_server", "kandinsky21_server"]
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

    @app.post("/switch_model", response_model=SwitchModelResponse)
    async def switch_model(
        request: SwitchModelRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Switch the active model backend."""
        try:
            # Validate model choice
            if request.model not in ["sd15_server", "sd21_server", "kandinsky21_server"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model '{request.model}'. Available models: sd15_server, sd21_server, kandinsky21_server"
                )
            
            # Load the new model (will be cached)
            img_gen = get_model_backend(request.model)
            
            return SwitchModelResponse(
                success=True,
                message=f"Successfully switched to {request.model}",
                current_model=request.model
            )
            
        except Exception as e:
            return SwitchModelResponse(
                success=False,
                message=f"Failed to switch to {request.model}: {str(e)}",
                current_model=app_state.get("current_model", "unknown")
            )
    
    @app.post("/generate", response_model=ImageResponse)
    async def generate_image(
        request: GenerateRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Generate an image from text prompt."""
        try:
            img_gen = get_model_backend(request.model)
            
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
            
            # Convert to base64 using PNG for lossless quality (default)
            buffer = BytesIO()
            image.save(buffer, format='PNG', optimize=True)
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
            img_gen = get_model_backend(request.model)

            batch_size = request.batch_size

            # Prepare generation parameters
            gen_params = {
                k: v for k, v in request.dict().items() 
                if v is not None and k not in ['prompt', 'batch_size', 'noise_magnitude', 'bifurcation_step', 'output_format', 'latent_cookie']
            }

            # Delegate batch generation with latent wiggle to the backend
            print(f"🎬 Generating batch of {batch_size} images with bifurcated wiggle (default method)...")
            start_time = time.time()
            base_seed = request.seed or 42  # Default seed if not provided

            # Always use bifurcated wiggle method (now the default)
            print(f"🔀 Using bifurcated wiggle with {request.bifurcation_step} refinement steps")
            result = img_gen.backend.generate(
                prompt=request.prompt,
                batch_size=batch_size,
                noise_magnitude=request.noise_magnitude,
                bifurcation_step=request.bifurcation_step,
                output_format="tensor" if request.output_format == "tensor" else "pil",
                latent_cookie=request.latent_cookie,
                **gen_params
            )

            all_images = result['images']
            result_format = result.get('format', 'pil')

            print(f"✅ Batch generation complete: {len(all_images)} images in {result_format} format")

            # Handle different output formats
            if request.output_format == "tensor" and result_format == "torch_tensor":
                # Ultra-fast tensor serialization for local clients
                print("🚀 Serializing tensors for high-speed local transfer...")
                encoding_start = time.time()
                
                # all_images is already a PyTorch tensor - no conversion needed!
                tensor_data = all_images
                
                # Use torch.save to BytesIO - direct tensor serialization
                buffer = BytesIO()
                torch.save(tensor_data, buffer)
                buffer.seek(0)
                tensor_bytes = buffer.getvalue()
                
                # Base64 encode for JSON transfer
                tensor_b64 = base64.b64encode(tensor_bytes).decode('utf-8')
                
                encoding_time = time.time() - encoding_start
                print(f"🎯 Tensor serialization complete in {encoding_time:.3f}s")
                print(f"   Tensor size: {len(tensor_bytes)/1024/1024:.1f}MB")
                print(f"   Base64 size: {len(tensor_b64)/1024/1024:.1f}MB")
                
                return BatchImageResponse(
                    images=[tensor_b64],  # Single serialized tensor containing all images
                    metadata={
                        "prompt": request.prompt,
                        "parameters": gen_params,
                        "batch_size": len(all_images),
                        "animation_ready": True,
                        "generation_time": time.time() - start_time,
                        "generation_method": "bifurcated_wiggle",
                        "output_format": "tensor",
                        "tensor_shape": list(all_images.shape),
                        "tensor_dtype": str(all_images.dtype),
                        "serialization": "torch_save",
                        "noise_magnitude": request.noise_magnitude,
                        "bifurcation_step": request.bifurcation_step
                    }
                )
            
            else:
                # Traditional PIL → Image Format → Base64 pipeline
                format_name = request.output_format.upper() if request.output_format in ['jpeg', 'png'] else 'PNG'
                print(f"📦 Encoding images to {format_name}...")
                encoding_start = time.time()

                def encode_single_image(args):
                    from io import BytesIO  # Import inside function to avoid scope issues
                    import base64
                    import time
                    i, image = args
                    
                    # Time the image encoding
                    encode_start = time.time()
                    
                    if request.output_format == "jpeg_optimized" and hasattr(image, 'dtype'):
                        # Direct numpy → JPEG encoding (skip PIL)
                        import cv2
                        # Convert from float [0,1] to uint8 [0,255] if needed
                        if image.dtype == 'float32' or image.dtype == 'float64':
                            image_array = (image * 255).astype('uint8')
                        else:
                            image_array = image
                        
                        # Direct JPEG encoding
                        _, image_bytes = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        buffer_size = len(image_bytes)
                        encode_time = time.time() - encode_start
                        
                        # Time the base64 encoding
                        b64_start = time.time()
                        image_b64 = base64.b64encode(image_bytes.tobytes()).decode('utf-8')
                        b64_time = time.time() - b64_start
                    else:
                        # Traditional PIL → Image Format encoding
                        buffer = BytesIO()
                        # Use requested format or default to PNG for lossless quality
                        if request.output_format == "jpeg":
                            image.save(buffer, format='JPEG', quality=90, optimize=True)
                        else:  # Default to PNG for lossless quality
                            image.save(buffer, format='PNG', optimize=True)
                        encode_time = time.time() - encode_start
                        
                        # Time the base64 encoding
                        b64_start = time.time()
                        buffer_size = len(buffer.getvalue())
                        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        b64_time = time.time() - b64_start
                    
                    return i, image_b64, buffer_size, encode_time, b64_time

                # Use ThreadPoolExecutor for parallel encoding if we have many images
                if len(all_images) > 8:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=4) as executor:
                        results = list(executor.map(encode_single_image, enumerate(all_images)))

                    # Sort results by original index and extract data
                    results.sort(key=lambda x: x[0])
                    image_b64_list = [r[1] for r in results]
                    total_size = sum(r[2] for r in results)
                    total_encode_time = sum(r[3] for r in results)
                    total_b64_time = sum(r[4] for r in results)

                    print(f"  Parallel encoded {len(all_images)} images")
                    print(f"  📸 Image encoding time: {total_encode_time:.3f}s total, {total_encode_time/len(all_images):.3f}s avg")
                    print(f"  🔤 Base64 encoding time: {total_b64_time:.3f}s total, {total_b64_time/len(all_images):.3f}s avg")
                else:
                    # Sequential encoding for smaller batches
                    image_b64_list = []
                    total_size = 0
                    total_encode_time = 0
                    total_b64_time = 0

                    for i, image in enumerate(all_images):
                        _, image_b64, buffer_size, encode_time, b64_time = encode_single_image((i, image))
                        image_b64_list.append(image_b64)
                        total_size += buffer_size
                        total_encode_time += encode_time
                        total_b64_time += b64_time

                    print(f"  📸 Image encoding time: {total_encode_time:.3f}s total, {total_encode_time/len(all_images):.3f}s avg")
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
                        "generation_method": "bifurcated_wiggle",
                        "multi_gpu": False,  # Using single GPU with batch generation
                        "gpu_count": 1,
                        "seed": request.seed,
                        "base_seed": base_seed,
                        "variation_method": "latent_noise",
                        "output_format": request.output_format or "jpeg",
                        "encoding_time": encoding_time,
                        "noise_magnitude": request.noise_magnitude,
                        "bifurcation_step": request.bifurcation_step
                    }
                )

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"❌ CRITICAL ERROR in generate_batch main logic: {e}")
            print(f"📋 Full traceback: {error_details}")
            raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")
    
    @app.post("/interpolate")
    async def interpolate_embeddings(
        request: InterpolateRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Interpolate between embeddings."""
        try:
            img_gen = get_img_gen()
            backend = img_gen.backend
            
            embedding1 = torch.tensor(request.embedding1)
            embedding2 = torch.tensor(request.embedding2)
            
            result = backend.interpolate_embeddings(embedding1, embedding2, request.alpha)
            
            return {"result": result.tolist() if torch.is_tensor(result) else result}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate_interpolated_embeddings", response_model=BatchImageResponse)
    async def generate_interpolated_embeddings(request: GenerateInterpolatedEmbeddingsRequest):
        """Generate a batch of images using interpolated embeddings between two prompts."""
        try:
            img_gen = get_model_backend(request.model)
            backend = img_gen.backend
            result = backend.generate_interpolated_embeddings(
                prompt1=request.prompt1,
                prompt2=request.prompt2,
                batch_size=request.batch_size,
                output_format=request.output_format,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                width=request.width,
                height=request.height,
                seed=request.seed,
                latent_cookie=request.latent_cookie
            )

            # Handle different output formats similar to generate_batch
            if request.output_format == "tensor" and result.get('format') == 'torch_tensor':
                # Return tensor format for high-speed processing
                import torch
                tensor_data = result['images']
                
                buffer = BytesIO()
                torch.save(tensor_data, buffer)
                buffer.seek(0)
                tensor_bytes = buffer.getvalue()
                tensor_b64 = base64.b64encode(tensor_bytes).decode('utf-8')
                
                return BatchImageResponse(
                    images=[tensor_b64],
                    metadata={
                        **result,
                        "tensor_shape": list(tensor_data.shape),
                        "tensor_dtype": str(tensor_data.dtype),
                        "serialization": "torch_save"
                    }
                )
            else:
                # Convert PIL images to base64
                images_base64 = []
                for image in result["images"]:
                    buffer = BytesIO()
                    if request.output_format == "jpeg":
                        image.save(buffer, format='JPEG', quality=90, optimize=True)
                    else:  # Default to PNG
                        image.save(buffer, format='PNG', optimize=True)
                    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    images_base64.append(image_b64)
                
                return BatchImageResponse(
                    images=images_base64,
                    metadata={
                        "prompt1": request.prompt1,
                        "prompt2": request.prompt2,
                        "batch_size": request.batch_size,
                        "output_format": request.output_format,
                        "generation_method": "interpolated_embeddings"
                    }
                )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate_async_multi_prompt", response_model=AsyncResponse)
    async def generate_async_multi_prompt(
        request: AsyncMultiPromptRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        """Start an asynchronous multi-prompt interpolation sequence generation.
        
        This endpoint immediately starts a background thread to handle the generation
        and returns a job ID. The generation results are saved directly to PNG files
        in the specified output directory.
        """
        try:
            # Validate request
            if len(request.prompts) < 2:
                raise HTTPException(status_code=400, detail="At least 2 prompts are required")
            if request.batch_size < 1 or request.batch_size > 32:
                raise HTTPException(status_code=400, detail="batch_size must be between 1 and 32")
            if request.width < 256 or request.height < 256:
                raise HTTPException(status_code=400, detail="width and height must be at least 256")
            if request.guidance_scale <= 0:
                raise HTTPException(status_code=400, detail="guidance_scale must be positive")
            if request.num_inference_steps < 1:
                raise HTTPException(status_code=400, detail="num_inference_steps must be at least 1")

            job_id = str(uuid.uuid4())
            num_segments = len(request.prompts)
            estimated_frames = num_segments * request.batch_size
            estimated_duration = f"{estimated_frames * 2}s-{estimated_frames * 4}s"

            thread = threading.Thread(
                target=_async_multi_prompt_worker,
                args=(job_id, request),
                daemon=True
            )
            thread.start()

            return AsyncResponse(
                success=True,
                job_id=job_id,
                message=f"Async multi-prompt generation started with {num_segments} segments",
                estimated_frames=estimated_frames,
                estimated_duration=estimated_duration,
                output_dir=request.output_dir
            )
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate_async_adaptive_multi_prompt", response_model=AsyncResponse)
    async def generate_async_adaptive_multi_prompt(
        request: AsyncAdaptiveMultiPromptRequest,
        authenticated: bool = Depends(auth_dependency)
    ):
        try:
            if len(request.prompts) < 2:
                raise HTTPException(status_code=400, detail="At least 2 prompts are required")
            if request.base_batch_size < 2:
                raise HTTPException(status_code=400, detail="base_batch_size must be at least 2")

            job_id = str(uuid.uuid4())
            num_segments = len(request.prompts)
            estimated_frames = num_segments * request.base_batch_size
            estimated_duration = f"{estimated_frames * 2}s-{estimated_frames * 4}s"

            thread = threading.Thread(
                target=_async_adaptive_multi_prompt_worker,
                args=(job_id, request),
                daemon=True
            )
            thread.start()

            return AsyncResponse(
                success=True,
                job_id=job_id,
                message=f"Adaptive multi-prompt generation started with {num_segments} segments",
                estimated_frames=estimated_frames,
                estimated_duration=estimated_duration,
                output_dir=request.output_dir
            )
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
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
