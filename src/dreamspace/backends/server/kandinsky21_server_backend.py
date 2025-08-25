"""Kandinsky 2.1 server backend - rewritten for proper Kandinsky architecture."""

import math
import time

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image

from ...core.base import ImgGenBackend
from ...core.utils import no_grad_method
from ...config.settings import Config, ModelConfig


class Kandinsky21ServerBackend(ImgGenBackend):
    """Kandinsky 2.1 server backend using proper two-stage architecture.

    Kandinsky uses a two-stage process:
    1. Prior: text -> image embeddings
    2. Decoder: image embeddings -> images
    
    This backend maintains the same high-level interface as SD2.1 but uses
    Kandinsky's native workflow for better compatibility and performance.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        disable_safety_checker: bool = False,
    ):
        """Initialize the Kandinsky 2.1 server backend.

        Args:
            config: Configuration instance
            device: Device to run on (overrides config)
            disable_safety_checker: Not used for Kandinsky
        """
        self.config = config or Config()
        self.device = device or "cuda"
        self.model_id = "kandinsky-community/kandinsky-2-1"
        self.disable_safety_checker = disable_safety_checker

        # Cache for image embeddings (replaces latent cache)
        self.image_embedding_cache = {}  # cookie -> image embedding tensor

        self._load_pipelines()

    def _calculate_sub_batch_size(
        self, total_batch_size: int, width: int, height: int, quiet: bool = False
    ) -> int:
        """Calculate optimal sub-batch size based on memory heuristics.

        Args:
            total_batch_size: Total number of images to generate
            width: Image width in pixels
            height: Image height in pixels
            quiet: If True, suppress progress output

        Returns:
            Optimal sub-batch size (1 to total_batch_size)
        """
        # Calculate megapixels per image
        megapixels = (width * height) / 1_000_000

        # Kandinsky is very memory-intensive due to two-stage process
        available_memory_gb = 4.0  # Conservative for Kandinsky

        # Kandinsky memory estimates:
        # - Prior model: ~1GB base + text processing
        # - Decoder model: ~3GB base + image generation
        # - Image embeddings: much smaller than SD latents
        memory_per_image_mb = megapixels * 50  # Much more conservative
        memory_per_image_gb = memory_per_image_mb / 1000

        # Calculate how many images we can fit in memory
        max_parallel_images = max(1, int(available_memory_gb / memory_per_image_gb))

        # Don't exceed the total batch size
        sub_batch_size = min(max_parallel_images, total_batch_size)

        # Very conservative limits for Kandinsky
        sub_batch_size = max(1, min(sub_batch_size, 2))  # Never more than 2 per sub-batch

        if not quiet:
            print(
                f"📊 Kandinsky 2.1 Memory heuristic for {width}x{height} ({megapixels:.1f}MP):"
            )
            print(
                f"   Estimated {memory_per_image_gb:.2f}GB per image (Kandinsky is very memory-intensive)"
            )
            print(
                f"   Sub-batch size: {sub_batch_size} (from total {total_batch_size})"
            )

        return sub_batch_size

    def _load_pipelines(self):
        """Load the Kandinsky pipelines using proper two-stage approach."""
        try:
            from diffusers import AutoPipelineForText2Image
            
            print(f"🔮 Loading Kandinsky 2.1 from {self.model_id} on {self.device}...")

            # Load the combined pipeline
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            print(f"  📍 Kandinsky pipeline moved to {self.device}")

            # Enable memory optimizations if available
            try:
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing(1)
                    print("✅ Attention slicing enabled")
            except Exception:
                print("⚠️ Attention slicing not available")

            try:
                if hasattr(self.pipe, 'enable_model_cpu_offload'):
                    self.pipe.enable_model_cpu_offload()
                    print("✅ Model CPU offloading enabled")
            except Exception:
                print("⚠️ CPU offloading not available")

            # Set models to eval mode
            if hasattr(self.pipe, 'prior') and self.pipe.prior is not None:
                self.pipe.prior.eval()
            if hasattr(self.pipe, 'decoder') and self.pipe.decoder is not None:
                self.pipe.decoder.eval()
            if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                self.pipe.text_encoder.eval()

            print(f"✅ Kandinsky 2.1 loaded successfully on {self.device}!")

        except Exception as e:
            print(f"❌ Failed to load Kandinsky 2.1: {e}")
            raise

    @no_grad_method
    def generate(
        self,
        prompt: str,
        batch_size: int = 1,
        output_format: str = "pil",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate images from a single prompt using Kandinsky's native pipeline.

        Args:
            prompt: Text prompt for image generation
            batch_size: Number of images to generate
            output_format: Format of output images ("pil", "tensor", "jpeg", "png")
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated images and metadata
        """
        total_start = time.time()
        print(f"🎨 Generating {batch_size} images with Kandinsky 2.1: '{prompt}'")

        # Set up generation parameters with Kandinsky defaults
        if "generator" not in kwargs and "seed" in kwargs:
            seed = kwargs.pop("seed")
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        # Kandinsky defaults
        height = kwargs.get("height", 768)
        width = kwargs.get("width", 768)
        num_inference_steps = kwargs.get("num_inference_steps", 100)
        guidance_scale = kwargs.get("guidance_scale", 4.0)

        # Calculate sub-batching if needed
        sub_batch_size = self._calculate_sub_batch_size(batch_size, width, height)

        if sub_batch_size < batch_size:
            # Sub-batch the generation
            print(f"🔄 Using sub-batching: {batch_size} images in chunks of {sub_batch_size}")
            all_images = []
            
            for start_idx in range(0, batch_size, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, batch_size)
                current_batch_size = end_idx - start_idx
                
                print(f"🔄 Processing sub-batch {start_idx//sub_batch_size + 1}/{math.ceil(batch_size/sub_batch_size)}")
                
                # Generate sub-batch
                sub_result = self.pipe(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=current_batch_size,
                    output_type="pil",
                    **kwargs
                )
                
                all_images.extend(sub_result.images)
            
            images = all_images
        else:
            # Generate all at once
            result = self.pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=batch_size,
                output_type="pil",
                **kwargs
            )
            images = result.images

        # Handle output format
        if output_format == "tensor":
            # Convert PIL to tensor
            tensor_images = []
            for img in images:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
                tensor_images.append(img_tensor)
            images = torch.stack(tensor_images)
        elif output_format in ["jpeg", "png"]:
            # Keep as PIL for now, format conversion handled by API server
            pass

        total_time = time.time() - total_start
        print(f"✅ Generation completed in {total_time:.3f}s")

        return {
            "images": images,
            "total_time": total_time,
            "batch_size": batch_size,
            "metadata": {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
        }

    @no_grad_method
    def _extract_text_embeddings(self, prompt: str) -> torch.Tensor:
        """Extract text embeddings using Kandinsky's text encoder.
        
        For Kandinsky, this returns the text encoder output that can be
        used for interpolation before going to the prior model.
        """
        try:
            if hasattr(self.pipe, 'text_encoder') and hasattr(self.pipe, 'tokenizer'):
                # Use Kandinsky's text encoder directly
                text_inputs = self.pipe.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                text_embeddings = self.pipe.text_encoder(
                    text_inputs.input_ids.to(self.device)
                )[0]
                
                return text_embeddings
            else:
                print("⚠️ Text encoder not available, using fallback")
                return None
        except Exception as e:
            print(f"⚠️ Failed to extract text embeddings: {e}")
            return None

    def _extract_image_embeddings(self, prompt: str, **kwargs) -> torch.Tensor:
        """Extract image embeddings using Kandinsky's prior model.
        
        This is Kandinsky-specific: converts text to image embeddings
        using the prior model. These embeddings can then be interpolated.
        """
        try:
            # Use the prior pipeline to get image embeddings
            if hasattr(self.pipe, 'prior'):
                # Set up generation parameters
                if "generator" not in kwargs and "seed" in kwargs:
                    seed = kwargs.pop("seed")
                    kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
                
                # Get image embeddings from prior
                image_embeddings = self.pipe.prior(
                    prompt,
                    num_inference_steps=kwargs.get("prior_num_inference_steps", 25),
                    generator=kwargs.get("generator"),
                    guidance_scale=kwargs.get("prior_guidance_scale", 1.0),
                ).image_embeds
                
                return image_embeddings
            else:
                # Fallback: try to use the full pipeline to get embeddings
                # This is less efficient but more compatible
                print("⚠️ Using fallback method for image embeddings")
                return None
        except Exception as e:
            print(f"⚠️ Failed to extract image embeddings: {e}")
            return None

    def interpolate_embeddings(
        self, embedding1: Any, embedding2: Any, alpha: float
    ) -> Any:
        """Interpolate between two embeddings.
        
        For Kandinsky, we use linear interpolation (LERP) instead of SLERP
        since Kandinsky embeddings are not necessarily normalized to unit length.
        """
        if embedding1 is None or embedding2 is None:
            return None
        
        # Use linear interpolation for Kandinsky
        return embedding1 * (1 - alpha) + embedding2 * alpha

    @no_grad_method
    def generate_interpolated_embeddings(
        self,
        prompt1: str,
        prompt2: str,
        batch_size: int,
        output_format: str = "pil",
        latent_cookie: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate interpolated images between two prompts using embedding interpolation.
        
        For Kandinsky, this works by:
        1. Converting prompts to image embeddings via prior model
        2. Interpolating between image embeddings
        3. Using decoder to generate images from interpolated embeddings
        """
        total_start = time.time()
        print(f"🌈 Kandinsky interpolation: '{prompt1}' → '{prompt2}' with {batch_size} steps")

        # Set up generation parameters
        if "generator" not in kwargs and "seed" in kwargs:
            seed = kwargs.pop("seed")
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        # Kandinsky defaults
        height = kwargs.get("height", 768)
        width = kwargs.get("width", 768)
        num_inference_steps = kwargs.get("num_inference_steps", 100)
        guidance_scale = kwargs.get("guidance_scale", 4.0)

        # Calculate sub-batching
        sub_batch_size = self._calculate_sub_batch_size(batch_size, width, height)

        if sub_batch_size < batch_size:
            return self._generate_interpolated_embeddings_with_sub_batching(
                prompt1, prompt2, batch_size, sub_batch_size, output_format, latent_cookie, **kwargs
            )

        # Extract image embeddings for both prompts
        encode_start = time.time()
        
        # Check cache first
        cache_key1 = f"{latent_cookie}_{prompt1}" if latent_cookie else None
        cache_key2 = f"{latent_cookie}_{prompt2}" if latent_cookie else None
        
        if cache_key1 and cache_key1 in self.image_embedding_cache:
            print(f"🍪 Using cached image embedding for prompt1")
            embedding1 = self.image_embedding_cache[cache_key1]
        else:
            embedding1 = self._extract_image_embeddings(prompt1, **kwargs)
            if embedding1 is None:
                raise ValueError(f"Failed to extract image embeddings for prompt1: '{prompt1}'")
            if cache_key1:
                self.image_embedding_cache[cache_key1] = embedding1.clone()

        if cache_key2 and cache_key2 in self.image_embedding_cache:
            print(f"🍪 Using cached image embedding for prompt2")
            embedding2 = self.image_embedding_cache[cache_key2]
        else:
            embedding2 = self._extract_image_embeddings(prompt2, **kwargs)
            if embedding2 is None:
                raise ValueError(f"Failed to extract image embeddings for prompt2: '{prompt2}'")
            if cache_key2:
                self.image_embedding_cache[cache_key2] = embedding2.clone()

        encode_time = time.time() - encode_start
        print(f"📝 Image embedding extraction completed in {encode_time:.3f}s")

        # Create interpolated embeddings
        alphas = torch.linspace(0, 1, steps=batch_size)
        interpolated_embeddings = []
        
        for alpha in alphas:
            interpolated = self.interpolate_embeddings(embedding1, embedding2, float(alpha))
            interpolated_embeddings.append(interpolated)
        
        # Stack embeddings for batch processing
        batch_image_embeds = torch.cat(interpolated_embeddings, dim=0)

        # Generate images from interpolated embeddings using decoder
        generation_start = time.time()
        
        try:
            # Use the decoder part of the pipeline
            if hasattr(self.pipe, 'decoder') and self.pipe.decoder is not None:
                # Direct decoder usage
                result = self.pipe.decoder(
                    image_embeds=batch_image_embeds,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_type="pil",
                    **{k: v for k, v in kwargs.items() if k not in ['seed']}  # Remove seed since we use generator
                )
                images = result.images
            else:
                # Fallback: generate images one by one
                print("⚠️ Using fallback generation method")
                images = []
                for i, embedding in enumerate(interpolated_embeddings):
                    # This is less efficient but more compatible
                    alpha = float(alphas[i])
                    interpolated_prompt = f"Interpolation at alpha={alpha:.3f}"
                    
                    result = self.pipe(
                        interpolated_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=1,
                        output_type="pil",
                        **{k: v for k, v in kwargs.items() if k not in ['seed']}
                    )
                    images.extend(result.images)
                    
        except Exception as e:
            print(f"⚠️ Decoder generation failed: {e}")
            raise ValueError(f"Failed to generate images from interpolated embeddings: {e}")

        generation_time = time.time() - generation_start
        print(f"🎨 Image generation completed in {generation_time:.3f}s")

        # Handle output format
        if output_format == "tensor":
            tensor_images = []
            for img in images:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                tensor_images.append(img_tensor)
            images = torch.stack(tensor_images)

        total_time = time.time() - total_start
        print(f"✅ Total Kandinsky interpolation time: {total_time:.3f}s")

        return {
            "images": images,
            "metadata": {
                "alphas": [float(a) for a in alphas.tolist()],
                "prompt1": prompt1,
                "prompt2": prompt2,
            },
            "total_time": total_time,
            "batch_size": batch_size,
        }

    def _generate_interpolated_embeddings_with_sub_batching(
        self,
        prompt1: str,
        prompt2: str,
        total_batch_size: int,
        sub_batch_size: int,
        output_format: str,
        latent_cookie: Optional[int],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate interpolated embeddings using sub-batching for memory efficiency."""
        total_start = time.time()
        print(f"🔄 Sub-batching {total_batch_size} images into chunks of {sub_batch_size}")

        # Extract embeddings once (shared across all sub-batches)
        encode_start = time.time()
        
        cache_key1 = f"{latent_cookie}_{prompt1}" if latent_cookie else None
        cache_key2 = f"{latent_cookie}_{prompt2}" if latent_cookie else None
        
        if cache_key1 and cache_key1 in self.image_embedding_cache:
            embedding1 = self.image_embedding_cache[cache_key1]
        else:
            embedding1 = self._extract_image_embeddings(prompt1, **kwargs)
            if embedding1 is None:
                raise ValueError(f"Failed to extract image embeddings for prompt1")
            if cache_key1:
                self.image_embedding_cache[cache_key1] = embedding1.clone()

        if cache_key2 and cache_key2 in self.image_embedding_cache:
            embedding2 = self.image_embedding_cache[cache_key2]
        else:
            embedding2 = self._extract_image_embeddings(prompt2, **kwargs)
            if embedding2 is None:
                raise ValueError(f"Failed to extract image embeddings for prompt2")
            if cache_key2:
                self.image_embedding_cache[cache_key2] = embedding2.clone()

        # Create all interpolated embeddings
        alphas = torch.linspace(0, 1, steps=total_batch_size)
        all_interpolated_embeddings = []
        for alpha in alphas:
            interpolated = self.interpolate_embeddings(embedding1, embedding2, float(alpha))
            all_interpolated_embeddings.append(interpolated)

        encode_time = time.time() - encode_start
        print(f"📝 Embedding interpolation completed in {encode_time:.3f}s")

        # Process in sub-batches
        all_images = []
        for start_idx in range(0, total_batch_size, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, total_batch_size)
            current_batch_size = end_idx - start_idx

            print(f"🔄 Processing sub-batch {start_idx//sub_batch_size + 1}/{math.ceil(total_batch_size/sub_batch_size)}")

            # Get sub-batch embeddings
            sub_batch_embeddings = all_interpolated_embeddings[start_idx:end_idx]
            batch_image_embeds = torch.cat(sub_batch_embeddings, dim=0)

            # Generate images for this sub-batch
            try:
                if hasattr(self.pipe, 'decoder') and self.pipe.decoder is not None:
                    result = self.pipe.decoder(
                        image_embeds=batch_image_embeds,
                        height=kwargs.get("height", 768),
                        width=kwargs.get("width", 768),
                        num_inference_steps=kwargs.get("num_inference_steps", 100),
                        guidance_scale=kwargs.get("guidance_scale", 4.0),
                        output_type="pil",
                        **{k: v for k, v in kwargs.items() if k not in ['seed']}
                    )
                    all_images.extend(result.images)
                else:
                    # Fallback method
                    for embedding in sub_batch_embeddings:
                        result = self.pipe(
                            f"Interpolated prompt",
                            height=kwargs.get("height", 768),
                            width=kwargs.get("width", 768),
                            num_inference_steps=kwargs.get("num_inference_steps", 100),
                            guidance_scale=kwargs.get("guidance_scale", 4.0),
                            num_images_per_prompt=1,
                            output_type="pil",
                            **{k: v for k, v in kwargs.items() if k not in ['seed']}
                        )
                        all_images.extend(result.images)
            except Exception as e:
                print(f"⚠️ Sub-batch generation failed: {e}")
                continue

        # Handle output format
        if output_format == "tensor":
            tensor_images = []
            for img in all_images:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                tensor_images.append(img_tensor)
            all_images = torch.stack(tensor_images)

        total_time = time.time() - total_start
        print(f"✅ Total sub-batched generation time: {total_time:.3f}s")

        return {
            "images": all_images,
            "metadata": {
                "alphas": [float(a) for a in alphas.tolist()],
                "prompt1": prompt1,
                "prompt2": prompt2,
            },
            "total_time": total_time,
            "batch_size": total_batch_size,
        }

    @no_grad_method  
    def generate_interpolated_embeddings_at_alphas(
        self,
        prompt1: str,
        prompt2: str,
        alphas: List[float],
        output_format: str = "pil",
        latent_cookie: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate images at specific interpolation alpha values.
        
        This method allows precise control over interpolation points,
        useful for adaptive interpolation algorithms.
        """
        total_start = time.time()
        batch_size = len(alphas)
        print(f"🎯 Kandinsky precise interpolation: '{prompt1}' → '{prompt2}' at {batch_size} alphas")
        print(f"🔢 Alpha values: {alphas}")

        # Set up generation parameters
        if "generator" not in kwargs and "seed" in kwargs:
            seed = kwargs.pop("seed")
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        # Extract image embeddings
        encode_start = time.time()
        
        cache_key1 = f"{latent_cookie}_{prompt1}" if latent_cookie else None
        cache_key2 = f"{latent_cookie}_{prompt2}" if latent_cookie else None
        
        if cache_key1 and cache_key1 in self.image_embedding_cache:
            embedding1 = self.image_embedding_cache[cache_key1]
        else:
            embedding1 = self._extract_image_embeddings(prompt1, **kwargs)
            if embedding1 is None:
                raise ValueError(f"Failed to extract image embeddings for prompt1")
            if cache_key1:
                self.image_embedding_cache[cache_key1] = embedding1.clone()

        if cache_key2 and cache_key2 in self.image_embedding_cache:
            embedding2 = self.image_embedding_cache[cache_key2]
        else:
            embedding2 = self._extract_image_embeddings(prompt2, **kwargs)
            if embedding2 is None:
                raise ValueError(f"Failed to extract image embeddings for prompt2")
            if cache_key2:
                self.image_embedding_cache[cache_key2] = embedding2.clone()

        # Create interpolated embeddings at specified alphas
        interpolated_embeddings = []
        for alpha in alphas:
            interpolated = self.interpolate_embeddings(embedding1, embedding2, alpha)
            interpolated_embeddings.append(interpolated)

        encode_time = time.time() - encode_start
        print(f"📝 Embedding interpolation completed in {encode_time:.3f}s")

        # Calculate sub-batching
        sub_batch_size = self._calculate_sub_batch_size(batch_size, kwargs.get("width", 768), kwargs.get("height", 768))
        
        if sub_batch_size < batch_size:
            # Sub-batch the generation
            all_images = []
            for start_idx in range(0, batch_size, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, batch_size)
                
                sub_batch_embeddings = interpolated_embeddings[start_idx:end_idx]
                batch_image_embeds = torch.cat(sub_batch_embeddings, dim=0)
                
                try:
                    if hasattr(self.pipe, 'decoder') and self.pipe.decoder is not None:
                        result = self.pipe.decoder(
                            image_embeds=batch_image_embeds,
                            height=kwargs.get("height", 768),
                            width=kwargs.get("width", 768),
                            num_inference_steps=kwargs.get("num_inference_steps", 100),
                            guidance_scale=kwargs.get("guidance_scale", 4.0),
                            output_type="pil",
                            **{k: v for k, v in kwargs.items() if k not in ['seed']}
                        )
                        all_images.extend(result.images)
                    else:
                        # Fallback
                        for embedding in sub_batch_embeddings:
                            result = self.pipe(
                                f"Interpolated",
                                height=kwargs.get("height", 768),
                                width=kwargs.get("width", 768),
                                num_inference_steps=kwargs.get("num_inference_steps", 100),
                                guidance_scale=kwargs.get("guidance_scale", 4.0),
                                num_images_per_prompt=1,
                                output_type="pil",
                                **{k: v for k, v in kwargs.items() if k not in ['seed']}
                            )
                            all_images.extend(result.images)
                except Exception as e:
                    print(f"⚠️ Sub-batch generation failed: {e}")
                    continue
            
            images = all_images
        else:
            # Generate all at once
            batch_image_embeds = torch.cat(interpolated_embeddings, dim=0)
            
            try:
                if hasattr(self.pipe, 'decoder') and self.pipe.decoder is not None:
                    result = self.pipe.decoder(
                        image_embeds=batch_image_embeds,
                        height=kwargs.get("height", 768),
                        width=kwargs.get("width", 768),
                        num_inference_steps=kwargs.get("num_inference_steps", 100),
                        guidance_scale=kwargs.get("guidance_scale", 4.0),
                        output_type="pil",
                        **{k: v for k, v in kwargs.items() if k not in ['seed']}
                    )
                    images = result.images
                else:
                    # Fallback
                    images = []
                    for embedding in interpolated_embeddings:
                        result = self.pipe(
                            f"Interpolated",
                            height=kwargs.get("height", 768),
                            width=kwargs.get("width", 768),
                            num_inference_steps=kwargs.get("num_inference_steps", 100),
                            guidance_scale=kwargs.get("guidance_scale", 4.0),
                            num_images_per_prompt=1,
                            output_type="pil",
                            **{k: v for k, v in kwargs.items() if k not in ['seed']}
                        )
                        images.extend(result.images)
            except Exception as e:
                raise ValueError(f"Failed to generate images: {e}")

        # Handle output format
        if output_format == "tensor":
            tensor_images = []
            for img in images:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                tensor_images.append(img_tensor)
            images = torch.stack(tensor_images)

        total_time = time.time() - total_start
        print(f"✅ Total alpha-specific generation time: {total_time:.3f}s")

        return {
            "images": images,
            "alphas": alphas,
            "metadata": {
                "prompt1": prompt1,
                "prompt2": prompt2,
            },
            "total_time": total_time,
            "batch_size": batch_size,
        }

    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, "pipe"):
            del self.pipe
        if hasattr(self, "image_embedding_cache"):
            self.image_embedding_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
