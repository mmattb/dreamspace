"""Stable Diffusion 2.1 server backend."""

import time

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...core.utils import no_grad_method
from ...config.settings import Config, ModelConfig


class StableDiffusion21ServerBackend(ImgGenBackend):
    """Stable Diffusion 2.1 server backend using AutoPipeline.
    
    Optimized for server deployment with CPU offloading and memory efficiency.
    Uses the AutoPipeline approach for simplified model loading.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 device: Optional[str] = None,
                 disable_safety_checker: bool = False):
        """Initialize the SD 2.1 server backend.
        
        Args:
            config: Configuration instance
            device: Device to run on (overrides config)
            disable_safety_checker: If True, disables NSFW safety checker (fixes false positives)
        """
        self.config = config or Config()
        self.device = device or "cuda"
        self.model_id = "stabilityai/stable-diffusion-2-1"
        self.disable_safety_checker = disable_safety_checker
        
        # Latent cache for shared initial latents across batches
        self.latent_cache = {}  # cookie -> latent tensor
        
        self._load_pipelines()
    
    def _calculate_sub_batch_size(self, total_batch_size: int, width: int, height: int) -> int:
        """Calculate optimal sub-batch size based on memory heuristics.
        
        Args:
            total_batch_size: Total number of images to generate
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Optimal sub-batch size (1 to total_batch_size)
        """
        # Calculate megapixels per image
        megapixels = (width * height) / 1_000_000
        
        # Rough heuristic: Base memory budget in GB (conservative estimate)
        # SD 2.1 uses roughly 6-10GB base (much larger than SD 1.5), leave room for operations
        available_memory_gb = 6.0  # More conservative for SD 2.1's larger model
        
        # Estimate memory usage per image during generation
        # SD 2.1 latents are 1/8 resolution, 4 channels, float16
        latent_megapixels = megapixels / 64  # 8x8 downsampling
        
        # Memory estimates for SD 2.1 (higher than SD 1.5):
        # - Latents: latent_megapixels * 4 channels * 2 bytes (float16) 
        # - UNet activations: ~5x latent size (larger UNet, more attention heads)
        # - VAE decode: ~6x latent size (RGB output)
        # - Overhead: 3x for gradient computation, larger attention maps, OpenCLIP overhead
        
        memory_per_image_mb = latent_megapixels * 4 * 2 * (5 + 6 + 3)  # More conservative for SD 2.1
        memory_per_image_gb = memory_per_image_mb / 1000
        
        # Calculate how many images we can fit in memory
        max_parallel_images = max(1, int(available_memory_gb / memory_per_image_gb))
        
        # Don't exceed the total batch size
        sub_batch_size = min(max_parallel_images, total_batch_size)
        
        # Apply more conservative practical limits for SD 2.1
        sub_batch_size = max(1, min(sub_batch_size, 8))  # Never more than 8 per sub-batch for SD 2.1
        
        print(f"📊 SD 2.1 Memory heuristic for {width}x{height} ({megapixels:.1f}MP):")
        print(f"   Estimated {memory_per_image_gb:.2f}GB per image (SD 2.1 is memory-intensive)")
        print(f"   Sub-batch size: {sub_batch_size} (from total {total_batch_size})")
        
        return sub_batch_size
    
    def _load_pipelines(self):
        """Load the diffusion pipelines using AutoPipeline."""
        from diffusers import AutoPipelineForText2Image
        
        print(f"🔮 Loading Stable Diffusion 2.1 from {self.model_id} on {self.device}...")
        
        # Prepare loading arguments
        pipeline_kwargs = {
            "torch_dtype": torch.float16,
        }
        
        # Optionally disable safety checker (fixes false positives)
        if self.disable_safety_checker:
            pipeline_kwargs["safety_checker"] = None
            pipeline_kwargs["requires_safety_checker"] = False
        
        # Load text-to-image pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            **pipeline_kwargs
        )
        
        # Move pipeline to specified device
        self.pipe = self.pipe.to(self.device)
        print(f"  📍 Text2Image pipeline moved to {self.device}")
        
        # Only enable CPU offload for single GPU setups
        # For multi-GPU, keep models on their assigned GPUs
        if self.device == "cuda" or self.device == "cuda:0":
            self.pipe.enable_model_cpu_offload()
            print(f"  💾 CPU offload enabled for {self.device}")
        else:
            print(f"  🎯 Multi-GPU mode: keeping pipeline on {self.device} (no CPU offload)")
        
        # Enable memory optimizations - SD 2.1 benefits from aggressive memory management
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory optimization enabled")
        except Exception:
            print("⚠️ XFormers not available, using default attention")
        
        # Enable additional memory optimizations for SD 2.1's larger model
        try:
            self.pipe.enable_attention_slicing(1)  # Slice attention computation for memory efficiency
            print("✅ Attention slicing enabled for SD 2.1")
        except Exception:
            print("⚠️ Attention slicing not available")
        
        try:
            if hasattr(self.pipe.vae, 'enable_slicing'):
                self.pipe.vae.enable_slicing()
                print("✅ VAE slicing enabled for SD 2.1")
        except Exception:
            print("⚠️ VAE slicing not available")
        
        print(f"✅ Stable Diffusion 2.1 loaded successfully on {self.device}!")
    
    @no_grad_method
    def generate(self, prompt: str, batch_size: int, noise_magnitude: float, bifurcation_step: int, output_format: str = "pil", latent_cookie: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate a batch of images with bifurcated latent wiggle variations.
        
        This approach runs shared denoising until bifurcation_step, then adds noise
        and continues denoising in parallel to ensure all variations stay on the manifold.
        
        Args:
            prompt: Text prompt for generation
            batch_size: Number of variations to generate
            noise_magnitude: Magnitude of noise to add at bifurcation point
            bifurcation_step: Number of steps from the end when to add noise and bifurcate
                            (e.g., 5 means bifurcate 5 steps before completion)
        """
        total_start = time.time()
        print(f"🔀 Starting bifurcated wiggle: {batch_size} variations, noise={noise_magnitude}, bifurcation={bifurcation_step}")
        
        # Setup phase
        setup_start = time.time()
        
        # Force all models to eval mode
        self.pipe.unet.eval()
        self.pipe.vae.eval() 
        self.pipe.text_encoder.eval()
        
        if hasattr(self.pipe.vae, 'enable_slicing'):
            self.pipe.vae.enable_slicing()

        # Set default generator for reproducibility on the correct device
        if 'generator' not in kwargs and 'seed' in kwargs:
            seed = kwargs.pop('seed')
            # Create generator on the same device as the pipeline
            device = self.device if hasattr(self, 'device') else 'cuda'
            kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)

        generator = kwargs['generator']
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        height = kwargs.get('height', 768)  # SD 2.1 default resolution
        width = kwargs.get('width', 768)   # SD 2.1 default resolution
        num_inference_steps = kwargs.get('num_inference_steps', 50)

        # Step 1: Set scheduler timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.pipe.device)
        
        setup_time = time.time() - setup_start
        print(f"⚙️ Setup completed in {setup_time:.3f}s")

        # Step 2: Encode the prompt
        encode_start = time.time()
        
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        # Concatenate negative and positive embeddings for classifier-free guidance
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        encode_time = time.time() - encode_start
        print(f"📝 Prompt encoding completed in {encode_time:.3f}s")

        # Step 3: Prepare initial noise (with optional latent caching)
        noise_start = time.time()
        
        # Check if we should use cached latent
        if latent_cookie is not None:
            latent_key = (latent_cookie, height, width)  # Include dimensions in key
            if latent_key in self.latent_cache:
                print(f"🍪 Using cached latent for cookie {latent_cookie}")
                latents = self.latent_cache[latent_key].clone()
            else:
                print(f"🍪 Creating new latent for cookie {latent_cookie}")
                latents = self.pipe.prepare_latents(
                    batch_size=1,
                    num_channels_latents=self.pipe.unet.config.in_channels,
                    height=height,
                    width=width,
                    dtype=self.pipe.unet.dtype,
                    device=self.pipe.device,
                    generator=generator,
                )
                # Cache the latent for future use
                self.latent_cache[latent_key] = latents.clone()
        else:
            # No caching - generate fresh latent each time
            latents = self.pipe.prepare_latents(
                batch_size=1,
                num_channels_latents=self.pipe.unet.config.in_channels,
                height=height,
                width=width,
                dtype=self.pipe.unet.dtype,
                device=self.pipe.device,
                generator=generator,
            )
        
        noise_time = time.time() - noise_start
        print(f"🎲 Initial noise preparation completed in {noise_time:.3f}s")

        # Step 4: Run shared denoising until bifurcation point
        denoise_start = time.time()
        
        timesteps = self.pipe.scheduler.timesteps
        bifurcation_index = len(timesteps) - bifurcation_step
        
        print(f"🔀 Running shared denoising for {bifurcation_index} steps, then bifurcating for final {bifurcation_step} steps")
        
        # Shared denoising phase
        for i, t in enumerate(timesteps[:bifurcation_index]):
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            # Use torch.no_grad() to prevent gradient accumulation
            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_input, t, encoder_hidden_states=prompt_embeds
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Clean up intermediate tensors to free GPU memory
            del latent_input, noise_pred, noise_pred_uncond, noise_pred_text

        denoise_time = time.time() - denoise_start
        print(f"🧠 Shared denoising ({bifurcation_index} steps) completed in {denoise_time:.3f}s")

        # Step 5: Bifurcate - create variations by adding noise
        bifurcate_start = time.time()
        
        latents_batch = [latents]
        if batch_size > 1:
            for _ in range(batch_size - 1):
                noise = torch.randn_like(latents) * noise_magnitude
                latents_batch.append(latents + noise)

        # Concatenate all latents into a single batch for parallel processing
        latents_batch = torch.cat(latents_batch, dim=0)
        
        bifurcate_time = time.time() - bifurcate_start
        print(f"🌀 Bifurcation (noise addition) completed in {bifurcate_time:.3f}s")
        
        # Step 6: Continue denoising all variations in parallel for remaining steps
        parallel_denoise_start = time.time()
        
        remaining_timesteps = timesteps[bifurcation_index:]
        
        # Expand prompt embeddings to match batch size for parallel processing
        batch_prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        
        for i, t in enumerate(remaining_timesteps):
            # Process entire batch at once
            latent_input = torch.cat([latents_batch] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            # Use torch.no_grad() to prevent gradient accumulation
            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_input, t, encoder_hidden_states=batch_prompt_embeds
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_batch = self.pipe.scheduler.step(noise_pred, t, latents_batch).prev_sample
            
            # Clean up intermediate tensors to free GPU memory
            del latent_input, noise_pred, noise_pred_uncond, noise_pred_text

        parallel_denoise_time = time.time() - parallel_denoise_start
        print(f"🔄 Parallel denoising ({bifurcation_step} steps × {batch_size} variations) completed in {parallel_denoise_time:.3f}s")

        # Step 7: Decode the batch of latents to images
        decode_start = time.time()
        
        # Use direct VAE decode with explicit no_grad
        vae_start = time.time()
        with torch.no_grad():
            # Ensure no computation graph is built
            latents_batch.requires_grad_(False)
            images = self.pipe.vae.decode(latents_batch / 0.18215).sample
        vae_time = time.time() - vae_start
        print(f"🔮 VAE decode completed in {vae_time:.3f}s")
        
        images = (images / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]

        # Handle output format processing
        if output_format == "tensor":
            # For tensor format: keep as PyTorch tensor, just normalize to [0,1]
            normalize_start = time.time()
            normalize_time = time.time() - normalize_start
            print(f"🚀 Keeping tensor format for ultra-fast serialization")
            print(f"⚡ Tensor normalization completed in {normalize_time:.6f}s")
            
            decode_time = time.time() - decode_start
            print(f"🎨 Total VAE decoding (tensor format) completed in {decode_time:.3f}s")
            print(f"   🔮 VAE decode: {vae_time:.3f}s ({vae_time/decode_time*100:.1f}%)")
            print(f"   ⚡ Tensor normalization: {normalize_time:.3f}s ({normalize_time/decode_time*100:.1f}%)")
            print(f"   📊 Other operations: {decode_time-vae_time-normalize_time:.3f}s ({(decode_time-vae_time-normalize_time)/decode_time*100:.1f}%)")
        else:
            # For other formats: convert to numpy then PIL
            tensor_convert_start = time.time()
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # BCHW -> BHWC and to numpy
            tensor_convert_time = time.time() - tensor_convert_start
            print(f"🔄 Tensor→numpy conversion completed in {tensor_convert_time:.3f}s")
            
            # Convert to PIL Images for other formats
            pil_start = time.time()
            pil_images = []
            for i in range(images.shape[0]):
                image_array = (images[i] * 255).astype('uint8')  # Convert to 0-255 range
                pil_image = Image.fromarray(image_array)
                pil_images.append(pil_image)
            pil_time = time.time() - pil_start
            print(f"🖼️ Numpy→PIL conversion completed in {pil_time:.3f}s")
            
            decode_time = time.time() - decode_start
            print(f"🎨 Total VAE decoding + PIL conversion completed in {decode_time:.3f}s")
            print(f"   🔮 VAE decode: {vae_time:.3f}s ({vae_time/decode_time*100:.1f}%)")
            print(f"   🔄 Tensor→numpy: {tensor_convert_time:.3f}s ({tensor_convert_time/decode_time*100:.1f}%)")
            print(f"   🖼️ Numpy→PIL: {pil_time:.3f}s ({pil_time/decode_time*100:.1f}%)")
        
        total_time = time.time() - total_start
        print(f"✅ Total bifurcated wiggle generation time: {total_time:.3f}s")
        print(f"📊 Timing breakdown:")
        print(f"   ⚙️ Setup: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
        print(f"   📝 Encoding: {encode_time:.3f}s ({encode_time/total_time*100:.1f}%)")
        print(f"   🎲 Noise prep: {noise_time:.3f}s ({noise_time/total_time*100:.1f}%)")
        print(f"   🧠 Shared denoise: {denoise_time:.3f}s ({denoise_time/total_time*100:.1f}%)")
        print(f"   🌀 Bifurcation: {bifurcate_time:.3f}s ({bifurcate_time/total_time*100:.1f}%)")
        print(f"   🔄 Parallel denoise: {parallel_denoise_time:.3f}s ({parallel_denoise_time/decode_time*100:.1f}%)")
        print(f"   🎨 Decoding: {decode_time:.3f}s ({decode_time/total_time*100:.1f}%)")

        # Return based on requested output format
        if output_format == "tensor":
            # Return raw PyTorch tensors for ultra-fast local processing
            print(f"🚀 Returning raw tensors (PyTorch format) for high-speed local processing")
            return {
                'images': images,  # PyTorch tensor [0,1] range, shape (batch, channels, height, width)
                'format': 'torch_tensor',
                'shape': tuple(images.shape),
                'device': str(images.device),
                'dtype': str(images.dtype),
                'latents': latents_batch,
                'embeddings': self._extract_text_embeddings(prompt)
            }
        elif output_format == "pil":
            # Original PIL format (default for backwards compatibility)
            return {
                'images': pil_images,  # PIL Images
                'format': 'pil',
                'latents': latents_batch,
                'embeddings': self._extract_text_embeddings(prompt)
            }
        else:
            # Default to PIL for now, could add other formats later
            return {
                'images': pil_images,  # PIL Images
                'format': 'pil', 
                'latents': latents_batch,
                'embeddings': self._extract_text_embeddings(prompt)
            }
    
    @no_grad_method
    def generate_interpolated_embeddings(self, prompt1: str, prompt2: str, batch_size: int, output_format: str = "pil", latent_cookie: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate a batch of images using interpolated embeddings between two prompts.

        Args:
            prompt1: The starting text prompt.
            prompt2: The ending text prompt.
            batch_size: Number of interpolation steps (including start and end).
            output_format: Format of the output images (e.g., "pil", "tensor", "jpeg", "png").

        Returns:
            A dictionary containing the generated images and metadata.
        """
        total_start = time.time()
        print(f"🌈 Starting interpolated embedding generation: '{prompt1}' → '{prompt2}' with {batch_size} steps")
        
        # Setup phase
        setup_start = time.time()
        
        # Force all models to eval mode
        self.pipe.unet.eval()
        self.pipe.vae.eval() 
        self.pipe.text_encoder.eval()
        
        if hasattr(self.pipe.vae, 'enable_slicing'):
            self.pipe.vae.enable_slicing()

        # Set default generator for reproducibility on the correct device
        if 'generator' not in kwargs and 'seed' in kwargs:
            seed = kwargs.pop('seed')
            device = self.device if hasattr(self, 'device') else 'cuda'
            kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)

        generator = kwargs.get('generator')
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        height = kwargs.get('height', 768)  # SD 2.1 default resolution
        width = kwargs.get('width', 768)   # SD 2.1 default resolution
        num_inference_steps = kwargs.get('num_inference_steps', 50)

        # Set scheduler timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.pipe.device)
        
        setup_time = time.time() - setup_start
        print(f"⚙️ Setup completed in {setup_time:.3f}s")

        # Encode both prompts into embeddings properly
        encode_start = time.time()
        
        # Extract embeddings for both prompts
        embedding1 = self._extract_text_embeddings(prompt1)
        embedding2 = self._extract_text_embeddings(prompt2)
        
        if embedding1 is None or embedding2 is None:
            raise ValueError("Failed to extract text embeddings from prompts")
        
        # Create interpolated embeddings
        interpolated_embeddings = []
        alphas = torch.linspace(0, 1, steps=batch_size)
        print(f"🔍 Creating {batch_size} interpolation steps from '{prompt1}' to '{prompt2}'")
        
        for i, alpha in enumerate(alphas):
            interpolated = embedding1 * (1 - alpha) + embedding2 * alpha
            interpolated_embeddings.append(interpolated)
        
        # Stack all interpolated embeddings into a batch
        batch_prompt_embeds = torch.cat(interpolated_embeddings, dim=0)
        print(f"� Batched {len(interpolated_embeddings)} interpolated embeddings, shape: {batch_prompt_embeds.shape}")
        
        # Create negative embeddings for classifier-free guidance (use empty prompt)
        negative_embedding = self._extract_text_embeddings("")
        batch_negative_embeds = negative_embedding.repeat(batch_size, 1, 1)
        
        # Concatenate negative and positive embeddings for classifier-free guidance
        batch_combined_embeds = torch.cat([batch_negative_embeds, batch_prompt_embeds])
        
        encode_time = time.time() - encode_start
        print(f"📝 Embedding interpolation completed in {encode_time:.3f}s")

        # Prepare initial noise for the batch - use same latent for all interpolation steps (with optional caching)
        noise_start = time.time()
        
        # Check if we should use cached latent
        if latent_cookie is not None:
            latent_key = (latent_cookie, height, width)  # Include dimensions in key
            if latent_key in self.latent_cache:
                print(f"🍪 Using cached latent for cookie {latent_cookie}")
                single_latent = self.latent_cache[latent_key].clone()
            else:
                print(f"🍪 Creating new latent for cookie {latent_cookie}")
                single_latent = self.pipe.prepare_latents(
                    batch_size=1,
                    num_channels_latents=self.pipe.unet.config.in_channels,
                    height=height,
                    width=width,
                    dtype=self.pipe.unet.dtype,
                    device=self.pipe.device,
                    generator=generator,
                )
                # Cache the latent for future use
                self.latent_cache[latent_key] = single_latent.clone()
        else:
            # No caching - generate fresh latent each time
            single_latent = self.pipe.prepare_latents(
                batch_size=1,
                num_channels_latents=self.pipe.unet.config.in_channels,
                height=height,
                width=width,
                dtype=self.pipe.unet.dtype,
                device=self.pipe.device,
                generator=generator,
            )
        
        # Calculate optimal sub-batch size for memory management
        sub_batch_size = self._calculate_sub_batch_size(batch_size, width, height)
        
        # If sub-batching is needed
        if sub_batch_size < batch_size:
            print(f"🔄 Using sub-batching: {batch_size} images in chunks of {sub_batch_size}")
            return self._generate_interpolated_embeddings_with_sub_batching(
                prompt1, prompt2, batch_size, sub_batch_size, output_format, latent_cookie, **kwargs
            )
        
        # Original single-batch logic for when no sub-batching is needed
        print(f"🚀 Processing full batch of {batch_size} images (no sub-batching needed)")
        
        # Repeat the same latent for all interpolation steps
        latents_batch = single_latent.repeat(batch_size, 1, 1, 1)
        
        noise_time = time.time() - noise_start
        print(f"🎲 Shared latent preparation completed in {noise_time:.3f}s (same initial noise for all {batch_size} interpolation steps)")

        # Run denoising for all interpolated embeddings in parallel
        denoise_start = time.time()
        
        timesteps = self.pipe.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            # Process entire batch at once
            latent_input = torch.cat([latents_batch] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_input, t, encoder_hidden_states=batch_combined_embeds
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_batch = self.pipe.scheduler.step(noise_pred, t, latents_batch).prev_sample
            
            # Clean up intermediate tensors to free GPU memory
            del latent_input, noise_pred, noise_pred_uncond, noise_pred_text

        denoise_time = time.time() - denoise_start
        print(f"🧠 Parallel denoising ({num_inference_steps} steps × {batch_size} interpolations) completed in {denoise_time:.3f}s")

        # Decode the batch of latents to images
        decode_start = time.time()
        
        # Use direct VAE decode with explicit no_grad
        vae_start = time.time()
        with torch.no_grad():
            latents_batch.requires_grad_(False)
            images = self.pipe.vae.decode(latents_batch / 0.18215).sample
        vae_time = time.time() - vae_start
        print(f"🔮 VAE decode completed in {vae_time:.3f}s")
        
        images = (images / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]

        # Handle output format processing (same as generate())
        if output_format == "tensor":
            # For tensor format: keep as PyTorch tensor, just normalize to [0,1]
            normalize_start = time.time()
            normalize_time = time.time() - normalize_start
            print(f"🚀 Keeping tensor format for ultra-fast serialization")
            print(f"⚡ Tensor normalization completed in {normalize_time:.6f}s")
            
            decode_time = time.time() - decode_start
            print(f"🎨 Total VAE decoding (tensor format) completed in {decode_time:.3f}s")
        else:
            # For other formats: convert to numpy then PIL
            tensor_convert_start = time.time()
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # BCHW -> BHWC and to numpy
            tensor_convert_time = time.time() - tensor_convert_start
            print(f"🔄 Tensor→numpy conversion completed in {tensor_convert_time:.3f}s")
            
            # Convert to PIL Images for other formats
            pil_start = time.time()
            pil_images = []
            for i in range(images.shape[0]):
                image_array = (images[i] * 255).astype('uint8')  # Convert to 0-255 range
                pil_image = Image.fromarray(image_array)
                pil_images.append(pil_image)
            pil_time = time.time() - pil_start
            print(f"🖼️ Numpy→PIL conversion completed in {pil_time:.3f}s")
            
            decode_time = time.time() - decode_start
            print(f"🎨 Total VAE decoding + PIL conversion completed in {decode_time:.3f}s")
        
        total_time = time.time() - total_start
        print(f"✅ Total interpolated embedding generation time: {total_time:.3f}s")
        print(f"📊 Timing breakdown:")
        print(f"   ⚙️ Setup: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
        print(f"   📝 Encoding: {encode_time:.3f}s ({encode_time/total_time*100:.1f}%)")
        print(f"   🎲 Noise prep: {noise_time:.3f}s ({noise_time/total_time*100:.1f}%)")
        print(f"   🧠 Denoising: {denoise_time:.3f}s ({denoise_time/total_time*100:.1f}%)")
        print(f"   🎨 Decoding: {decode_time:.3f}s ({decode_time/total_time*100:.1f}%)")

        # Return based on requested output format (same as generate())
        if output_format == "tensor":
            return {
                'images': images,  # PyTorch tensor [0,1] range, shape (batch, channels, height, width)
                'format': 'torch_tensor',
                'shape': tuple(images.shape),
                'device': str(images.device),
                'dtype': str(images.dtype),
                'latents': latents_batch,
                'embeddings': batch_prompt_embeds
            }
        elif output_format == "pil":
            return {
                'images': pil_images,  # PIL Images
                'format': 'pil',
                'latents': latents_batch,
                'embeddings': batch_prompt_embeds
            }
        else:
            # Default to PIL for now, could add other formats later
            return {
                'images': pil_images,  # PIL Images
                'format': 'pil', 
                'latents': latents_batch,
                'embeddings': batch_prompt_embeds
            }
    
    def _generate_interpolated_embeddings_with_sub_batching(self, prompt1: str, prompt2: str, 
                                                           total_batch_size: int, sub_batch_size: int,
                                                           output_format: str, latent_cookie: Optional[int],
                                                           **kwargs) -> Dict[str, Any]:
        """Generate interpolated embeddings using sub-batching for memory efficiency."""
        
        total_start = time.time()
        print(f"🔄 Sub-batching {total_batch_size} images into chunks of {sub_batch_size}")
        
        # Setup phase (same as original)
        setup_start = time.time()
        self.pipe.unet.eval()
        self.pipe.vae.eval() 
        self.pipe.text_encoder.eval()
        
        if hasattr(self.pipe.vae, 'enable_slicing'):
            self.pipe.vae.enable_slicing()

        # Set default generator for reproducibility on the correct device
        if 'generator' not in kwargs and 'seed' in kwargs:
            seed = kwargs.pop('seed')
            device = self.device if hasattr(self, 'device') else 'cuda'
            kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)

        generator = kwargs.get('generator')
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        height = kwargs.get('height', 768)  # SD 2.1 default resolution  
        width = kwargs.get('width', 768)   # SD 2.1 default resolution
        num_inference_steps = kwargs.get('num_inference_steps', 50)

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.pipe.device)
        setup_time = time.time() - setup_start
        
        # Extract embeddings once (shared across all sub-batches)
        encode_start = time.time()
        embedding1 = self._extract_text_embeddings(prompt1)
        embedding2 = self._extract_text_embeddings(prompt2)
        
        if embedding1 is None or embedding2 is None:
            raise ValueError("Failed to extract text embeddings from prompts")
        
        # Create ALL interpolated embeddings first
        interpolated_embeddings = []
        alphas = torch.linspace(0, 1, steps=total_batch_size)
        print(f"🔍 Creating {total_batch_size} interpolation steps from '{prompt1}' to '{prompt2}'")
        
        for i, alpha in enumerate(alphas):
            interpolated = embedding1 * (1 - alpha) + embedding2 * alpha
            interpolated_embeddings.append(interpolated)
        
        encode_time = time.time() - encode_start
        print(f"📝 Embedding interpolation completed in {encode_time:.3f}s")

        # Get shared latent once
        noise_start = time.time()
        if latent_cookie is not None:
            latent_key = (latent_cookie, height, width)
            if latent_key in self.latent_cache:
                print(f"🍪 Using cached latent for cookie {latent_cookie}")
                single_latent = self.latent_cache[latent_key].clone()
            else:
                print(f"🍪 Creating new latent for cookie {latent_cookie}")
                single_latent = self.pipe.prepare_latents(
                    batch_size=1,
                    num_channels_latents=self.pipe.unet.config.in_channels,
                    height=height,
                    width=width,
                    dtype=self.pipe.unet.dtype,
                    device=self.pipe.device,
                    generator=generator,
                )
                self.latent_cache[latent_key] = single_latent.clone()
        else:
            single_latent = self.pipe.prepare_latents(
                batch_size=1,
                num_channels_latents=self.pipe.unet.config.in_channels,
                height=height,
                width=width,
                dtype=self.pipe.unet.dtype,
                device=self.pipe.device,
                generator=generator,
            )
        
        noise_time = time.time() - noise_start
        print(f"🎲 Shared latent preparation completed in {noise_time:.3f}s")

        # Process sub-batches
        all_final_latents = []
        denoise_start = time.time()
        
        for sub_batch_idx in range(0, total_batch_size, sub_batch_size):
            end_idx = min(sub_batch_idx + sub_batch_size, total_batch_size)
            current_sub_batch_size = end_idx - sub_batch_idx
            
            print(f"🔄 Processing sub-batch {sub_batch_idx//sub_batch_size + 1}/{(total_batch_size + sub_batch_size - 1)//sub_batch_size}: images {sub_batch_idx}-{end_idx-1}")
            
            # Get embeddings for this sub-batch
            sub_batch_embeddings = interpolated_embeddings[sub_batch_idx:end_idx]
            batch_prompt_embeds = torch.cat(sub_batch_embeddings, dim=0)
            
            # Create negative embeddings for this sub-batch
            negative_embedding = self._extract_text_embeddings("")
            batch_negative_embeds = negative_embedding.repeat(current_sub_batch_size, 1, 1)
            batch_combined_embeds = torch.cat([batch_negative_embeds, batch_prompt_embeds], dim=0)
            
            # Create latents for this sub-batch (same initial latent repeated)
            sub_batch_latents = single_latent.repeat(current_sub_batch_size, 1, 1, 1)
            
            # Run denoising for this sub-batch
            timesteps = self.pipe.scheduler.timesteps
            
            for i, t in enumerate(timesteps):
                latent_input = torch.cat([sub_batch_latents] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

                with torch.no_grad():
                    noise_pred = self.pipe.unet(
                        latent_input, t, encoder_hidden_states=batch_combined_embeds
                    ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                sub_batch_latents = self.pipe.scheduler.step(noise_pred, t, sub_batch_latents).prev_sample
                
                # Clean up intermediate tensors
                del latent_input, noise_pred, noise_pred_uncond, noise_pred_text

            # Store the final latents for this sub-batch
            all_final_latents.append(sub_batch_latents.cpu())  # Move to CPU to free GPU memory
            
            # Clean up sub-batch tensors
            del sub_batch_latents, batch_combined_embeds, batch_prompt_embeds, batch_negative_embeds
            torch.cuda.empty_cache()  # Free GPU memory between sub-batches
        
        # Combine all sub-batch results
        final_latents_batch = torch.cat(all_final_latents, dim=0).to(self.pipe.device)
        
        denoise_time = time.time() - denoise_start
        print(f"🧠 Sub-batched denoising ({num_inference_steps} steps × {total_batch_size} interpolations) completed in {denoise_time:.3f}s")

        # Decode all images at once (or in sub-batches if needed)
        decode_start = time.time()
        
        # For decoding, we can also sub-batch if needed - SD 2.1 VAE is more memory intensive
        decode_sub_batch_size = min(sub_batch_size * 1, 4)  # SD 2.1 VAE decode needs smaller batches
        all_decoded_images = []
        
        for decode_idx in range(0, total_batch_size, decode_sub_batch_size):
            decode_end_idx = min(decode_idx + decode_sub_batch_size, total_batch_size)
            
            decode_latents = final_latents_batch[decode_idx:decode_end_idx]
            
            with torch.no_grad():
                decode_latents.requires_grad_(False)
                decoded_images = self.pipe.vae.decode(decode_latents / 0.18215).sample
            
            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
            all_decoded_images.append(decoded_images.cpu())
            
            del decode_latents, decoded_images
            torch.cuda.empty_cache()
        
        # Combine all decoded images
        final_images = torch.cat(all_decoded_images, dim=0)
        
        decode_time = time.time() - decode_start
        print(f"🎨 Sub-batched VAE decoding completed in {decode_time:.3f}s")

        # Handle output format processing
        if output_format == "tensor":
            images = final_images.to(self.pipe.device)  # Move back to GPU for output
            print(f"🚀 Returning tensor format for ultra-fast serialization")
        else:
            # Convert to PIL Images
            images = final_images.permute(0, 2, 3, 1).float().numpy()
            pil_images = []
            for i in range(images.shape[0]):
                image_array = (images[i] * 255).astype('uint8')
                pil_image = Image.fromarray(image_array)
                pil_images.append(pil_image)
            images = pil_images

        total_time = time.time() - total_start
        print(f"✅ Total sub-batched interpolated embedding generation time: {total_time:.3f}s")
        print(f"📊 Sub-batching timing breakdown:")
        print(f"   ⚙️ Setup: {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
        print(f"   📝 Encoding: {encode_time:.3f}s ({encode_time/total_time*100:.1f}%)")
        print(f"   🎲 Noise prep: {noise_time:.3f}s ({noise_time/total_time*100:.1f}%)")
        print(f"   🧠 Denoising: {denoise_time:.3f}s ({denoise_time/total_time*100:.1f}%)")
        print(f"   🎨 Decoding: {decode_time:.3f}s ({decode_time/total_time*100:.1f}%)")

        # Return same format as original method
        if output_format == "tensor":
            return {
                'images': images,
                'format': 'torch_tensor',
                'shape': tuple(images.shape),
                'device': str(images.device),
                'dtype': str(images.dtype),
                'latents': final_latents_batch,
                'embeddings': torch.cat(interpolated_embeddings, dim=0)  # All interpolated embeddings
            }
        elif output_format == "pil":
            return {
                'images': images,
                'format': 'pil',
                'latents': final_latents_batch,
                'embeddings': torch.cat(interpolated_embeddings, dim=0)
            }
        else:
            return {
                'images': images,
                'format': 'pil', 
                'latents': final_latents_batch,
                'embeddings': torch.cat(interpolated_embeddings, dim=0)
            }
    
    def interpolate_embeddings(self, embedding1: Any, embedding2: Any, alpha: float) -> Any:
        """Interpolate between two text embeddings."""
        if embedding1 is None or embedding2 is None:
            return None
        return self._slerp(embedding1, embedding2, alpha)
    
    def _extract_text_embeddings(self, prompt: str) -> torch.Tensor:
        """Extract text embeddings from prompt."""
        try:
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                text_embeddings = self.pipe.text_encoder(
                    text_inputs.input_ids.to(self.pipe.device)
                )[0]
            
            return text_embeddings
        except Exception:
            return None
    
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'pipe'):
            del self.pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
