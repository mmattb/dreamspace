"""Stable Diffusion 1.5 server backend."""

import torch
from typing import Dict, Any, Optional
from PIL import Image

from ...core.base import ImgGenBackend
from ...config.settings import Config, ModelConfig


class StableDiffusion15ServerBackend(ImgGenBackend):
    """Stable Diffusion 1.5 server backend using AutoPipeline.
    
    Optimized for server deployment with CPU offloading and memory efficiency.
    Uses the AutoPipeline approach for simplified model loading.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 device: Optional[str] = None,
                 disable_safety_checker: bool = False):
        """Initialize the SD 1.5 server backend.
        
        Args:
            config: Configuration instance
            device: Device to run on (overrides config)
            disable_safety_checker: If True, disables NSFW safety checker (fixes false positives)
        """
        self.config = config or Config()
        self.device = device or "cuda"
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.disable_safety_checker = disable_safety_checker
        self._load_pipelines()
    
    def _load_pipelines(self):
        """Load the diffusion pipelines using AutoPipeline."""
        from diffusers import AutoPipelineForText2Image
        
        print(f"🔮 Loading Stable Diffusion 1.5 from {self.model_id} on {self.device}...")
        
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
        
        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory optimization enabled")
        except Exception:
            print("⚠️ XFormers not available, using default attention")
        
        print(f"✅ Stable Diffusion 1.5 loaded successfully on {self.device}!")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        # Set default generator for reproducibility on the correct device
        if 'generator' not in kwargs and 'seed' in kwargs:
            seed = kwargs.pop('seed')
            # Create generator on the same device as the pipeline
            device = self.device if hasattr(self, 'device') else 'cuda'
            kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)

        generator = kwargs['generator']
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        height = kwargs.get('height', 512)
        width = kwargs.get('width', 512)

        # Step 1: Encode the prompt
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            device=self.pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        # Step 2: Prepare initial noise
        latents = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipe.unet.in_channels,
            height=height,
            width=width,
            dtype=self.pipe.unet.dtype,
            device=self.pipe.device,
            generator=generator,
        )

        # Step 3: Run the diffusion loop manually
        for t in self.pipe.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.pipe.unet(
                latent_input, t, encoder_hidden_states=prompt_embeds
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Step 4: Decode latents to single image
        images = self.pipe.vae.decode(latents / 0.18215).sample

        return {
            'image': images,  # Single image tensor
            'latents': latents,
            'embeddings': self._extract_text_embeddings(prompt)
        }
    
    def generate_batch_with_latent_wiggle(self, prompt: str, batch_size: int, noise_magnitude: float, **kwargs) -> Dict[str, Any]:
        """Generate a batch of images with latent wiggle variations.
        
        This method now defaults to bifurcated wiggle for better manifold adherence.
        """
        # Default bifurcated wiggle parameters
        bifurcation_step = kwargs.pop('bifurcation_step', 3)  # Default to 3 steps from end
        output_format = kwargs.get('output_format', 'pil')
        
        print(f"🔀 Using bifurcated wiggle as default (bifurcation_step={bifurcation_step})")
        
        # Call the bifurcated method with the same parameters
        return self.generate_batch_with_bifurcated_wiggle(
            prompt=prompt,
            batch_size=batch_size,
            noise_magnitude=noise_magnitude,
            bifurcation_step=bifurcation_step,
            output_format=output_format,
            **kwargs
        )
    
    def generate_batch_with_bifurcated_wiggle(self, prompt: str, batch_size: int, noise_magnitude: float, bifurcation_step: int, output_format: str = "pil", **kwargs) -> Dict[str, Any]:
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
        import time
        
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
        height = kwargs.get('height', 512)
        width = kwargs.get('width', 512)
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

        # Step 3: Prepare initial noise
        noise_start = time.time()
        
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
        
        # Handle output format processing
        if output_format == "tensor":
            # For tensor format: keep as PyTorch tensor, just normalize to [0,1]
            images = (images / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]
            print(f"🚀 Keeping tensor format for ultra-fast serialization")
            
            decode_time = time.time() - decode_start
            print(f"🎨 Total VAE decoding (tensor format) completed in {decode_time:.3f}s")
            print(f"   🔮 VAE decode: {vae_time:.3f}s ({vae_time/decode_time*100:.1f}%)")
            print(f"   ⚡ Tensor normalization: {decode_time-vae_time:.3f}s ({(decode_time-vae_time)/decode_time*100:.1f}%)")
        else:
            # For other formats: convert to numpy then PIL
            tensor_convert_start = time.time()
            images = (images / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # BCHW -> BHWC and to numpy
            tensor_convert_time = time.time() - tensor_convert_start
            print(f"🔄 Tensor→numpy conversion completed in {tensor_convert_time:.3f}s")
            
            # Convert to PIL Images for other formats
            pil_start = time.time()
            from PIL import Image
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
