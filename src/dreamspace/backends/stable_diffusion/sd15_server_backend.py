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
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        
        print(f"🔮 Loading Stable Diffusion 1.5 from {self.model_id} on {self.device}...")
        
        # Prepare loading arguments
        pipeline_kwargs = {
            "torch_dtype": torch.float16,
        }
        
        # Optionally disable safety checker (fixes false positives)
        if self.disable_safety_checker:
            pipeline_kwargs["safety_checker"] = None
            pipeline_kwargs["requires_safety_checker"] = False
            print("  🚫 NSFW safety checker disabled (fixes false positives)")
        
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
        
        # Load image-to-image pipeline
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            **pipeline_kwargs
        )
        
        # Move pipeline to specified device
        self.img2img_pipe = self.img2img_pipe.to(self.device)
        print(f"  📍 Image2Image pipeline moved to {self.device}")
        
        if self.device == "cuda" or self.device == "cuda:0":
            self.img2img_pipe.enable_model_cpu_offload()
            print(f"  💾 CPU offload enabled for {self.device}")
        else:
            print(f"  🎯 Multi-GPU mode: keeping pipeline on {self.device} (no CPU offload)")
        
        # Enable memory optimizations
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
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

        # Step 3.5: Create additional latents by adding noise (wiggle)
        batch_size = kwargs.get('num_images_per_prompt', 1)
        latents_batch = [latents]
        if batch_size > 1:
            for _ in range(batch_size - 1):
                noise = torch.randn_like(latents) * 0.05  # Adjust noise magnitude as needed
                latents_batch.append(latents + noise)

        # Concatenate all latents into a single batch
        latents_batch = torch.cat(latents_batch, dim=0)

        # Step 4: Decode the batch of latents to images
        images = self.pipe.vae.decode(latents_batch / 0.18215).sample

        return {
            'image': images,  # Return the full list, not just the first image
            'latents': latents,
            'embeddings': self._extract_text_embeddings(prompt)
        }
    
    def img2img(self, image: Image.Image, prompt: str, strength: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Transform an existing image using a text prompt."""
        # Ensure we're working with the correct CUDA context
        # Handle different device formats (string like "cuda:0" or int like 0)
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            device_index = int(self.device.split(":")[1])
            torch.cuda.set_device(device_index)
        elif isinstance(self.device, int):
            torch.cuda.set_device(self.device)
        elif self.device == "cuda":
            torch.cuda.set_device(0)  # Default to GPU 0
        
        # Set default generator for reproducibility on the correct device
        if 'generator' not in kwargs and 'seed' in kwargs:
            seed = kwargs.pop('seed')
            # Create generator on the same device as the pipeline
            device = self.device if hasattr(self, 'device') else 'cuda'
            kwargs['generator'] = torch.Generator(device=device).manual_seed(seed)
        
        # Check if batch generation is requested
        num_images = kwargs.get('num_images_per_prompt', 1)
        print(f"🎯 SD15 img2img backend on {self.device}: generating {num_images} variations in parallel with strength={strength}")
        
        try:
            # Ensure image is in correct format and clean any CUDA references
            if hasattr(image, '_tensor'):
                # If image has tensor attributes, recreate it cleanly
                from io import BytesIO
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                image = Image.open(buffer)
            
            # Clear any cached GPU memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = self.img2img_pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                return_dict=True,
                **kwargs
            )
            
            # Handle single or multiple images correctly
            images = result.images
            print(f"🖼️ Generated {len(images)} img2img variations in parallel on {self.device}")
            
            # Always return the full list of images for batch processing
            return {
                'image': images,  # Return the full list, not just the first image
                'latents': getattr(result, 'latents', None),
                'embeddings': self._extract_text_embeddings(prompt)
            }
            
        except RuntimeError as e:
            if "CUDA error" in str(e) or "device" in str(e).lower():
                print(f"⚠️ CUDA device error in img2img on {self.device}: {e}")
                # Try to recover by clearing cache and retrying once
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Recreate the image cleanly to remove any GPU references
                from io import BytesIO
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                clean_image = Image.open(buffer)
                
                # Retry with clean image
                result = self.img2img_pipe(
                    prompt=prompt,
                    image=clean_image,
                    strength=strength,
                    return_dict=True,
                    **kwargs
                )
                
                # Handle single or multiple images correctly
                images = result.images
                print(f"🖼️ Retry successful: Generated {len(images)} img2img variations on {self.device}")
                
                return {
                    'image': images,  # Return the full list, not just the first image
                    'latents': getattr(result, 'latents', None),
                    'embeddings': self._extract_text_embeddings(prompt)
                }
            else:
                raise
    
    def generate_batch_with_latent_wiggle(self, prompt: str, batch_size: int, noise_magnitude: float, **kwargs) -> Dict[str, Any]:
        """Generate a batch of images with latent wiggle variations."""
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

        # Step 4: Create additional latents by adding noise (wiggle)
        latents_batch = [latents]
        if batch_size > 1:
            for _ in range(batch_size - 1):
                noise = torch.randn_like(latents) * noise_magnitude
                latents_batch.append(latents + noise)

        # Concatenate all latents into a single batch
        latents_batch = torch.cat(latents_batch, dim=0)

        # Step 5: Decode the batch of latents to images
        images = self.pipe.vae.decode(latents_batch / 0.18215).sample

        return {
            'images': images,
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
        if hasattr(self, 'img2img_pipe'):
            del self.img2img_pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
