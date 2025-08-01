"""Remote keyboard-controlled image space navigation example.

This example demonstrates smooth navigation through image space using keyboard
controls via a remote API server. It uses img2img generation to maintain visual 
continuity between frames, creating a dream-like exploration experience.

Controls:
- Arrow Keys: Navigate through parameter space
- Space: Add effects to the prompt
- R: Reset effects
- Escape: Exit

The goal is to create smooth transitions where each keypress causes motion
in image space rather than just flipping through unrelated images.
"""

import sys
import os
import pygame
import requests
import base64
from io import BytesIO
from PIL import Image
import json
import time


class RemoteImgGen:
    """Remote image generator client for API server."""
    
    def __init__(self, server_url: str, initial_prompt: str = "a surreal dreamlike forest, ethereal lighting"):
        self.server_url = server_url.rstrip('/')
        self.prompt = initial_prompt
        self.current_image = None
        
        # Test connection
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Connected to server: {self.server_url}")
                print(f"   Model loaded: {health.get('model_loaded', False)}")
                print(f"   GPU available: {health.get('gpu_available', False)}")
            else:
                raise Exception(f"Server health check failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to server: {e}")
    
    def gen(self, prompt: str = None, **kwargs):
        """Generate initial image from text prompt."""
        use_prompt = prompt or self.prompt
        
        request_data = {
            "prompt": use_prompt,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "num_inference_steps": kwargs.get("num_inference_steps", 25),
            "guidance_scale": kwargs.get("guidance_scale", 7.5)
        }
        
        if "seed" in kwargs and kwargs["seed"] is not None:
            request_data["seed"] = kwargs["seed"]
        
        print(f"🎨 Generating: '{use_prompt[:50]}...'")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json=request_data,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Generation failed: {response.status_code} - {response.text}")
            
            result = response.json()
            image_data = base64.b64decode(result["image"])
            image = Image.open(BytesIO(image_data))
            
            # Store current image for img2img
            self.current_image = image
            
            elapsed = time.time() - start_time
            print(f"✅ Generated in {elapsed:.1f}s")
            
            return image
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            raise
    
    def gen_img2img(self, strength: float = 0.5, prompt: str = None, **kwargs):
        """Transform current image using img2img."""
        if self.current_image is None:
            print("⚠️ No current image, generating new one...")
            return self.gen(prompt, **kwargs)
        
        use_prompt = prompt or self.prompt
        
        # Encode current image to base64
        buffer = BytesIO()
        self.current_image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        request_data = {
            "prompt": use_prompt,
            "image": image_b64,
            "strength": strength,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "num_inference_steps": kwargs.get("num_inference_steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 7.5)
        }
        
        if "seed" in kwargs and kwargs["seed"] is not None:
            request_data["seed"] = kwargs["seed"]
        
        print(f"🔄 Transforming (strength={strength:.2f}): '{use_prompt[:40]}...'")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/img2img",
                json=request_data,
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Transformation failed: {response.status_code} - {response.text}")
            
            result = response.json()
            image_data = base64.b64decode(result["image"])
            image = Image.open(BytesIO(image_data))
            
            # Update current image
            self.current_image = image
            
            elapsed = time.time() - start_time
            print(f"✅ Transformed in {elapsed:.1f}s")
            
            return image
            
        except Exception as e:
            print(f"❌ Transformation error: {e}")
            raise


def show_image(img: Image.Image, window):
    """Display PIL Image in pygame window."""
    img = img.resize((512, 512))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    py_img = pygame.image.fromstring(data, size, mode)
    window.blit(py_img, (0, 0))
    pygame.display.flip()


def main():
    """Main keyboard navigation loop."""
    # Server configuration
    server_url = input("Enter server URL (or press Enter for http://172.28.5.21:8001): ").strip()
    if not server_url:
        server_url = "http://172.28.5.21:8001"
    
    # Initialize pygame
    pygame.init()
    win = pygame.display.set_mode((512, 512))
    pygame.display.set_caption("Dreamspace Navigator - Remote Server")
    
    # Initialize remote image generator
    print("🔮 Connecting to remote server...")
    try:
        img_gen = RemoteImgGen(
            server_url=server_url,
            initial_prompt="a surreal dreamlike forest, ethereal lighting"
        )
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        pygame.quit()
        return
    
    # Navigation parameters
    current_prompt = img_gen.prompt
    strength = 0.3  # Lower strength for smoother transitions
    strength_delta = 0.05
    
    # Effect modifiers to add variety
    effects = [
        "glowing light",
        "misty atmosphere", 
        "golden hour lighting",
        "ethereal glow",
        "deep shadows",
        "vibrant colors",
        "soft focus",
        "mystical energy",
        "dreamy blur",
        "cosmic energy"
    ]
    current_effects = []
    
    print("🎨 Generating initial image...")
    try:
        # Generate initial image
        image = img_gen.gen()
        show_image(image, win)
        print("🖼️ Initial image generated!")
    except Exception as e:
        print(f"❌ Failed to generate initial image: {e}")
        pygame.quit()
        return
    
    print("\\n🎮 Controls:")
    print("  ← → : Adjust transformation strength")
    print("  ↑ ↓ : Navigate through variations") 
    print("  Space: Add random effects")
    print("  R: Reset effects")
    print("  S: Save current image")
    print("  Escape: Exit")
    print(f"\\n🌟 Starting navigation with prompt: '{current_prompt}'")
    print(f"🌐 Server: {server_url}")
    
    # Main event loop
    running = True
    image_counter = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                
                # Initialize generation parameters
                gen_prompt = current_prompt
                if current_effects:
                    gen_prompt += ", " + ", ".join(current_effects)
                
                try:
                    # Navigation controls
                    if event.key == pygame.K_RIGHT:
                        # Increase strength for more dramatic changes
                        strength = min(1.0, strength + strength_delta)
                        print(f"🔥 Increased strength: {strength:.2f}")
                        image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                    
                    elif event.key == pygame.K_LEFT:
                        # Decrease strength for subtler changes
                        strength = max(0.1, strength - strength_delta)
                        print(f"🌙 Decreased strength: {strength:.2f}")
                        image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                    
                    elif event.key == pygame.K_UP:
                        # Move "forward" in image space with current settings
                        print("⬆️ Moving forward in image space...")
                        image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                    
                    elif event.key == pygame.K_DOWN:
                        # Move "backward" with lower strength for stability
                        print("⬇️ Moving backward in image space...")
                        image = img_gen.gen_img2img(strength=max(0.2, strength * 0.7), prompt=gen_prompt)
                    
                    elif event.key == pygame.K_SPACE:
                        # Add a random effect
                        import random
                        if len(current_effects) < 3:  # Limit effects to avoid prompt overflow
                            new_effect = random.choice([e for e in effects if e not in current_effects])
                            current_effects.append(new_effect)
                            print(f"✨ Added effect: '{new_effect}'")
                            gen_prompt = current_prompt + ", " + ", ".join(current_effects)
                            image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                        else:
                            print("🚫 Maximum effects reached. Press 'R' to reset.")
                            continue
                    
                    elif event.key == pygame.K_r:
                        # Reset effects
                        current_effects.clear()
                        print("🔄 Effects reset")
                        image = img_gen.gen_img2img(strength=strength, prompt=current_prompt)
                    
                    elif event.key == pygame.K_s:
                        # Save current image
                        if img_gen.current_image:
                            filename = f"dreamspace_nav_{image_counter:03d}.png"
                            img_gen.current_image.save(filename)
                            print(f"💾 Saved: {filename}")
                            image_counter += 1
                        continue
                    
                    else:
                        continue  # No action for other keys
                    
                    # Display the new image
                    show_image(image, win)
                    
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    continue
    
    # Cleanup
    print("🧹 Shutting down...")
    pygame.quit()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
