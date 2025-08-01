"""Animated keyboard-controlled image space navigation.

This creates smooth animated loops between navigation points, making transitions
feel more like floating through a continuous dreamspace rather than jumping
between discrete images.

Controls:
- Arrow Keys: Navigate through parameter space (generates new animation loops)
- Space: Add effects to the prompt
- A: Toggle animation on/off
- F: Adjust animation speed
- S: Save current frame
- Escape: Exit

Each navigation generates a batch of 32 variations that loop continuously,
creating flowing motion that makes transitions between states feel natural.
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
import threading
from collections import deque


class AnimatedRemoteImgGen:
    """Remote image generator with batch animation support."""
    
    def __init__(self, server_url: str, initial_prompt: str = "a surreal dreamlike forest, ethereal lighting"):
        self.server_url = server_url.rstrip('/')
        self.prompt = initial_prompt
        self.current_frames = []
        self.frame_index = 0
        self.is_generating = False
        
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
    
    def generate_animation_batch(self, prompt: str = None, batch_size: int = 32, **kwargs):
        """Generate a batch of variations for smooth animation."""
        use_prompt = prompt or self.prompt
        
        request_data = {
            "prompt": use_prompt,
            "batch_size": batch_size,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "num_inference_steps": kwargs.get("num_inference_steps", 20),  # Faster for batches
            "guidance_scale": kwargs.get("guidance_scale", 7.5)
        }
        
        if "seed" in kwargs and kwargs["seed"] is not None:
            request_data["seed"] = kwargs["seed"]
        
        print(f"🎬 Generating {batch_size} frame animation: '{use_prompt[:50]}...'")
        start_time = time.time()
        
        try:
            self.is_generating = True
            response = requests.post(
                f"{self.server_url}/generate_batch",
                json=request_data,
                timeout=300  # Longer timeout for batch generation
            )
            
            if response.status_code != 200:
                raise Exception(f"Batch generation failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Convert all base64 images to PIL Images
            frames = []
            for i, image_b64 in enumerate(result["images"]):
                image_data = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_data))
                frames.append(image)
                print(f"  Decoded frame {i+1}/{batch_size}")
            
            self.current_frames = frames
            self.frame_index = 0
            
            elapsed = time.time() - start_time
            print(f"✅ Animation batch generated in {elapsed:.1f}s ({elapsed/batch_size:.2f}s per frame)")
            
            return frames
            
        except Exception as e:
            print(f"❌ Batch generation error: {e}")
            raise
        finally:
            self.is_generating = False
    
    def get_current_frame(self):
        """Get the current animation frame."""
        if not self.current_frames:
            return None
        return self.current_frames[self.frame_index]
    
    def advance_frame(self):
        """Advance to next frame in animation loop."""
        if self.current_frames:
            self.frame_index = (self.frame_index + 1) % len(self.current_frames)
    
    def has_frames(self):
        """Check if animation frames are available."""
        return len(self.current_frames) > 0


def show_image(img: Image.Image, window):
    """Display PIL Image in pygame window."""
    if img is None:
        return
    img = img.resize((512, 512))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    py_img = pygame.image.fromstring(data, size, mode)
    window.blit(py_img, (0, 0))


def draw_ui(window, font, status_text, frame_info, generation_status):
    """Draw UI overlay."""
    # Semi-transparent overlay
    overlay = pygame.Surface((512, 100))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    window.blit(overlay, (0, 412))
    
    # Status text
    y_offset = 420
    for line in status_text:
        text_surface = font.render(line, True, (255, 255, 255))
        window.blit(text_surface, (10, y_offset))
        y_offset += 20
    
    # Frame info
    if frame_info:
        frame_surface = font.render(frame_info, True, (0, 255, 0))
        window.blit(frame_surface, (10, 490))
    
    # Generation status
    if generation_status:
        gen_surface = font.render(generation_status, True, (255, 255, 0))
        window.blit(gen_surface, (10, 470))


def main():
    """Main animated keyboard navigation loop."""
    # Server configuration
    server_url = input("Enter server URL (or press Enter for http://172.28.5.21:8001): ").strip()
    if not server_url:
        server_url = "http://172.28.5.21:8001"
    
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    win = pygame.display.set_mode((512, 512))
    pygame.display.set_caption("Dreamspace Navigator - Animated")
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()
    
    # Initialize animated image generator
    print("🔮 Connecting to remote server...")
    try:
        img_gen = AnimatedRemoteImgGen(
            server_url=server_url,
            initial_prompt="a surreal dreamlike forest, ethereal lighting"
        )
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        pygame.quit()
        return
    
    # Animation settings
    animation_enabled = True
    animation_speed = 8  # FPS for animation
    batch_size = 16  # Smaller batches for faster generation
    
    # Navigation parameters
    current_prompt = img_gen.prompt
    
    # Effect modifiers
    effects = [
        "glowing light", "misty atmosphere", "golden hour lighting",
        "ethereal glow", "deep shadows", "vibrant colors",
        "soft focus", "mystical energy", "dreamy blur", "cosmic energy"
    ]
    current_effects = []
    
    print(f"🎬 Generating initial animation batch ({batch_size} frames)...")
    try:
        # Generate initial animation
        def generate_initial():
            img_gen.generate_animation_batch(batch_size=batch_size)
        
        # Start generation in background
        gen_thread = threading.Thread(target=generate_initial)
        gen_thread.start()
        
        # Show loading screen
        loading_frame = 0
        while gen_thread.is_alive():
            win.fill((20, 20, 30))
            loading_text = f"Generating animation batch... {'.' * (loading_frame % 4)}"
            text_surface = font.render(loading_text, True, (255, 255, 255))
            win.blit(text_surface, (10, 250))
            pygame.display.flip()
            pygame.time.wait(100)
            loading_frame += 1
            
            # Handle quit events during loading
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("🛑 Stopping generation...")
                    pygame.quit()
                    return
        
        gen_thread.join()
        print("🖼️ Initial animation ready!")
        
    except Exception as e:
        print(f"❌ Failed to generate initial animation: {e}")
        pygame.quit()
        return
    
    print("\\n🎮 Animated Controls:")
    print("  ← → ↑ ↓ : Navigate (generates new animation loops)")
    print("  Space: Add random effects")
    print("  R: Reset effects")
    print("  A: Toggle animation on/off")
    print("  F: Cycle animation speed")
    print("  S: Save current frame")
    print("  Escape: Exit")
    print(f"\\n🌟 Starting with prompt: '{current_prompt}'")
    print(f"🌐 Server: {server_url}")
    
    # Main event loop
    running = True
    frame_counter = 0
    last_generation_time = time.time()
    
    # Background generation thread tracking
    generation_thread = None
    pending_prompt = None
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                
                # Skip input if currently generating
                if img_gen.is_generating:
                    print("⏳ Generation in progress, please wait...")
                    continue
                
                # Prepare generation parameters
                gen_prompt = current_prompt
                if current_effects:
                    gen_prompt += ", " + ", ".join(current_effects)
                
                new_animation_needed = False
                
                # Navigation controls
                if event.key == pygame.K_RIGHT:
                    gen_prompt += ", more vibrant, enhanced details"
                    new_animation_needed = True
                    print("➡️ Moving right in parameter space...")
                
                elif event.key == pygame.K_LEFT:
                    gen_prompt += ", softer, muted tones"
                    new_animation_needed = True
                    print("⬅️ Moving left in parameter space...")
                
                elif event.key == pygame.K_UP:
                    gen_prompt += ", brighter, uplifting mood"
                    new_animation_needed = True
                    print("⬆️ Moving up in parameter space...")
                
                elif event.key == pygame.K_DOWN:
                    gen_prompt += ", darker, mysterious atmosphere"
                    new_animation_needed = True
                    print("⬇️ Moving down in parameter space...")
                
                elif event.key == pygame.K_SPACE:
                    # Add a random effect
                    import random
                    if len(current_effects) < 3:
                        new_effect = random.choice([e for e in effects if e not in current_effects])
                        current_effects.append(new_effect)
                        print(f"✨ Added effect: '{new_effect}'")
                        gen_prompt = current_prompt + ", " + ", ".join(current_effects)
                        new_animation_needed = True
                    else:
                        print("🚫 Maximum effects reached. Press 'R' to reset.")
                
                elif event.key == pygame.K_r:
                    # Reset effects
                    current_effects.clear()
                    print("🔄 Effects reset")
                    gen_prompt = current_prompt
                    new_animation_needed = True
                
                elif event.key == pygame.K_a:
                    # Toggle animation
                    animation_enabled = not animation_enabled
                    status = "enabled" if animation_enabled else "disabled"
                    print(f"🎬 Animation {status}")
                
                elif event.key == pygame.K_f:
                    # Cycle animation speed
                    speeds = [4, 8, 12, 16, 24]
                    current_idx = speeds.index(animation_speed) if animation_speed in speeds else 0
                    animation_speed = speeds[(current_idx + 1) % len(speeds)]
                    print(f"⚡ Animation speed: {animation_speed} FPS")
                
                elif event.key == pygame.K_s:
                    # Save current frame
                    current_frame = img_gen.get_current_frame()
                    if current_frame:
                        filename = f"dreamspace_animated_{int(time.time())}.png"
                        current_frame.save(filename)
                        print(f"💾 Saved: {filename}")
                
                # Generate new animation if needed
                if new_animation_needed:
                    # Start generation in background thread
                    def generate_batch():
                        try:
                            img_gen.generate_animation_batch(prompt=gen_prompt, batch_size=batch_size)
                        except Exception as e:
                            print(f"❌ Background generation failed: {e}")
                    
                    generation_thread = threading.Thread(target=generate_batch)
                    generation_thread.start()
                    pending_prompt = gen_prompt
        
        # Update animation frame
        if animation_enabled and img_gen.has_frames():
            img_gen.advance_frame()
        
        # Draw current frame
        win.fill((0, 0, 0))
        current_frame = img_gen.get_current_frame()
        if current_frame:
            show_image(current_frame, win)
        
        # Prepare UI text
        status_lines = [
            f"Effects: {', '.join(current_effects) if current_effects else 'None'}",
            f"Animation: {'ON' if animation_enabled else 'OFF'} ({animation_speed} FPS)"
        ]
        
        frame_info = ""
        if img_gen.has_frames():
            frame_info = f"Frame: {img_gen.frame_index + 1}/{len(img_gen.current_frames)}"
        
        generation_status = ""
        if img_gen.is_generating:
            generation_status = "🎬 Generating new animation..."
        elif generation_thread and generation_thread.is_alive():
            generation_status = "🎬 Generating in background..."
        
        # Draw UI
        draw_ui(win, font, status_lines, frame_info, generation_status)
        
        pygame.display.flip()
        clock.tick(animation_speed if animation_enabled else 30)
        frame_counter += 1
    
    # Cleanup
    print("🧹 Shutting down...")
    if generation_thread and generation_thread.is_alive():
        print("⏳ Waiting for background generation to complete...")
        generation_thread.join(timeout=5)
    
    pygame.quit()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
