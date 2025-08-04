"""Animated keyboard-controlled image space navigation with artistic rhythm modulation.

This creates smooth animated loops between navigation points, making transitions
feel more like floating through a continuous dreamspace rather than jumping
between discrete images. Features artistic enhancements including:

- Randomized frame ordering for non-linear visual flow
- Client-side image interpolation for smooth transitions  
- Rhythmic modulation patterns (heartbeat, breathing, waves)
- Natural timing variations for organic feel

Controls:
- Arrow Keys: Navigate through parameter space (generates new animation loops)
- Space: Add effects to the prompt
- A: Toggle animation on/off
- F: Adjust animation speed
- S: Save current frame
- I: Toggle image interpolation
- X: Shuffle frame order
- 1-4: Switch rhythm patterns
- Escape: Exit

Each navigation generates a batch of variations that display in randomized order
with natural rhythm patterns, creating an organic, meditative visual experience.
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
import random
import math
from collections import deque

MORPHS = [
        "a cave",
        "shawdowy figure",
        "bright summer day",
        "twisted",
        "will-o-wisps",
    ]

class RhythmModulator:
    """Base class for rhythm modulation patterns."""
    
    def next_interval(self) -> float:
        """Return the time in seconds until the next transition."""
        raise NotImplementedError


class HeartbeatRhythm(RhythmModulator):
    """Heartbeat-like rhythm: boom-boom-pause-boom-boom-pause..."""
    
    def __init__(self, base_bpm: float = 60, variation: float = 0.2):
        self.base_bpm = base_bpm
        self.variation = variation
        self.beat_phase = 0  # 0=first beat, 1=pause between beats, 2=second beat, 3=long pause
        self.base_interval = 60.0 / base_bpm  # seconds per beat
        
    def next_interval(self) -> float:
        if self.beat_phase == 0:
            # First beat - quick transition
            interval = self.base_interval * 0.15
            self.beat_phase = 1
        elif self.beat_phase == 1:
            # Brief pause between boom-boom
            interval = self.base_interval * 0.15  
            self.beat_phase = 2
        #elif self.beat_phase == 2:
        #    # Second beat - quick transition
        #    interval = self.base_interval * 0.15
        #    self.beat_phase = 3
        else:
            # Long pause before next heartbeat
            interval = self.base_interval * 1.55
            self.beat_phase = 0
            
        # Add natural variation
        variation_factor = 1.0 + random.uniform(-self.variation, self.variation)
        return interval * variation_factor


class BreathingRhythm(RhythmModulator):
    """Breathing-like rhythm with slow inhale/exhale cycles."""
    
    def __init__(self, base_period: float = 8.0, variation: float = 0.15):
        self.base_period = base_period
        self.variation = variation
        self.phase = 0.0  # 0-1 breathing cycle
        
    def next_interval(self) -> float:
        # Sinusoidal breathing pattern
        breath_intensity = math.sin(self.phase * 2 * math.pi)
        # Map to interval (longer transitions for smoother breathing feel)
        base_interval = 1.5 + 2.0 * abs(breath_intensity)
        
        # Add variation
        variation_factor = 1.0 + random.uniform(-self.variation, self.variation)
        interval = base_interval * variation_factor
        
        # Advance phase
        self.phase = (self.phase + 0.08) % 1.0  # Slower phase progression
        
        return interval


class WaveRhythm(RhythmModulator):
    """Ocean wave-like rhythm with irregular intervals."""
    
    def __init__(self, base_interval: float = 2.5, chaos: float = 0.4):
        self.base_interval = base_interval
        self.chaos = chaos
        self.wave_phase = 0.0
        
    def next_interval(self) -> float:
        # Multiple overlapping sine waves for natural irregularity
        wave1 = math.sin(self.wave_phase * 1.3)
        wave2 = math.sin(self.wave_phase * 2.7) * 0.5
        wave3 = math.sin(self.wave_phase * 0.8) * 0.3
        
        combined_wave = wave1 + wave2 + wave3
        
        # Map to interval (longer base for smoother transitions)
        interval = self.base_interval * (1.0 + combined_wave * self.chaos)
        
        # Ensure positive interval with longer minimum
        interval = max(0.8, interval)
        
        # Advance phase
        self.wave_phase += 0.12
        
        return interval


class AnimatedRemoteImgGen:
    """Remote image generator with batch animation support and artistic modulation."""
    
    def __init__(self, server_url: str, initial_prompt: str = "a surreal dreamlike forest"):
        self.server_url = server_url.rstrip('/')
        self.prompt = initial_prompt
        self.current_frames = []
        self.frame_order = []  # Randomized indices for frame display
        self.frame_index = 0
        self.is_generating = False
        self.current_request_id = None
        self.cancel_current_request = False
        
        # Artistic modulation
        self.rhythm_modulator = HeartbeatRhythm(base_bpm=35)  # Very slow, meditative
        self.last_transition_time = time.time()
        self.current_frame_cache = None
        self.next_frame_cache = None
        self.interpolation_enabled = True
        self.interpolation_steps = 8  # Number of blend steps
        self.current_interpolation_step = 0
        
        # New batch introduction with smooth transitions
        self.new_batch_received = False
        self.new_batch_introduction_duration = 2.0  # Duration for smooth cross-batch transition (increased)
        self.batch_introduction_active = False
        self.old_batch_last_frame = None  # Store last frame from previous batch
        self.cross_batch_transition = False  # Flag for cross-batch interpolation
        self.cross_batch_start_time = None  # Track when cross-batch transition started
        
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
    
    def set_rhythm_modulator(self, modulator: RhythmModulator):
        """Change the rhythm modulation pattern."""
        self.rhythm_modulator = modulator
        print(f"🎵 Rhythm changed to: {modulator.__class__.__name__}")
    
    def _create_randomized_order(self):
        """Create a randomized order for frame display."""
        if not self.current_frames:
            return
        
        self.frame_order = list(range(len(self.current_frames)))
        random.shuffle(self.frame_order)
        self.frame_index = 0
        print(f"🎲 Randomized frame order: {len(self.frame_order)} frames")
    
    def _interpolate_frames(self, frame1: Image.Image, frame2: Image.Image, alpha: float) -> Image.Image:
        """Create a smooth blend between two frames."""
        if not frame1 or not frame2:
            return frame1 or frame2
        
        # Ensure both images are the same size
        if frame1.size != frame2.size:
            frame2 = frame2.resize(frame1.size)
        
        # Use PIL's blend function for smooth interpolation
        return Image.blend(frame1, frame2, alpha)
    
    def generate_animation_batch(self, prompt: str = None, batch_size: int = 32, request_id: str = None, **kwargs):
        """Generate a batch of variations for smooth animation."""
        use_prompt = prompt or self.prompt
        
        # Set up request tracking
        if request_id is None:
            request_id = str(time.time())
        
        self.current_request_id = request_id
        self.cancel_current_request = False
        
        request_data = {
            "prompt": use_prompt,
            "batch_size": batch_size,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "num_inference_steps": kwargs.get("num_inference_steps", 20),  # Faster for batches
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "seed": kwargs.get("seed", 42)  # Use consistent seed for testing coherence
        }
        
        print(f"🎬 Generating {batch_size} frame animation [{request_id[:8]}]: '{use_prompt[:50]}...'")
        print(f"🎯 Using seed: {request_data['seed']} for coherent variations")
        start_time = time.time()
        
        try:
            self.is_generating = True
            
            # Check for cancellation before making request
            if self.cancel_current_request or self.current_request_id != request_id:
                print(f"❌ Request {request_id[:8]} cancelled before starting")
                return []
            
            print(f"📡 Sending request to server...")
            
            response = requests.post(
                f"{self.server_url}/generate_batch",
                json=request_data,
                timeout=300  # Longer timeout for batch generation
            )
            
            # Check for cancellation after request
            if self.cancel_current_request or self.current_request_id != request_id:
                print(f"❌ Request {request_id[:8]} cancelled after server response")
                return []
            
            if response.status_code != 200:
                raise Exception(f"Batch generation failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Show chunking info if available
            metadata = result.get("metadata", {})
            if metadata.get("chunks", 1) > 1:
                chunk_info = metadata.get("chunk_sizes", [])
                print(f"  📊 Server used {metadata['chunks']} chunks: {chunk_info}")
            
            # Convert all base64 images to PIL Images
            frames = []
            for i, image_b64 in enumerate(result["images"]):
                # Check for cancellation during decoding
                if self.cancel_current_request or self.current_request_id != request_id:
                    print(f"❌ Request {request_id[:8]} cancelled during decoding at frame {i+1}")
                    return []
                
                image_data = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_data))
                frames.append(image)
                if i % 8 == 0:  # Print every 8th frame to reduce spam
                    print(f"  Decoded frame {i+1}/{batch_size} [{request_id[:8]}]")
            
            # Final check before updating frames
            if self.cancel_current_request or self.current_request_id != request_id:
                print(f"❌ Request {request_id[:8]} cancelled before frame update")
                return []
            
            # Store the current frame as the last frame from the old batch for smooth transition
            if self.current_frames and len(self.current_frames) > 0 and self.frame_order and len(self.frame_order) > 0:
                current_idx = self.frame_order[self.frame_index]
                self.old_batch_last_frame = self.current_frames[current_idx]
                print(f"🎭 Stored old batch last frame for smooth transition")
                
                # Set up cross-batch transition
                self.cross_batch_transition = True
                self.cross_batch_start_time = time.time()
                print(f"🎬 Starting {self.new_batch_introduction_duration}s cross-batch transition")
            
            # Update to new batch
            self.current_frames = frames
            self.frame_index = 0
            self._create_randomized_order()  # Create randomized display order
            
            elapsed = time.time() - start_time
            print(f"✅ Animation batch [{request_id[:8]}] completed in {elapsed:.1f}s ({elapsed/batch_size:.2f}s per frame)")
            
            return frames
            
        except Exception as e:
            if not (self.cancel_current_request or self.current_request_id != request_id):
                print(f"❌ Batch generation error [{request_id[:8]}]: {e}")
            raise
        finally:
            self.is_generating = False
    
    def cancel_current_generation(self):
        """Cancel the current generation request."""
        if self.is_generating:
            self.cancel_current_request = True
            print(f"🛑 Cancelling current generation request")
    
    def get_current_frame(self):
        """Get the current animation frame with optional interpolation."""
        if not self.current_frames or not self.frame_order:
            return None
        
        current_time = time.time()
        
        # Handle cross-batch transition FIRST - this takes priority
        if self.cross_batch_transition and self.old_batch_last_frame is not None and hasattr(self, 'cross_batch_start_time'):
            # Get the first frame of the new batch
            new_batch_first_idx = self.frame_order[0]
            new_batch_first_frame = self.current_frames[new_batch_first_idx]
            
            # Calculate interpolation progress for cross-batch transition
            time_since_cross_batch_start = current_time - self.cross_batch_start_time
            progress = min(time_since_cross_batch_start / self.new_batch_introduction_duration, 1.0)
            
            # Check if cross-batch transition should end
            if progress >= 1.0:
                # End cross-batch transition
                self.cross_batch_transition = False
                self.batch_introduction_active = False
                self.new_batch_received = False
                self.old_batch_last_frame = None
                self.last_transition_time = current_time  # Reset for normal transitions
                print(f"🎭 Cross-batch transition completed")
                return new_batch_first_frame
            
            if not self.interpolation_enabled:
                return new_batch_first_frame
            
            # Smooth interpolation curve (ease-in-out) for cross-batch transition
            smooth_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
            
            # Interpolate from old batch last frame to new batch first frame
            return self._interpolate_frames(self.old_batch_last_frame, new_batch_first_frame, smooth_progress)
        
        # Normal frame transitions - only when NOT in cross-batch transition
        # Get the interval for current transition
        if not hasattr(self, '_current_interval'):
            self._current_interval = self.rhythm_modulator.next_interval()
        
        # Check if it's time for the next transition
        if current_time - self.last_transition_time >= self._current_interval:
            self._advance_to_next_frame()
            self.last_transition_time = current_time
            self._current_interval = self.rhythm_modulator.next_interval()
        
        # Normal within-batch frame transitions
        current_idx = self.frame_order[self.frame_index]
        next_idx = self.frame_order[(self.frame_index + 1) % len(self.frame_order)]
        
        current_frame = self.current_frames[current_idx]
        next_frame = self.current_frames[next_idx]
        
        if not self.interpolation_enabled:
            return current_frame
        
        # Calculate interpolation progress within the transition interval
        time_since_transition = current_time - self.last_transition_time
        progress = min(time_since_transition / self._current_interval, 1.0)
        
        # Smooth interpolation curve (ease-in-out)
        smooth_progress = 0.5 - 0.5 * math.cos(progress * math.pi)
        
        # Return interpolated frame with stronger blending for smoother transitions
        return self._interpolate_frames(current_frame, next_frame, smooth_progress * 0.7)  # Much stronger blending
    
    def _advance_to_next_frame(self):
        """Advance to the next frame in the randomized sequence."""
        if self.frame_order:
            self.frame_index = (self.frame_index + 1) % len(self.frame_order)
    
    def advance_frame(self):
        """Manual frame advance (kept for compatibility but rhythm-based now)."""
        # This is now handled automatically by get_current_frame()
        pass
    
    def toggle_interpolation(self):
        """Toggle image interpolation on/off."""
        self.interpolation_enabled = not self.interpolation_enabled
        status = "enabled" if self.interpolation_enabled else "disabled"
        print(f"🎨 Interpolation {status}")
        return self.interpolation_enabled
    
    def has_frames(self):
        """Check if animation frames are available."""
        return len(self.current_frames) > 0


def show_image(img: Image.Image, window, target_width: int = 768, target_height: int = 768):
    """Display PIL Image in pygame window."""
    if img is None:
        return
    
    # Resize image to target dimensions
    img = img.resize((target_width, target_height))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    py_img = pygame.image.fromstring(data, size, mode)
    
    # Since window is now the exact size of the image, no centering needed
    window.blit(py_img, (0, 0))


def draw_ui(window, font, status_text, frame_info, generation_status, window_width=512, window_height=512):
    """Draw UI overlay."""
    # Semi-transparent overlay
    overlay = pygame.Surface((window_width, 100))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    window.blit(overlay, (0, window_height - 100))
    
    # Status text
    y_offset = window_height - 92
    for line in status_text:
        text_surface = font.render(line, True, (255, 255, 255))
        window.blit(text_surface, (10, y_offset))
        y_offset += 20
    
    # Frame info
    if frame_info:
        frame_surface = font.render(frame_info, True, (0, 255, 0))
        window.blit(frame_surface, (10, window_height - 22))
    
    # Generation status
    if generation_status:
        gen_surface = font.render(generation_status, True, (255, 255, 0))
        window.blit(gen_surface, (10, window_height - 42))


def main():
    """Main animated keyboard navigation loop."""
    # Server configuration
    server_url = input("Enter server URL (or press Enter for http://172.28.5.21:8001): ").strip()
    if not server_url:
        server_url = "http://172.28.5.21:8001"
    
    # Image size configuration
    size_input = input("Enter image size (512, 768, 1024) or WxH (e.g., 768x512): ").strip()
    if 'x' in size_input:
        try:
            width_str, height_str = size_input.split('x')
            image_width = int(width_str)
            image_height = int(height_str)
        except ValueError:
            image_width = image_height = 768
    elif size_input and size_input.isdigit():
        image_width = image_height = int(size_input)
    else:
        image_width = image_height = 768
    
    print(f"🖼️ Using image size: {image_width}x{image_height}")
    
    # Initialize pygame with the actual image dimensions
    pygame.init()
    pygame.font.init()
    win = pygame.display.set_mode((image_width, image_height))
    pygame.display.set_caption("Dreamspace Navigator - Animated")
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()
    
    # Initialize animated image generator
    print("🔮 Connecting to remote server...")
    try:
        img_gen = AnimatedRemoteImgGen(
            server_url=server_url,
            initial_prompt="a surreal dreamlike forest"
        )
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        pygame.quit()
        return
    
    # Animation settings
    animation_enabled = True
    animation_speed = 8  # FPS for animation
    # Use larger batch size since server now supports intelligent chunking
    batch_size = 16  # Server will automatically chunk this if needed
    
    # Generation parameters
    generation_params = {
        "width": image_width,
        "height": image_height
    }
    
    # Navigation parameters
    current_prompt = img_gen.prompt
    
    # Effect modifiers
    effects = MORPHS
    current_effects = []
    
    print(f"🎬 Generating initial animation batch ({batch_size} frames)...")
    try:
        # Generate initial animation
        def generate_initial():
            img_gen.generate_animation_batch(batch_size=batch_size, request_id="initial", **generation_params)
        
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
    print("  I: Toggle image interpolation")
    print("  X: Shuffle frame order")
    print("  1: Heartbeat rhythm (slow)")
    print("  2: Breathing rhythm")
    print("  3: Wave rhythm")
    print("  4: Heartbeat rhythm (fast)")
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
                
                elif event.key == pygame.K_i:
                    # Toggle interpolation
                    img_gen.toggle_interpolation()
                
                elif event.key == pygame.K_1:
                    # Switch to heartbeat rhythm
                    img_gen.set_rhythm_modulator(HeartbeatRhythm(base_bpm=35))
                
                elif event.key == pygame.K_2:
                    # Switch to breathing rhythm
                    img_gen.set_rhythm_modulator(BreathingRhythm(base_period=12.0))
                
                elif event.key == pygame.K_3:
                    # Switch to wave rhythm
                    img_gen.set_rhythm_modulator(WaveRhythm(base_interval=2.8, chaos=0.6))
                
                elif event.key == pygame.K_4:
                    # Switch to faster heartbeat
                    img_gen.set_rhythm_modulator(HeartbeatRhythm(base_bpm=55))
                
                elif event.key == pygame.K_x:
                    # Shuffle frame order
                    if img_gen.has_frames():
                        img_gen._create_randomized_order()
                        print("🔀 Frame order reshuffled!")
                
                # Generate new animation if needed
                if new_animation_needed:
                    # Cancel any existing generation
                    if generation_thread and generation_thread.is_alive():
                        print("� Cancelling previous generation...")
                        img_gen.cancel_current_generation()
                        generation_thread.join(timeout=2)  # Wait briefly for cancellation
                    
                    # Start generation in background thread
                    def generate_batch():
                        try:
                            # Generate a unique request ID
                            request_id = f"req_{time.time():.3f}"
                            print(f"🎬 Starting generation request {request_id}")
                            
                            frames = img_gen.generate_animation_batch(
                                prompt=gen_prompt, 
                                batch_size=batch_size, 
                                request_id=request_id,
                                **generation_params
                            )
                            
                            if frames:  # Only log if not cancelled
                                print(f"✅ Completed generation request {request_id}")
                        except Exception as e:
                            if not img_gen.cancel_current_request:  # Don't log errors from cancelled requests
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
            show_image(current_frame, win, image_width, image_height)
        
        # Prepare UI text
        rhythm_name = img_gen.rhythm_modulator.__class__.__name__.replace('Rhythm', '')
        interpolation_status = "ON" if img_gen.interpolation_enabled else "OFF"
        
        status_lines = [
            f"Effects: {', '.join(current_effects) if current_effects else 'None'}",
            f"Animation: {'ON' if animation_enabled else 'OFF'} ({animation_speed} FPS)",
            f"Rhythm: {rhythm_name} | Interpolation: {interpolation_status}"
        ]
        
        # Add batch introduction indicator
        if hasattr(img_gen, 'cross_batch_transition') and img_gen.cross_batch_transition:
            status_lines.append("🎭 CROSS-BATCH TRANSITION")
        elif hasattr(img_gen, 'batch_introduction_active') and img_gen.batch_introduction_active:
            status_lines.append("🎭 NEW BATCH SHOWCASE")
        
        frame_info = ""
        if img_gen.has_frames():
            frame_info = f"Frame: {img_gen.frame_index + 1}/{len(img_gen.current_frames)}"
        
        generation_status = ""
        if img_gen.is_generating:
            generation_status = "🎬 Generating animation..."
        elif generation_thread and generation_thread.is_alive():
            generation_status = "🎬 Background generation..."
        
        # Draw UI
        draw_ui(win, font, status_lines, frame_info, generation_status, image_width, image_height)
        
        pygame.display.flip()
        clock.tick(animation_speed if animation_enabled else 30)
        frame_counter += 1
    
    # Cleanup
    print("🧹 Shutting down...")
    if generation_thread and generation_thread.is_alive():
        print("🛑 Cancelling background generation...")
        img_gen.cancel_current_generation()
        generation_thread.join(timeout=3)
    
    pygame.quit()
    print("👋 Goodbye!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Animated Dreamspace Navigation")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height") 
    parser.add_argument("--size", type=int, help="Square image size (overrides width/height)")
    parser.add_argument("--server", type=str, default="http://172.28.5.21:8001",
                       help="Server URL")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Animation batch size")
    
    args = parser.parse_args()
    
    # Handle size configuration
    if args.size:
        image_width = image_height = args.size
    else:
        image_width = args.width
        image_height = args.height
    
    # Override the main function to use command line args
    def main_with_args():
        """Main with command line arguments."""
        server_url = args.server
        
        print(f"🖼️ Using image size: {image_width}x{image_height}")
        print(f"🌐 Server: {server_url}")
        
        # Initialize pygame with the actual image dimensions
        pygame.init()
        pygame.font.init()
        win = pygame.display.set_mode((image_width, image_height))
        pygame.display.set_caption("Dreamspace Navigator - Animated")
        font = pygame.font.Font(None, 20)
        clock = pygame.time.Clock()
        
        # Initialize animated image generator
        print("🔮 Connecting to remote server...")
        try:
            img_gen = AnimatedRemoteImgGen(
                server_url=server_url,
                initial_prompt="a surreal dreamlike forest"
            )
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            pygame.quit()
            return
        
        # Animation settings
        animation_enabled = True
        animation_speed = 8  # FPS for animation
        batch_size = args.batch_size  # Use command line batch size
        
        # Generation parameters
        generation_params = {
            "width": image_width,
            "height": image_height
        }
        
        # Navigation parameters
        current_prompt = img_gen.prompt
        
        # Effect modifiers
        effects = MORPHS
        current_effects = []
        
        print(f"🎬 Generating initial animation batch ({batch_size} frames)...")
        try:
            # Generate initial animation
            def generate_initial():
                img_gen.generate_animation_batch(batch_size=batch_size, request_id="initial", **generation_params)
            
            # Start generation in background
            gen_thread = threading.Thread(target=generate_initial)
            gen_thread.start()
            
            # Show loading screen
            loading_frame = 0
            while gen_thread.is_alive():
                win.fill((20, 20, 30))
                loading_text = f"Generating animation batch... {'.' * (loading_frame % 4)}"
                text_surface = font.render(loading_text, True, (255, 255, 255))
                win.blit(text_surface, (10, image_height // 2))
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
        print("  I: Toggle image interpolation")
        print("  X: Shuffle frame order")
        print("  1: Heartbeat rhythm (slow)")
        print("  2: Breathing rhythm")
        print("  3: Wave rhythm")
        print("  4: Heartbeat rhythm (fast)")
        print("  Escape: Exit")
        print(f"\\n🌟 Starting with prompt: '{current_prompt}'")
        print(f"🌐 Server: {server_url}")
        print(f"📏 Size: {image_width}x{image_height}")
        
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
                            filename = f"dreamspace_animated_{image_width}x{image_height}_{int(time.time())}.png"
                            current_frame.save(filename)
                            print(f"💾 Saved: {filename}")
                    
                    elif event.key == pygame.K_i:
                        # Toggle interpolation
                        img_gen.toggle_interpolation()
                    
                    elif event.key == pygame.K_1:
                        # Switch to heartbeat rhythm
                        img_gen.set_rhythm_modulator(HeartbeatRhythm(base_bpm=35))
                    
                    elif event.key == pygame.K_2:
                        # Switch to breathing rhythm
                        img_gen.set_rhythm_modulator(BreathingRhythm(base_period=12.0))
                    
                    elif event.key == pygame.K_3:
                        # Switch to wave rhythm
                        img_gen.set_rhythm_modulator(WaveRhythm(base_interval=2.8, chaos=0.6))
                    
                    elif event.key == pygame.K_4:
                        # Switch to faster heartbeat
                        img_gen.set_rhythm_modulator(HeartbeatRhythm(base_bpm=55))
                    
                    elif event.key == pygame.K_x:
                        # Shuffle frame order
                        if img_gen.has_frames():
                            img_gen._create_randomized_order()
                            print("🔀 Frame order reshuffled!")
                    
                    # Generate new animation if needed
                    if new_animation_needed:
                        # Cancel any existing generation
                        if generation_thread and generation_thread.is_alive():
                            print("� Cancelling previous generation...")
                            img_gen.cancel_current_generation()
                            generation_thread.join(timeout=2)  # Wait briefly for cancellation
                        
                        # Start generation in background thread
                        def generate_batch():
                            try:
                                # Generate a unique request ID
                                request_id = f"req_{time.time():.3f}"
                                print(f"🎬 Starting generation request {request_id}")
                                
                                frames = img_gen.generate_animation_batch(
                                    prompt=gen_prompt, 
                                    batch_size=batch_size, 
                                    request_id=request_id,
                                    **generation_params
                                )
                                
                                if frames:  # Only log if not cancelled
                                    print(f"✅ Completed generation request {request_id}")
                            except Exception as e:
                                if not img_gen.cancel_current_request:  # Don't log errors from cancelled requests
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
                show_image(current_frame, win, image_width, image_height)
            
            # Prepare UI text
            rhythm_name = img_gen.rhythm_modulator.__class__.__name__.replace('Rhythm', '')
            interpolation_status = "ON" if img_gen.interpolation_enabled else "OFF"
            
            status_lines = [
                f"Effects: {', '.join(current_effects) if current_effects else 'None'}",
                f"Animation: {'ON' if animation_enabled else 'OFF'} ({animation_speed} FPS)",
                f"Rhythm: {rhythm_name} | Interpolation: {interpolation_status}"
            ]
            
            # Add batch introduction indicator
            if hasattr(img_gen, 'cross_batch_transition') and img_gen.cross_batch_transition:
                status_lines.append("🎭 CROSS-BATCH TRANSITION")
            elif hasattr(img_gen, 'batch_introduction_active') and img_gen.batch_introduction_active:
                status_lines.append("🎭 NEW BATCH SHOWCASE")
            
            frame_info = ""
            if img_gen.has_frames():
                frame_info = f"Frame: {img_gen.frame_index + 1}/{len(img_gen.current_frames)}"
            
            generation_status = ""
            if img_gen.is_generating:
                generation_status = "🎬 Generating animation..."
            elif generation_thread and generation_thread.is_alive():
                generation_status = "🎬 Background generation..."
            
            # Draw UI
            draw_ui(win, font, status_lines, frame_info, generation_status, image_width, image_height)
            
            pygame.display.flip()
            clock.tick(animation_speed if animation_enabled else 30)
            frame_counter += 1
        
        # Cleanup
        print("🧹 Shutting down...")
        if generation_thread and generation_thread.is_alive():
            print("🛑 Cancelling background generation...")
            img_gen.cancel_current_generation()
            generation_thread.join(timeout=3)
        
        pygame.quit()
        print("👋 Goodbye!")
    
    main_with_args()
