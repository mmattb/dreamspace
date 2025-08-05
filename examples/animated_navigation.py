"""Animated Dreamspace Navigation Example

This example demonstrates how to use the dreamspace animation framework to create
smooth, rhythm-based transitions through AI-generated visual landscapes.

Features:
- Rhythm-based frame transitions (heartbeat, breathing, wave patterns)
- Smooth interpolation between frames and batches
- Real-time parameter space navigation
- Background batch generation with seamless transitions
- Interactive keyboard controls

Usage:
    # Run with interactive prompts
    PYTHONPATH=src python examples/animated_navigation.py
    
    # Run with command line arguments
    PYTHONPATH=src python examples/animated_navigation.py --size 512 --batch-size 8
    
    # Run with custom server
    PYTHONPATH=src python examples/animated_navigation.py --server http://localhost:8001

Controls:
  Arrow Keys: Navigate parameter space
  Space: Add random effects
  R: Reset effects
  A: Toggle animation
  F: Cycle animation speed
  I: Toggle interpolation
  1-4: Switch rhythm patterns
  S: Save current frame
  X: Shuffle frame order
"""

import pygame
import time
import threading
import random
from typing import List, Optional, Tuple

# Import from the main dreamspace library
from dreamspace.core.animation import HeartbeatRhythm, BreathingRhythm, WaveRhythm
from dreamspace.core.remote_generator import AnimatedRemoteImgGen
from dreamspace.cli.navigation import (
    parse_arguments, get_image_dimensions, get_interactive_config, print_welcome_message
)

MORPHS = [
    "figure cloaked in white",
    "steampunk trees",
    "forest spirits",
]


class DreamspaceNavigator:
    """Main application controller for the animated dreamspace navigator."""
    
    def __init__(self, server_url: str, image_size: Tuple[int, int], batch_size: int, initial_prompt: str):
        self.server_url = server_url
        self.image_width, self.image_height = image_size
        self.batch_size = batch_size
        self.initial_prompt = initial_prompt
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((self.image_width, self.image_height))
        pygame.display.set_caption("Dreamspace Navigator - Animated")
        self.font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()
        
        # Initialize image generator
        print("🔮 Connecting to remote server...")
        try:
            self.img_gen = AnimatedRemoteImgGen(server_url, initial_prompt)
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            pygame.quit()
            raise
        
        # Animation state
        self.animation_enabled = True
        self.animation_speed = 8  # FPS
        
        # Generation parameters
        self.generation_params = {
            "width": self.image_width,
            "height": self.image_height
        }
        
        # Prompt state
        self.current_prompt = initial_prompt
        self.current_effects: List[str] = []
        
        # Background generation
        self.generation_thread: Optional[threading.Thread] = None
        
        print(f"✅ Navigator initialized: {self.image_width}x{self.image_height}")
    
    def show_image(self, img, window, target_width: int, target_height: int):
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
    
    def draw_ui(self, status_text: List[str], frame_info: str, generation_status: str):
        """Draw UI overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.image_width, 100))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, (0, self.image_height - 100))
        
        # Status text
        y_offset = self.image_height - 92
        for line in status_text:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.window.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        # Frame info
        if frame_info:
            frame_surface = self.font.render(frame_info, True, (0, 255, 0))
            self.window.blit(frame_surface, (10, self.image_height - 22))
        
        # Generation status
        if generation_status:
            gen_surface = self.font.render(generation_status, True, (255, 255, 0))
            self.window.blit(gen_surface, (10, self.image_height - 42))
    
    def generate_initial_batch(self):
        """Generate the initial animation batch."""
        print(f"🎬 Generating initial animation batch ({self.batch_size} frames)...")
        
        def generate_initial():
            start_time = time.time()
            print(f"⏰ Initial batch start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
            
            self.img_gen.generate_animation_batch(
                batch_size=self.batch_size, 
                request_id="initial", 
                **self.generation_params
            )
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Initial batch generation time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
            print(f"📊 Initial batch rate: {self.batch_size/duration:.1f} images/second")
        
        # Start generation in background
        gen_thread = threading.Thread(target=generate_initial)
        gen_thread.start()
        
        # Show loading screen
        loading_frame = 0
        while gen_thread.is_alive():
            self.window.fill((20, 20, 30))
            loading_text = f"Generating animation batch... {'.' * (loading_frame % 4)}"
            text_surface = self.font.render(loading_text, True, (255, 255, 255))
            self.window.blit(text_surface, (10, self.image_height // 2))
            pygame.display.flip()
            pygame.time.wait(100)
            loading_frame += 1
            
            # Handle quit events during loading
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("🛑 Stopping generation...")
                    pygame.quit()
                    return False
        
        gen_thread.join()
        print("🖼️ Initial animation ready!")
        return True
    
    def start_background_generation(self, prompt: str):
        """Start background generation with the given prompt."""
        # Cancel any existing generation
        if self.generation_thread and self.generation_thread.is_alive():
            print("🛑 Cancelling previous generation...")
            self.img_gen.cancel_current_generation()
            self.generation_thread.join(timeout=2)
        
        def generate_batch():
            try:
                request_id = f"req_{time.time():.3f}"
                start_time = time.time()
                print(f"🎬 Starting generation request {request_id}")
                print(f"⏰ Request start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
                
                frames = self.img_gen.generate_animation_batch(
                    prompt=prompt,
                    batch_size=self.batch_size,
                    request_id=request_id,
                    **self.generation_params
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                if frames:
                    print(f"✅ Completed generation request {request_id}")
                    print(f"⏱️ Generation time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
                    print(f"📊 Rate: {self.batch_size/duration:.1f} images/second")
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                if not self.img_gen.cancel_current_request:
                    print(f"❌ Background generation failed after {duration:.2f}s: {e}")
        
        self.generation_thread = threading.Thread(target=generate_batch)
        self.generation_thread.start()
    
    def handle_navigation_key(self, key: int) -> bool:
        """Handle navigation keys and return True if new animation needed."""
        gen_prompt = self.current_prompt
        if self.current_effects:
            gen_prompt += ", " + ", ".join(self.current_effects)
        
        if key == pygame.K_RIGHT:
            gen_prompt += ", more vibrant, enhanced details"
            print("➡️ Moving right in parameter space...")
            
        elif key == pygame.K_LEFT:
            gen_prompt += ", softer, muted tones"
            print("⬅️ Moving left in parameter space...")
            
        elif key == pygame.K_UP:
            gen_prompt += ", brighter, uplifting mood"
            print("⬆️ Moving up in parameter space...")
            
        elif key == pygame.K_DOWN:
            gen_prompt += ", darker, mysterious atmosphere"
            print("⬇️ Moving down in parameter space...")
            
        elif key == pygame.K_SPACE:
            if len(self.current_effects) < 3:
                new_effect = random.choice([e for e in MORPHS if e not in self.current_effects])
                self.current_effects.append(new_effect)
                print(f"✨ Added effect: '{new_effect}'")
                gen_prompt = self.current_prompt + ", " + ", ".join(self.current_effects)
            else:
                print("🚫 Maximum effects reached. Press 'R' to reset.")
                return False
            
        elif key == pygame.K_r:
            self.current_effects.clear()
            print("🔄 Effects reset")
            gen_prompt = self.current_prompt
            
        else:
            return False
        
        # Start background generation
        self.start_background_generation(gen_prompt)
        return True
    
    def handle_control_key(self, key: int):
        """Handle control keys (non-navigation)."""
        if key == pygame.K_a:
            self.animation_enabled = not self.animation_enabled
            status = "enabled" if self.animation_enabled else "disabled"
            print(f"🎬 Animation {status}")
            
        elif key == pygame.K_f:
            speeds = [4, 8, 12, 16, 24]
            current_idx = speeds.index(self.animation_speed) if self.animation_speed in speeds else 0
            self.animation_speed = speeds[(current_idx + 1) % len(speeds)]
            print(f"⚡ Animation speed: {self.animation_speed} FPS")
            
        elif key == pygame.K_s:
            current_frame = self.img_gen.get_current_frame()
            if current_frame:
                filename = f"dreamspace_animated_{self.image_width}x{self.image_height}_{int(time.time())}.png"
                current_frame.save(filename)
                print(f"💾 Saved: {filename}")
                
        elif key == pygame.K_i:
            self.img_gen.animation_controller.toggle_interpolation()
            
        elif key == pygame.K_x:
            self.img_gen.shuffle_frames()
            
        elif key == pygame.K_1:
            self.img_gen.animation_controller.set_rhythm_modulator(HeartbeatRhythm(base_bpm=35))
            
        elif key == pygame.K_2:
            self.img_gen.animation_controller.set_rhythm_modulator(BreathingRhythm(base_period=12.0))
            
        elif key == pygame.K_3:
            self.img_gen.animation_controller.set_rhythm_modulator(WaveRhythm(base_interval=2.8, chaos=0.6))
            
        elif key == pygame.K_4:
            self.img_gen.animation_controller.set_rhythm_modulator(HeartbeatRhythm(base_bpm=55))
    
    def get_status(self):
        """Get current status information from the animation system."""
        return self.img_gen.get_status()
    
    def run(self):
        """Main application loop."""
        # Generate initial batch
        if not self.generate_initial_batch():
            return
        
        print_welcome_message(self.server_url, (self.image_width, self.image_height), self.batch_size)
        print(f"🌟 Starting with prompt: '{self.current_prompt}'")
        
        running = True
        frame_counter = 0
        
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
                    
                    # Try navigation keys first
                    if not self.handle_navigation_key(event.key):
                        # If not a navigation key, try control keys
                        self.handle_control_key(event.key)
            
            # Update animation frame
            if self.animation_enabled and self.img_gen.has_frames():
                self.img_gen.advance_frame()
            
            # Draw current frame
            self.window.fill((0, 0, 0))
            current_frame = self.img_gen.get_current_frame()
            if current_frame:
                self.show_image(current_frame, self.window, self.image_width, self.image_height)
            
            # Prepare UI status
            status = self.img_gen.get_status()
            rhythm_name = status['rhythm_type'].replace('Rhythm', '')
            interpolation_status = "ON" if status['interpolation_enabled'] else "OFF"
            
            status_lines = [
                f"Effects: {', '.join(self.current_effects) if self.current_effects else 'None'}",
                f"Animation: {'ON' if self.animation_enabled else 'OFF'} ({self.animation_speed} FPS)",
                f"Rhythm: {rhythm_name} | Interpolation: {interpolation_status}"
            ]
            
            if status['cross_batch_active']:
                status_lines.append("🎭 CROSS-BATCH TRANSITION")
            
            frame_info = ""
            if status['has_frames']:
                frame_info = f"Frame: {status['current_frame_index']}/{status['frame_count']}"
            
            generation_status = ""
            if status['is_generating']:
                generation_status = "🎬 Generating animation..."
            elif self.generation_thread and self.generation_thread.is_alive():
                generation_status = "🎬 Background generation..."
            
            # Draw UI
            self.draw_ui(status_lines, frame_info, generation_status)
            
            pygame.display.flip()
            self.clock.tick(self.animation_speed if self.animation_enabled else 30)
            frame_counter += 1
        
        # Cleanup
        print("🧹 Shutting down...")
        if self.generation_thread and self.generation_thread.is_alive():
            print("🛑 Cancelling background generation...")
            self.img_gen.cancel_current_generation()
            self.generation_thread.join(timeout=3)
        
        pygame.quit()
        print("👋 Goodbye!")


def main():
    """Main function using interactive prompts."""
    server_url, image_size = get_interactive_config()
    
    navigator = DreamspaceNavigator(
        server_url=server_url,
        image_size=image_size,
        batch_size=16,
        initial_prompt="strange bright forest land, steampunk trees",
    )
    
    navigator.run()


def main_with_args():
    """Main function using command line arguments."""
    args = parse_arguments()
    
    # Get configuration from args
    server_url = args.server
    image_size = get_image_dimensions(args)
    batch_size = args.batch_size
    initial_prompt = args.prompt
    
    navigator = DreamspaceNavigator(
        server_url=server_url,
        image_size=image_size,
        batch_size=batch_size,
        initial_prompt=initial_prompt
    )
    
    # Set initial configuration
    if args.no_interpolation:
        navigator.img_gen.animation_controller.interpolation_enabled = False
    
    navigator.animation_speed = args.fps
    
    # New latent wiggle and noise magnitude settings
    navigator.latent_wiggle = args.latent_wiggle
    navigator.noise_magnitude = args.noise_magnitude
    navigator.bifurcation_step = args.bifurcation_step
    
    # If bifurcated-wiggle flag is used, ensure bifurcation_step is set
    if hasattr(args, 'bifurcated_wiggle') and args.bifurcated_wiggle:
        navigator.bifurcation_step = max(navigator.bifurcation_step, 3)  # Ensure minimum of 3
        navigator.latent_wiggle = True  # Imply latent wiggle is enabled
    
    # Add noise_magnitude and bifurcation_step to generation parameters
    navigator.generation_params["noise_magnitude"] = navigator.noise_magnitude
    navigator.generation_params["bifurcation_step"] = navigator.bifurcation_step

    if navigator.latent_wiggle:
        method_name = "Bifurcated Latent Wiggle" if navigator.bifurcation_step > 0 else "Latent Wiggle"
        print(f"✨ {method_name} Pipeline Enabled")
        print(f"🔧 Noise Magnitude: {navigator.noise_magnitude}")
        if navigator.bifurcation_step > 0:
            print(f"🔀 Bifurcation Step: {navigator.bifurcation_step}")
    
    navigator.run()


if __name__ == "__main__":
    import sys
    
    # Check if we have command line arguments (other than script name)
    if len(sys.argv) > 1:
        main_with_args()
    else:
        main()
