"""Animated Dreamspace Navigation Example

This example demonstrates how to use the dreamspace animation framework to create
smooth, rhythm-based transitions through AI-generated visual landscapes.

Features:
- Rhythm-based frame transitions (heartbeat, breathing, wave patterns)
- Smooth interpolation between frames and batches
- Real-time parameter space navigation
- Background batch generation with seamless transitions
- Interactive keyboard controls
- Maximized window display for full-screen viewing

Usage:
    # Run with interactive prompts
    PYTHONPATH=src python examples/animated_navigation.py
    
    # Run with command line arguments
    PYTHONPATH=src python examples/animated_navigation.py --size 512 --batch-size 8
    
    # Run maximized to fill screen while maintaining aspect ratio
    PYTHONPATH=src python examples/animated_navigation.py --maximize
    
    # Run with custom server (bifurcated wiggle is now the default)
    PYTHONPATH=src python examples/animated_navigation.py --server http://localhost:8001
    
    # Run with tensor output format for maximum speed
    PYTHONPATH=src python examples/animated_navigation.py --output-format tensor
    
    # Run with custom noise magnitude and bifurcation step
    PYTHONPATH=src python examples/animated_navigation.py --noise-magnitude 0.5 --bifurcation-step 5
    
    # Save all images to a directory (clears directory on each new batch)
    PYTHONPATH=src python examples/animated_navigation.py --output-dir ./generated_images
    
    # Multi-prompt interpolation sequence with looping and consistent composition
    PYTHONPATH=src python examples/animated_navigation.py --prompt-list "forest scene" "desert landscape" "ocean view" "mountain vista" --batch-size 8 --output-dir ./sequence --latent-cookie 12345 --seed 42
    
    
    # Multi-prompt interpolation sequence with looping and consistent composition
    PYTHONPATH=src python examples/animated_navigation.py --prompt-list "forest scene" "desert landscape" "ocean view" "mountain vista" --batch-size 8 --output-dir ./sequence --latent-cookie 12345 --seed 42

Controls:
  Arrow Keys: Navigate parameter space
  Space: Add random effects
  R: Reset effects
  A: Toggle animation
  F: Cycle animation speed
  I: Toggle interpolation
  1-6: Switch rhythm patterns
  S: Save current frame
  X: Shuffle frame order
"""

import pygame
import time
import threading
import random
import os
import shutil
from typing import List, Optional, Tuple
from PIL import Image
from screeninfo import get_monitors
from tqdm import tqdm

# Import from the main dreamspace library
from dreamspace.core.animation import HeartbeatRhythm, BreathingRhythm, WaveRhythm, LinearRhythm, ContinuousLinearRhythm
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
    
    def __init__(self, server_url: str, initial_prompt: str, image_size: Tuple[int, int] = (2048, 1280), batch_size: int = 2, 
                 noise_magnitude: float = 0.17, bifurcation_step: int = 3, output_format: str = "jpeg",
                 maximize_window: bool = False, interpolation_mode: bool = False, prompt2: str = None, output_dir: str = None, prompt_list: List[str] = None, latent_cookie: int = None, seed: int = None):
        self.server_url = server_url
        self.original_image_width, self.original_image_height = image_size
        self.batch_size = batch_size
        self.initial_prompt = initial_prompt
        self.noise_magnitude = noise_magnitude
        self.bifurcation_step = bifurcation_step
        self.output_format = output_format
        self.maximize_window = maximize_window
        self.interpolation_mode = interpolation_mode
        self.prompt2 = prompt2
        self.output_dir = output_dir
        self.prompt_list = prompt_list
        self.latent_cookie = latent_cookie
        self.seed = seed
        
        # Multi-prompt sequence state
        self.is_multi_prompt_mode = prompt_list is not None and len(prompt_list) > 1
        self.current_prompt_pair_index = 0
        self.total_frames_saved = 0  # Track consecutive frame numbering
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Calculate optimal window and display sizes
        if self.maximize_window:
            try:
                # Use screeninfo to get the actual screen resolution
                monitor = get_monitors()[0]  # Get the primary monitor
                screen_width, screen_height = monitor.width, monitor.height
                print(f"🔍 Using screeninfo resolution: {screen_width}x{screen_height}")
            except Exception as e:
                print(f"❌ Failed to get resolution with screeninfo: {e}")
                # Fallback to pygame detection
                info = pygame.display.Info()
                screen_width, screen_height = info.current_w, info.current_h
                print(f"🔍 Using pygame resolution (fallback): {screen_width}x{screen_height}")

            # Use full screen dimensions for window
            self.window_width = screen_width
            self.window_height = screen_height

            # Keep original generation size - don't scale up generation!
            self.image_width = self.original_image_width
            self.image_height = self.original_image_height

            # Calculate display scale factor for reference
            display_scale = max(self.window_width / self.image_width, 
                                self.window_height / self.image_height)

            print(f"🖥️ Screen size: {screen_width}x{screen_height}")
            print(f"🎨 Generation size: {self.image_width}x{self.image_height} (unchanged)")
            print(f"🪟 Window size: {self.window_width}x{self.window_height} (fullscreen)")
            print(f"🔍 Display scale: {display_scale:.2f}x (images will be scaled up for display)")
            
        else:
            # Use original size for both window and generation
            self.window_width = self.image_width = self.original_image_width
            self.window_height = self.image_height = self.original_image_height
        
        self.window = pygame.display.set_mode((self.window_width, self.window_height), 
                                             pygame.FULLSCREEN if self.maximize_window else 0)
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
        self.animation_speed = 16  # FPS
        
        # Generation parameters
        self.generation_params = {
            "width": self.image_width,
            "height": self.image_height,
            "noise_magnitude": self.noise_magnitude,
            "bifurcation_step": self.bifurcation_step,
            "output_format": self.output_format
        }
        
        # Add latent cookie if specified
        if self.latent_cookie is not None:
            self.generation_params["latent_cookie"] = self.latent_cookie
            print(f"🍪 Using latent cookie {self.latent_cookie} for consistent composition")
        
        # Add seed if specified  
        if self.seed is not None:
            self.generation_params["seed"] = self.seed
            print(f"🎲 Using seed {self.seed} for consistent generation")
        
        # Prompt state
        self.current_prompt = initial_prompt
        self.current_effects: List[str] = []
        
        # Background generation
        self.generation_thread: Optional[threading.Thread] = None
        
        print(f"✅ Navigator initialized: {self.image_width}x{self.image_height} → {self.window_width}x{self.window_height}")
        
        # Show output directory status
        if self.output_dir:
            print(f"📁 Output directory: {self.output_dir} (will be cleared on each batch)")
        else:
            print(f"📁 No output directory specified (images will not be saved to disk)")
        
        # Show mode status
        if self.is_multi_prompt_mode:
            print(f"🌈 Multi-prompt mode: {len(self.prompt_list)} prompts with looping")
        elif self.interpolation_mode and self.prompt2:
            print(f"🌈 Two-prompt interpolation mode enabled")
        else:
            print(f"🎬 Regular animation mode (wiggle variations)")
    
    def show_image(self, img, window, target_width: int, target_height: int):
        """Display PIL Image in pygame window, scaling to fill screen while maintaining aspect ratio."""
        if img is None:
            return
        
        # Calculate the scaling to fill the screen (touch opposite boundaries)
        img_width, img_height = img.size
        
        # Calculate scale factors for both dimensions
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        
        # Use the SMALLER scale factor to ensure image fits completely (touches 2 opposite sides, black borders on other 2)
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image to calculated dimensions
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center the image in the window (black borders will appear on the sides that don't touch)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Clear the window with black background
        window.fill((0, 0, 0))
        
        # Convert PIL image to pygame surface
        mode = img.mode
        size = img.size
        data = img.tobytes()
        py_img = pygame.image.fromstring(data, size, mode)
        
        # Blit the image (will crop if larger than window)
        window.blit(py_img, (x_offset, y_offset))
    
    def draw_ui(self, status_text: List[str], frame_info: str, generation_status: str):
        """Draw UI overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_width, 100))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, (0, self.window_height - 100))
        
        # Status text
        y_offset = self.window_height - 92
        for line in status_text:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.window.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        # Frame info
        if frame_info:
            frame_surface = self.font.render(frame_info, True, (0, 255, 0))
            self.window.blit(frame_surface, (10, self.window_height - 22))
        
        # Generation status
        if generation_status:
            gen_surface = self.font.render(generation_status, True, (255, 255, 0))
            self.window.blit(gen_surface, (10, self.window_height - 42))
    
    def save_images_to_directory(self):
        """Save all current animation frames to the output directory."""
        if not self.output_dir or not self.img_gen.has_frames():
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Clear directory contents
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"⚠️ Failed to delete {file_path}: {e}")
            
            # Save all current frames
            frames = self.img_gen.current_frames
            if frames:
                print(f"💾 Saving {len(frames)} images to {self.output_dir}")
                for i, frame in enumerate(frames):
                    # Determine file extension based on output format
                    if self.output_format.lower() == "jpeg":
                        ext = "jpg"
                    elif self.output_format.lower() == "png":
                        ext = "png"
                    else:
                        ext = "png"  # Default fallback
                    
                    filename = f"frame_{i:04d}.{ext}"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save with appropriate format and quality
                    if ext == "jpg":
                        frame.save(filepath, format='JPEG', quality=90, optimize=True)
                    else:
                        frame.save(filepath, format='PNG', optimize=True)
                
                print(f"✅ Saved {len(frames)} frames to {self.output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving images to directory: {e}")
    
    def save_images_to_directory_with_consecutive_numbering(self, start_frame_number: int = None):
        """Save all current animation frames to the output directory with consecutive numbering."""
        if not self.output_dir or not self.img_gen.has_frames():
            return 0
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save all current frames with consecutive numbering
            frames = self.img_gen.current_frames
            if frames:
                start_num = start_frame_number if start_frame_number is not None else self.total_frames_saved
                
                for i, frame in enumerate(frames):
                    # Determine file extension based on output format
                    if self.output_format.lower() == "jpeg":
                        ext = "jpg"
                    elif self.output_format.lower() == "png":
                        ext = "png"
                    else:
                        ext = "png"  # Default fallback
                    
                    frame_number = start_num + i
                    filename = f"frame_{frame_number:06d}.{ext}"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save with appropriate format and quality
                    if ext == "jpg":
                        frame.save(filepath, format='JPEG', quality=90, optimize=True)
                    else:
                        frame.save(filepath, format='PNG', optimize=True)
                
                print(f"💾 Saved frames {start_num}-{start_num + len(frames) - 1} to {self.output_dir}")
                return len(frames)
            
            return 0
            
        except Exception as e:
            print(f"❌ Error saving images to directory: {e}")
            return 0
    
    def generate_multi_prompt_sequence(self):
        """Generate interpolated animations between all prompts in the list, with looping."""
        if not self.is_multi_prompt_mode:
            return False
        
        prompts = self.prompt_list
        num_prompts = len(prompts)
        
        # Create pairs including the loop back to start
        prompt_pairs = []
        for i in range(num_prompts):
            next_i = (i + 1) % num_prompts
            prompt_pairs.append((prompts[i], prompts[next_i]))
        
        print(f"🌈 Starting multi-prompt interpolation sequence:")
        print(f"📝 {num_prompts} prompts → {len(prompt_pairs)} interpolation segments")
        print(f"🔄 Including loop back: '{prompts[-1]}' → '{prompts[0]}'")
        print(f"📊 {self.batch_size} frames per segment = {len(prompt_pairs) * self.batch_size} total frames")
        
        # Clear output directory if it exists
        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                # Clear directory contents
                for filename in os.listdir(self.output_dir):
                    file_path = os.path.join(self.output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}: {e}")
                print(f"🗑️ Cleared output directory: {self.output_dir}")
            except Exception as e:
                print(f"❌ Error preparing output directory: {e}")
        
        # Generate each interpolation segment with progress bar
        self.total_frames_saved = 0
        all_frames = []
        
        # Use a shared latent cookie for the entire sequence to maintain composition
        if self.latent_cookie is not None:
            latent_cookie = self.latent_cookie
            print(f"🍪 Using user-specified latent cookie {latent_cookie} for consistent composition across sequence")
        else:
            import random
            latent_cookie = random.randint(1, 1000000)
            print(f"🍪 Generated latent cookie {latent_cookie} for consistent composition across sequence")
        
        # Use a shared seed for the entire sequence to maintain consistency
        if self.seed is not None:
            shared_seed = self.seed
            print(f"🎲 Using user-specified seed {shared_seed} for consistent generation across all segments")
        else:
            shared_seed = random.randint(0, 2**32 - 1)
            print(f"🎲 Generated shared seed {shared_seed} for consistent generation across all segments")
        
        with tqdm(total=len(prompt_pairs), desc="Generating interpolations", unit="segment") as pbar:
            for i, (prompt1, prompt2) in enumerate(prompt_pairs):
                segment_start_time = time.time()
                
                # Update progress bar description
                pbar.set_description(f"Segment {i+1}/{len(prompt_pairs)}: '{prompt1[:20]}...' → '{prompt2[:20]}...'")
                
                try:
                    # Generate interpolated embeddings for this segment
                    request_id = f"multi_prompt_seg_{i:03d}"
                    frames = self.img_gen.generate_interpolated_embeddings(
                        prompt1=prompt1,
                        prompt2=prompt2,
                        batch_size=self.batch_size,
                        request_id=request_id,
                        latent_cookie=latent_cookie,
                        seed=shared_seed,
                        **self.generation_params
                    )
                    
                    if frames:
                        all_frames.extend(frames)
                        
                        # Save frames with consecutive numbering
                        frames_saved = self.save_images_to_directory_with_consecutive_numbering(self.total_frames_saved)
                        self.total_frames_saved += frames_saved
                        
                        segment_duration = time.time() - segment_start_time
                        pbar.set_postfix({
                            'frames': f"{len(frames)}",
                            'time': f"{segment_duration:.1f}s",
                            'rate': f"{len(frames)/segment_duration:.1f} img/s",
                            'total_frames': self.total_frames_saved
                        })
                        
                    else:
                        print(f"❌ No frames generated for segment {i+1}")
                        
                except Exception as e:
                    print(f"❌ Error generating segment {i+1}: {e}")
                
                pbar.update(1)
        
        # Update the current frames to the last generated batch for display
        if all_frames:
            print(f"\n✅ Multi-prompt sequence complete!")
            print(f"📊 Generated {self.total_frames_saved} total frames across {len(prompt_pairs)} segments")
            print(f"⏱️ Average: {self.batch_size} frames per segment")
            
            if self.output_dir:
                print(f"💾 All frames saved to: {self.output_dir}")
                print(f"🎬 Ready for video creation: frame_000000.{self.output_format} to frame_{self.total_frames_saved-1:06d}.{self.output_format}")
            
            # Set the last batch as current frames for immediate display
            last_batch_start = len(all_frames) - self.batch_size
            if last_batch_start >= 0:
                self.img_gen.current_frames = all_frames[last_batch_start:]
                self.img_gen.is_interpolated_sequence = True
                self.img_gen._create_randomized_order()
            
            return True
        else:
            print(f"❌ Multi-prompt sequence failed - no frames generated")
            return False
    
    def generate_initial_batch(self):
        """Generate the initial animation batch."""
        if self.is_multi_prompt_mode:
            print(f"🌈 Generating multi-prompt interpolation sequence ({len(self.prompt_list)} prompts)...")
            return self.generate_multi_prompt_sequence()
        elif self.interpolation_mode and self.prompt2:
            print(f"🌈 Generating initial interpolated animation batch ({self.batch_size} frames)...")
            print(f"   '{self.initial_prompt}' → '{self.prompt2}'")
        else:
            print(f"🎬 Generating initial animation batch ({self.batch_size} frames)...")
        
        def generate_initial():
            start_time = time.time()
            print(f"⏰ Initial batch start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
            
            if self.interpolation_mode and self.prompt2:
                # Use interpolated embeddings generation
                self.img_gen.generate_interpolated_embeddings(
                    prompt1=self.initial_prompt,
                    prompt2=self.prompt2,
                    batch_size=self.batch_size, 
                    request_id="initial", 
                    **self.generation_params
                )
            else:
                # Use regular bifurcated wiggle generation
                self.img_gen.generate_animation_batch(
                    batch_size=self.batch_size, 
                    request_id="initial", 
                    **self.generation_params
                )
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Initial batch generation time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
            print(f"📊 Initial batch rate: {self.batch_size/duration:.1f} images/second")
        
        # For multi-prompt mode, we've already generated everything
        if self.is_multi_prompt_mode:
            return True
        
        # Start generation in background
        gen_thread = threading.Thread(target=generate_initial)
        gen_thread.start()
        
        # Show loading screen
        loading_frame = 0
        while gen_thread.is_alive():
            self.window.fill((20, 20, 30))
            loading_text = f"Generating animation batch... {'.' * (loading_frame % 4)}"
            text_surface = self.font.render(loading_text, True, (255, 255, 255))
            self.window.blit(text_surface, (10, self.window_height // 2))
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
        
        # Set optimal rhythm for the animation type
        if self.interpolation_mode and self.prompt2:
            # Use continuous linear rhythm for smooth interpolated embeddings ping-pong
            self.img_gen.animation_controller.set_rhythm_modulator(ContinuousLinearRhythm(speed=1.2))
            print("🎵 Set Continuous Linear rhythm for interpolated embeddings ping-pong animation")
        
        # Save images to directory if specified
        self.save_images_to_directory()
        
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
                    
                    # Save images to directory if specified
                    self.save_images_to_directory()
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
            
        elif key == pygame.K_5:
            self.img_gen.animation_controller.set_rhythm_modulator(LinearRhythm(interval=1.2))
            
        elif key == pygame.K_6:
            self.img_gen.animation_controller.set_rhythm_modulator(ContinuousLinearRhythm(speed=2.5))
    
    def get_status(self):
        """Get current status information from the animation system."""
        return self.img_gen.get_status()
    
    def generate_interpolated_animation(self, prompt1: str, prompt2: str):
        """Generate an animated interpolation between two prompts."""
        print(f"🎬 Generating interpolated animation: '{prompt1}' → '{prompt2}'")

        # Generate interpolated embeddings
        interpolated_embeddings = self.img_gen.generate_interpolated_embeddings(
            prompt1, prompt2, self.batch_size
        )

        # Generate images for each interpolated embedding
        images = []
        for idx, embedding in enumerate(interpolated_embeddings):
            print(f"🖼️ Generating image {idx + 1}/{len(interpolated_embeddings)}")
            image = self.img_gen.generate_image_from_embedding(embedding)
            images.append(image)

        print(f"✅ Interpolated animation generated successfully")
        return images
    
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
                self.show_image(current_frame, self.window, self.window_width, self.window_height)
            
            # Prepare UI status
            status = self.img_gen.get_status()
            rhythm_name = status['rhythm_type'].replace('Rhythm', '')
            interpolation_status = "ON" if status['interpolation_enabled'] else "OFF"
            
            status_lines = [
                f"Effects: {', '.join(self.current_effects) if self.current_effects else 'None'}",
                f"Animation: {'ON' if self.animation_enabled else 'OFF'} ({self.animation_speed} FPS)",
                f"Rhythm: {rhythm_name} | Interpolation: {interpolation_status}",
                f"Format: {self.output_format} | Noise: {self.noise_magnitude} | Bifurcation: {self.bifurcation_step}"
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
    
    # Ask user if they want to maximize the window
    maximize_input = input("Maximize window to fill screen? (y/n, default=y): ").strip().lower()
    maximize_window = maximize_input != 'n'
    
    # Ask user if they want to save images to a directory
    output_dir_input = input("Save images to directory? (leave empty to skip): ").strip()
    output_dir = output_dir_input if output_dir_input else None
    
    navigator = DreamspaceNavigator(
        server_url=server_url,
        image_size=image_size,
        batch_size=2,  # Set default back to 2 for interactive mode
        initial_prompt="strange bright forest land, steampunk trees",
        noise_magnitude=0.27,  # Updated default noise magnitude
        bifurcation_step=3,
        output_format="png",  # Use PNG default for better quality
        maximize_window=maximize_window,
        output_dir=output_dir
    )
    
    navigator.run()


def main_with_args():
    """Main function using command line arguments."""
    args = parse_arguments()
    
    # Check for interpolation mode
    interpolation_mode = getattr(args, 'interpolation_mode', False) or getattr(args, 'prompt2', None) is not None or getattr(args, 'prompt_list', None) is not None
    prompt2 = getattr(args, 'prompt2', None)
    prompt_list = getattr(args, 'prompt_list', None)
    
    if interpolation_mode and not prompt2 and not prompt_list:
        print("❌ Interpolation mode requires --prompt2 or --prompt-list to be specified")
        return
    
    if prompt2 and prompt_list:
        print("❌ Cannot specify both --prompt2 and --prompt-list. Choose one interpolation mode.")
        return
    
    # Get configuration from args
    server_url = args.server
    image_size = get_image_dimensions(args)
    batch_size = args.batch_size  # Use the CLI argument instead of hardcoding
    initial_prompt = args.prompt
    
    # Bifurcated wiggle is now the default method (ignored in interpolation mode)
    bifurcation_step = 3
    
    # Get output format if available, default to png for better quality
    output_format = getattr(args, 'output_format', 'png')
    
    # Check if maximize option is available
    maximize_window = getattr(args, 'maximize', False)
    
    # Get output directory if specified
    output_dir = getattr(args, 'output_dir', None)
    
    navigator = DreamspaceNavigator(
        server_url=server_url,
        image_size=image_size,
        batch_size=batch_size,
        initial_prompt=initial_prompt,
        noise_magnitude=args.noise_magnitude,
        bifurcation_step=bifurcation_step,
        output_format=output_format,
        maximize_window=maximize_window,
        interpolation_mode=interpolation_mode,
        prompt2=prompt2,
        output_dir=output_dir,
        prompt_list=prompt_list,
        latent_cookie=getattr(args, 'latent_cookie', None),
        seed=getattr(args, 'seed', None)
    )
    
    # Set initial configuration
    if args.no_interpolation:
        navigator.img_gen.animation_controller.interpolation_enabled = False
    
    navigator.animation_speed = args.fps
    
    if interpolation_mode:
        if prompt_list:
            print(f"🌈 Multi-Prompt Interpolation Mode Enabled")
            print(f"🔢 Prompts: {prompt_list}")
            print(f"📊 Interpolation Steps per segment: {batch_size}")
            print(f"🔄 Total segments (with loop): {len(prompt_list)}")
            print(f"🎬 Total frames: {len(prompt_list) * batch_size}")
        else:
            print(f"🌈 Two-Prompt Interpolation Mode Enabled")
            print(f"🔢 Prompt 1: {initial_prompt}")
            print(f"🔢 Prompt 2: {prompt2}")
            print(f"📊 Interpolation Steps: {batch_size}")
    else:
        # Bifurcated wiggle is always enabled (default method)
        navigator.latent_wiggle = True
        print(f"✨ Bifurcated Latent Wiggle Pipeline Enabled (default method)")
        print(f"🔧 Noise Magnitude: {navigator.noise_magnitude}")
        print(f"🔀 Bifurcation Step: {navigator.bifurcation_step}")
    
    print(f"📄 Output Format: {navigator.output_format}")
    
    if navigator.output_dir:
        print(f"📁 Images will be saved to: {navigator.output_dir}")
    
    navigator.run()


if __name__ == "__main__":
    import sys
    
    # Check if we have command line arguments (other than script name)
    if len(sys.argv) > 1:
        main_with_args()
    else:
        main()
