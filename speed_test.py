#!/usr/bin/env python3
"""Dreamspace Speed Test Utility

A minimalist CLI utility for testing image generation performance witho        print(f"        print(f"🔮 Connecting to server: {server_url}")
        print(f"📏 Image size: {self.image_width}x{self.image_height}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🎚️ Noise magnitude: {self.noise_magnitude}")
        print(f"🔀 Bifurcation step: {self.bifurcation_step}")
        print(f"📄 Output format: {self.output_format}")
        print(f"💭 Prompt: '{prompt}'")
        print()ge size: {self.image_width}x{self.image_height}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔧 Noise magnitude: {self.noise_magnitude}")
        print(f"🔀 Bifurcation step: {self.bifurcation_step}")
        print(f"💭 Prompt: '{prompt}'")isual output.
Uses the same argument parsing as animated_navigation.py but runs headless.

Usage:
    # Basic speed test
    PYTHONPATH=src python speed_test.py --server http://localhost:8001
    
    # Custom configuration
    PYTHONPATH=src python speed_test.py --server http://localhost:8001 --size 512 --batch-size 8 --prompt "forest landscape"
    
    # Multiple test rounds
    PYTHONPATH=src python speed_test.py --server http://localhost:8001 --rounds 3
"""

import sys
import time
import argparse
from typing import Tuple

# Import from the main dreamspace library
from dreamspace.core.remote_generator import AnimatedRemoteImgGen
from dreamspace.cli.navigation import parse_arguments, get_image_dimensions

DEFAULT_IP = "172.28.5.21"


def create_speed_test_parser():
    """Create argument parser with speed test specific options."""
    parser = argparse.ArgumentParser(
        description="Dreamspace Speed Test Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic speed test
  speed_test.py --server http://localhost:8001
  
  # Custom batch size and image size
  speed_test.py --server http://localhost:8001 --size 512 --batch-size 16
  
  # Multiple test rounds for averaging
  speed_test.py --server http://localhost:8001 --rounds 3
        """
    )
    
    # Server configuration
    parser.add_argument(
        "--server", type=str, default=f"http://{DEFAULT_IP}:8001",
        help=f"Server URL (default: http://{DEFAULT_IP}:8001)"
    )
    
    # Image dimensions
    parser.add_argument(
        "--size", type=int, choices=[256, 512, 768, 1024, 2048], default=2048,
        help="Image size (square) - choose from 256, 512, 768, 1024, 2048 (default: 2048)"
    )
    
    parser.add_argument(
        "--width", type=int, default=2048, help="Image width (overrides --size)"
    )
    
    parser.add_argument(
        "--height", type=int, default=1280, help="Image height (overrides --size)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Animation batch size (default: 16)"
    )
    
    parser.add_argument(
        "--prompt", type=str, default="steampunk forest with glass and brass",
        help="Initial prompt for image generation"
    )
    
    # Model selection
    parser.add_argument(
        "--model", type=str, default="sd15_server",
        choices=["sd15_server", "sd21_server", "kandinsky21_server"],
        help="Model to use for generation (default: sd15_server)"
    )
    
    # Interpolated embeddings mode
    parser.add_argument(
        "--prompt2", type=str, 
        help="Second prompt for interpolated embeddings mode (enables interpolation between prompt and prompt2)"
    )
    
    parser.add_argument(
        "--interpolation-mode", action="store_true",
        help="Enable interpolated embeddings mode (requires --prompt2)"
    )
    
    # Generation options (bifurcated wiggle is now the default method)
    parser.add_argument(
        "--noise-magnitude", type=float, default=0.3,
        help="Magnitude of noise for latent wiggle variations (default: 0.3)"
    )

    parser.add_argument(
        "--bifurcation-step", type=int, default=3,
        help="Number of steps from end to bifurcate in bifurcated wiggle (default: 3, set to 0 for original wiggle)"
    )
    
    parser.add_argument(
        "--output-format", type=str, default="jpeg", 
        choices=["jpeg", "tensor", "png", "jpeg_optimized"],
        help="Output format: 'jpeg' (default), 'tensor' (fast local), 'png', or 'jpeg_optimized' (skip PIL)"
    )
    
    # Speed test specific options
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of test rounds to run for averaging (default: 1)"
    )
    
    parser.add_argument(
        "--warm-up", action="store_true",
        help="Run a warm-up batch before timing (recommended for accurate results)"
    )
    
    return parser


class SpeedTester:
    """Minimalist speed testing utility for dreamspace generation."""
    
    def __init__(self, server_url: str, prompt: str, image_size: Tuple[int, int] = (2048, 1280), batch_size: int = 2, noise_magnitude: float = 0.17, bifurcation_step: int = 3, output_format: str = "png", interpolation_mode: bool = False, prompt2: str = None, model: str = "sd15_server"):
        self.server_url = server_url
        self.image_width, self.image_height = image_size
        self.batch_size = batch_size
        self.prompt = prompt
        self.noise_magnitude = noise_magnitude
        self.bifurcation_step = bifurcation_step
        self.output_format = output_format
        self.interpolation_mode = interpolation_mode
        self.prompt2 = prompt2
        self.output_format = output_format
        self.model = model
        
        # Generation parameters
        self.generation_params = {
            "width": self.image_width,
            "height": self.image_height,
            "noise_magnitude": self.noise_magnitude,
            "bifurcation_step": self.bifurcation_step,
            "output_format": self.output_format
        }
        
        print(f"🔮 Connecting to server: {server_url}")
        print(f"📏 Image size: {self.image_width}x{self.image_height}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🎯 Model: {model}")
        print(f"🔧 Noise magnitude: {self.noise_magnitude}")
        print(f"🔀 Bifurcation step: {self.bifurcation_step}")
        print(f"📄 Output format: {self.output_format}")
        print(f"💭 Prompt: '{prompt}'")
        print()
        
        # Initialize image generator
        try:
            self.img_gen = AnimatedRemoteImgGen(server_url, prompt, model)
            print("✅ Connection established!")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            sys.exit(1)
    
    def run_warm_up(self):
        """Run a warm-up batch to initialize the server pipeline."""
        print("🔥 Running warm-up batch...")
        start_time = time.time()
        
        try:
            self.img_gen.generate_animation_batch(
                batch_size=self.batch_size,
                request_id="warmup",
                **self.generation_params
            )
            
            duration = time.time() - start_time
            print(f"✅ Warm-up completed in {duration:.2f} seconds")
            print(f"   (Warm-up results not counted in final statistics)")
            print()
            
        except Exception as e:
            print(f"❌ Warm-up failed: {e}")
            print("⚠️ Continuing with main test anyway...")
            print()
    
    def run_speed_test(self, round_num: int = 1):
        """Run a single speed test round."""
        if self.interpolation_mode and self.prompt2:
            print(f"🌈 Running interpolated embeddings speed test round {round_num}...")
            print(f"   '{self.prompt}' → '{self.prompt2}'")
        else:
            print(f"🏃 Running speed test round {round_num}...")
        
        start_time = time.time()
        request_id = f"test_round_{round_num}_{start_time:.3f}"
        
        print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        try:
            if self.interpolation_mode and self.prompt2:
                # Use interpolated embeddings generation
                frames = self.img_gen.generate_interpolated_embeddings(
                    prompt1=self.prompt,
                    prompt2=self.prompt2,
                    batch_size=self.batch_size,
                    request_id=request_id,
                    **self.generation_params
                )
            else:
                # Use regular bifurcated wiggle generation
                frames = self.img_gen.generate_animation_batch(
                    prompt=self.prompt,
                    batch_size=self.batch_size,
                    request_id=request_id,
                    **self.generation_params
                )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if frames:
                images_per_second = self.batch_size / duration
                print(f"✅ Round {round_num} completed successfully!")
                print(f"⏱️ Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
                print(f"📊 Rate: {images_per_second:.2f} images/second")
                print(f"🖼️ Generated: {len(frames)} images")
                if self.interpolation_mode:
                    print(f"🌈 Mode: Interpolated embeddings")
                else:
                    print(f"🔀 Mode: Bifurcated wiggle")
                print()
                
                return {
                    'duration': duration,
                    'images_per_second': images_per_second,
                    'images_generated': len(frames),
                    'success': True
                }
            else:
                print(f"❌ Round {round_num} failed: No images returned")
                print()
                return {'success': False}
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Round {round_num} failed after {duration:.2f}s: {e}")
            print()
            return {'success': False, 'duration': duration}
    
    def run_multiple_rounds(self, num_rounds: int, warm_up: bool = False):
        """Run multiple speed test rounds and calculate statistics."""
        if warm_up:
            self.run_warm_up()
        
        print(f"🎯 Starting {num_rounds} speed test round{'s' if num_rounds > 1 else ''}...")
        print("=" * 50)
        print()
        
        results = []
        
        for round_num in range(1, num_rounds + 1):
            result = self.run_speed_test(round_num)
            if result['success']:
                results.append(result)
        
        if not results:
            print("❌ All test rounds failed!")
            return
        
        # Calculate statistics
        print("📈 SPEED TEST RESULTS")
        print("=" * 50)
        
        durations = [r['duration'] for r in results]
        rates = [r['images_per_second'] for r in results]
        
        avg_duration = sum(durations) / len(durations)
        avg_rate = sum(rates) / len(rates)
        min_duration = min(durations)
        max_duration = max(durations)
        max_rate = max(rates)
        min_rate = min(rates)
        
        print(f"✅ Successful rounds: {len(results)}/{num_rounds}")
        print(f"📦 Batch size: {self.batch_size} images")
        print(f"📏 Image size: {self.image_width}x{self.image_height}")
        print()
        print("⏱️ TIMING RESULTS:")
        print(f"   Average: {avg_duration:.2f} seconds ({avg_duration/60:.1f} minutes)")
        print(f"   Fastest: {min_duration:.2f} seconds")
        print(f"   Slowest: {max_duration:.2f} seconds")
        print()
        print("📊 GENERATION RATE:")
        print(f"   Average: {avg_rate:.2f} images/second")
        print(f"   Peak:    {max_rate:.2f} images/second")
        print(f"   Lowest:  {min_rate:.2f} images/second")
        print()
        print(f"🎯 Total images generated: {len(results) * self.batch_size}")
        print(f"🕐 Total test time: {sum(durations):.1f} seconds ({sum(durations)/60:.1f} minutes)")


def main():
    """Main function for speed testing."""
    parser = create_speed_test_parser()
    args = parser.parse_args()
    
    # Check for interpolation mode
    interpolation_mode = getattr(args, 'interpolation_mode', False) or getattr(args, 'prompt2', None) is not None
    prompt2 = getattr(args, 'prompt2', None)
    
    if interpolation_mode and not prompt2:
        print("❌ Interpolation mode requires --prompt2 to be specified")
        return
    
    # Get image dimensions
    if args.width and args.height:
        image_size = (args.width, args.height)
    else:
        size = args.size
        image_size = (size, size)
    
    # Bifurcated wiggle is now the default method (ignored in interpolation mode)
    bifurcation_step = 3

    # Create speed tester
    tester = SpeedTester(
        server_url=args.server,
        image_size=image_size,
        batch_size=2,
        prompt=args.prompt,
        noise_magnitude=args.noise_magnitude,
        bifurcation_step=bifurcation_step,
        output_format=args.output_format,
        interpolation_mode=interpolation_mode,
        prompt2=prompt2,
        model=getattr(args, 'model', 'sd15_server')
    )
    
    # Print test configuration
    if interpolation_mode:
        print(f"🌈 Interpolated Embeddings Speed Test")
        print(f"🔢 Prompt 1: {args.prompt}")
        print(f"🔢 Prompt 2: {prompt2}")
        print(f"📊 Interpolation Steps: {tester.batch_size}")
    else:
        print(f"✨ Bifurcated Latent Wiggle Speed Test")
        print(f"🔧 Noise Magnitude: {tester.noise_magnitude}")
        print(f"🔀 Bifurcation Step: {tester.bifurcation_step}")
    
    # Run speed test
    tester.run_multiple_rounds(
        num_rounds=args.rounds,
        warm_up=args.warm_up
    )
    
    print("🏁 Speed test completed!")


if __name__ == "__main__":
    main()
