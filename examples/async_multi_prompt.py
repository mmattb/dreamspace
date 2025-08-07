#!/usr/bin/env python3
"""Asynchronous Multi-Prompt Interpolation Generator

This script sends a multi-prompt interpolation request to the API server and exits immediately.
The server handles all the generation work in the background and saves PNG files directly to
the specified output directory.

Features:
- Async server-side generation (client exits immediately)
- Multi-prompt interpolation with looping
- Progressive PNG saving for monitoring progress
- Shared latent cookie for consistent composition
- Configurable generation parameters

Usage:
    # Basic multi-prompt sequence
    PYTHONPATH=src python examples/async_multi_prompt.py --prompts "forest scene" "desert landscape" "ocean view" --output-dir ./generated_sequence
    
    # With custom parameters
    PYTHONPATH=src python examples/async_multi_prompt.py \
        --prompts "mystical forest" "ancient desert" "cosmic ocean" "ethereal mountains" \
        --output-dir ./my_sequence \
        --batch-size 12 \
        --width 1024 \
        --height 1024 \
        --seed 42 \
        --latent-cookie 12345 \
        --model sd21_server
    
    # High resolution with more steps
    PYTHONPATH=src python examples/async_multi_prompt.py \
        --prompts "serene landscape" "vibrant cityscape" "peaceful waterfall" \
        --output-dir ./hd_sequence \
        --width 1024 \
        --height 768 \
        --num-inference-steps 75 \
        --guidance-scale 8.0 \
        --batch-size 16
"""

import argparse
import sys
import time
from typing import List

# Import the remote generator
from dreamspace.core.remote_generator import AnimatedRemoteImgGen


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Asynchronous Multi-Prompt Interpolation Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )
    
    # Required arguments
    parser.add_argument(
        "--prompts", type=str, nargs='+', required=True,
        help="List of prompts for multi-prompt interpolation sequence (minimum 2 prompts)"
    )
    
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where PNG files will be saved (will be created if it doesn't exist)"
    )
    
    # Server configuration
    parser.add_argument(
        "--server", type=str, default="http://172.28.5.21:8001",
        help="Server URL (default: http://172.28.5.21:8001)"
    )
    
    # Model selection
    parser.add_argument(
        "--model", type=str, default="sd15_server",
        choices=["sd15_server", "sd21_server", "kandinsky21_server"],
        help="Model to use for generation (default: sd15_server)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of interpolation steps per prompt segment (default: 8)"
    )
    
    parser.add_argument(
        "--width", type=int, default=768,
        help="Image width in pixels (default: 768)"
    )
    
    parser.add_argument(
        "--height", type=int, default=768,
        help="Image height in pixels (default: 768)"
    )
    
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5,
        help="Guidance scale (default: 7.5)"
    )
    
    parser.add_argument(
        "--num-inference-steps", type=int, default=50,
        help="Number of inference steps (default: 50)"
    )
    
    parser.add_argument(
        "--seed", type=int,
        help="Random seed for consistent generation (if not provided, server will generate one)"
    )
    
    parser.add_argument(
        "--latent-cookie", type=int,
        help="Integer cookie for shared latent across all segments (maintains consistent composition)"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    if len(args.prompts) < 2:
        print("❌ Error: At least 2 prompts are required for interpolation")
        return False
    
    if args.batch_size < 1 or args.batch_size > 32:
        print("❌ Error: Batch size must be between 1 and 32")
        return False
    
    if args.width < 256 or args.width > 2048 or args.width % 64 != 0:
        print("❌ Error: Width must be between 256-2048 and divisible by 64")
        return False
        
    if args.height < 256 or args.height > 2048 or args.height % 64 != 0:
        print("❌ Error: Height must be between 256-2048 and divisible by 64")
        return False
    
    if args.guidance_scale < 1.0 or args.guidance_scale > 20.0:
        print("❌ Error: Guidance scale must be between 1.0 and 20.0")
        return False
    
    if args.num_inference_steps < 10 or args.num_inference_steps > 150:
        print("❌ Error: Inference steps must be between 10 and 150")
        return False
    
    return True


def send_async_request(args: argparse.Namespace) -> bool:
    """Send the async multi-prompt request to the server using remote generator."""
    
    try:
        print(f"🚀 Connecting to server: {args.server}")
        
        # Initialize the remote generator
        generator = AnimatedRemoteImgGen(args.server, args.model)
        
        # Test connection
        if not generator.test_connection():
            print(f"❌ Failed to connect to server at {args.server}")
            print("💡 Check that the server is running and the URL is correct")
            return False
        
        print(f"✅ Connected successfully!")
        print(f"📝 Prompts: {args.prompts}")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🎯 Model: {args.model}")
        print(f"📊 Parameters: {args.batch_size} steps × {len(args.prompts)} segments = {args.batch_size * len(args.prompts)} total frames")
        print(f"🖼️ Resolution: {args.width}×{args.height}")
        
        if args.seed:
            print(f"🎲 Seed: {args.seed}")
        if args.latent_cookie:
            print(f"🍪 Latent cookie: {args.latent_cookie}")
        
        print(f"\n⏳ Sending async multi-prompt request...")
        
        # Prepare generation parameters
        generation_kwargs = {
            "width": args.width,
            "height": args.height,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "output_format": "png"
        }
        
        # Add optional parameters
        if args.seed is not None:
            generation_kwargs["seed"] = args.seed
        
        if args.latent_cookie is not None:
            generation_kwargs["latent_cookie"] = args.latent_cookie
        
        # Send the async request
        job_id = generator.async_multi_prompt_generation(
            prompts=args.prompts,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            **generation_kwargs
        )
        
        if job_id:
            print(f"✅ Request accepted successfully!")
            print(f"� Job ID: {job_id}")
            print(f"📊 Estimated total frames: {args.batch_size * len(args.prompts)}")
            print(f"\n🔄 The server is now generating images in the background.")
            print(f"📁 Monitor progress by checking files in: {args.output_dir}")
            print(f"🖼️ Images will be saved as: frame_000000.png, frame_000001.png, etc.")
            print(f"\n👋 Client finished successfully. Generation continues on server.")
            return True
        else:
            print(f"❌ Request failed - no job ID returned")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main function."""
    print("🌈 Async Multi-Prompt Interpolation Generator")
    print("=" * 50)
    
    # Parse and validate arguments
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    # Send the request
    success = send_async_request(args)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
