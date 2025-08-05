"""Command line interface for the animated dreamspace navigator.

Handles argument parsing and configuration setup for the application.
"""

import argparse
from typing import Tuple


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Animated Dreamspace Navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  ← → ↑ ↓ : Navigate through parameter space (generates new animation loops)
  Space   : Add random effects to the prompt
  R       : Reset effects
  A       : Toggle animation on/off
  F       : Cycle animation speed
  S       : Save current frame
  I       : Toggle image interpolation
  X       : Shuffle frame order
  1-4     : Switch rhythm patterns (heartbeat slow/fast, breathing, waves)
  Escape  : Exit

Examples:
  python animated_navigation.py --size 512 --batch-size 8
  python animated_navigation.py --width 1024 --height 768 --server http://localhost:8001
        """
    )
    
    # Image size options
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--size", type=int, 
        help="Square image size (e.g., 512, 768, 1024)"
    )
    size_group.add_argument(
        "--width", type=int, default=768, 
        help="Image width (default: 768)"
    )
    parser.add_argument(
        "--height", type=int, default=768, 
        help="Image height (default: 768)"
    )
    
    # Server configuration
    parser.add_argument(
        "--server", type=str, default="http://172.28.5.21:8001",
        help="Server URL (default: http://172.28.5.21:8001)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Animation batch size (default: 16)"
    )
    
    parser.add_argument(
        "--prompt", type=str, default="strange bright forest land, steampunk trees",
        help="Initial prompt for image generation"
    )
    
    # Animation settings
    parser.add_argument(
        "--fps", type=int, default=8, choices=[4, 8, 12, 16, 24],
        help="Animation FPS (default: 8)"
    )
    
    parser.add_argument(
        "--no-interpolation", action="store_true",
        help="Disable image interpolation"
    )
    
    # Interactive mode
    parser.add_argument(
        "--interactive", action="store_true",
        help="Use interactive prompts for configuration"
    )
    
    # Latent wiggle settings
    parser.add_argument(
        "--latent-wiggle", action="store_true",
        help="Enable latent wiggle pipeline (skips img2img pipeline)"
    )

    parser.add_argument(
        "--noise-magnitude", type=float, default=0.05,
        help="Magnitude of noise for latent wiggle variations (default: 0.05)"
    )
    
    return parser.parse_args()


def get_image_dimensions(args: argparse.Namespace) -> Tuple[int, int]:
    """Get image dimensions from arguments."""
    if args.size:
        return args.size, args.size
    else:
        return args.width, args.height


def get_interactive_config() -> Tuple[str, Tuple[int, int]]:
    """Get configuration through interactive prompts."""
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
    
    return server_url, (image_width, image_height)


def print_welcome_message(server_url: str, image_size: Tuple[int, int], batch_size: int):
    """Print welcome message with configuration details."""
    width, height = image_size
    print(f"🖼️ Using image size: {width}x{height}")
    print(f"🌐 Server: {server_url}")
    print(f"📏 Batch size: {batch_size}")
    
    print("\\n🎮 Controls:")
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
