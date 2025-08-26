#!/usr/bin/env python3
"""PNG Animation CLI Utility

A simple command-line tool for displaying PNG image sequences as animations.
Automatically sorts images by filename (e.g., frame_000847.png, frame_000848.png)
and provides smooth playback with configurable FPS and loop modes.

Features:
- Automatic filename-based frame sorting
- Ping-pong animation (forward→backward→forward) or forward loop
- Fullscreen mode with aspect ratio preservation
- Configurable FPS
- ESC key to quit

Usage:
    # Basic usage - ping-pong animation at 24fps
    PYTHONPATH=src python examples/png_animator.py ./generated_frames/

    # Custom FPS
    PYTHONPATH=src python examples/png_animator.py ./frames/ --fps 30

    # Maximized window (full-screen with aspect ratio preservation)
    PYTHONPATH=src python examples/png_animator.py ./frames/ --maximize

    # Forward loop mode (wrap-around instead of ping-pong)
    PYTHONPATH=src python examples/png_animator.py ./frames/ --loop-mode forward

    # Combined options
    PYTHONPATH=src python examples/png_animator.py ./my_sequence/ --fps 16 --maximize --loop-mode forward

Controls:
    ESC: Exit the animation viewer
"""

import argparse
import sys
import time
import pygame
from pathlib import Path

# Import shared display utilities
try:
    from dreamspace.cli.display import (
        ImageDisplay,
        FrameAnimator,
        LoopMode,
        load_png_frames,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to run with PYTHONPATH=src")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PNG Animation CLI Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    # Required arguments
    parser.add_argument(
        "directory", type=str, help="Directory containing PNG files to animate"
    )

    # Animation options
    parser.add_argument(
        "--fps", type=int, default=24, help="Animation frames per second (default: 24)"
    )

    parser.add_argument(
        "--loop-mode",
        type=str,
        choices=["ping-pong", "forward"],
        default="ping-pong",
        help="Animation loop mode: 'ping-pong' (forward→backward→forward) or 'forward' (forward→wrap→forward) (default: ping-pong)",
    )

    # Display options
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize window to fill screen while maintaining aspect ratio",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # Check if directory exists
    if not Path(args.directory).exists():
        print(f"❌ Error: Directory does not exist: {args.directory}")
        return False

    if not Path(args.directory).is_dir():
        print(f"❌ Error: Path is not a directory: {args.directory}")
        return False

    # Validate FPS
    if args.fps < 1 or args.fps > 120:
        print("❌ Error: FPS must be between 1 and 120")
        return False

    return True


def main():
    """Main function."""
    print("🎬 PNG Animation CLI Utility")
    print("=" * 40)

    # Parse and validate arguments
    args = parse_arguments()

    if not validate_arguments(args):
        sys.exit(1)

    # Convert loop mode string to enum
    loop_mode = (
        LoopMode.PING_PONG if args.loop_mode == "ping-pong" else LoopMode.FORWARD
    )

    try:
        # Load PNG frames
        print(f"📁 Loading frames from: {args.directory}")
        frames = load_png_frames(args.directory)

        if not frames:
            print("❌ No frames loaded, exiting")
            sys.exit(1)

        # Initialize display
        display = ImageDisplay(
            maximize=args.maximize,
            window_title=f"PNG Animator - {Path(args.directory).name}",
        )

        # Set window size based on first frame if not maximized
        if not args.maximize:
            display.set_window_size_from_image(frames[0])

        # Initialize frame animator
        animator = FrameAnimator(frames, fps=args.fps, loop_mode=loop_mode)

        print(f"🎮 Controls: ESC to exit")
        print(f"🚀 Starting animation...")

        # Main animation loop
        clock = pygame.time.Clock()
        running = True
        frame_count = 0
        start_time = time.time()

        while running:
            # Handle events (returns False if should quit)
            running = display.handle_events()

            # Get current frame and display it
            current_frame = animator.get_current_frame()
            if current_frame:
                try:
                    display.show_image(current_frame)
                except OSError:
                    sys.stderr.write(f"Error rendering frame {frame_count}\n")
                    raise

                # Draw frame info overlay
                frame_info = animator.get_frame_info()
                elapsed_time = time.time() - start_time
                info_text = f"{frame_info} | FPS: {args.fps} | Mode: {args.loop_mode} | Time: {elapsed_time:.1f}s"

                display.draw_text_overlay(info_text, "bottom-left")

                # Update display
                display.update()

            # Control animation timing
            clock.tick(
                60
            )  # Run at 60 FPS for smooth display, animator controls frame timing
            frame_count += 1

        # Cleanup
        display.cleanup()

        # Print statistics
        total_time = time.time() - start_time
        print(f"\\n📊 Animation Statistics:")
        print(f"   Total frames: {len(frames)}")
        print(f"   Animation time: {total_time:.1f}s")
        print(f"   Display frames: {frame_count}")
        print(f"   Display FPS: {frame_count / total_time:.1f}")
        print("👋 Goodbye!")

    except KeyboardInterrupt:
        print("\\n🛑 Animation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
