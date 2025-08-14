"""Shared display and animation utilities for CLI tools.

This module provides reusable components for pygame-based image display,
window management, and animation timing that can be shared across different
CLI tools in the dreamspace project.
"""

import glob
import os
import time
from typing import List, Optional, Tuple, Union

from enum import Enum
from pathlib import Path
import pygame
from PIL import Image
from screeninfo import get_monitors


class LoopMode(Enum):
    """Animation loop modes."""

    PING_PONG = "ping-pong"  # Forward then backward: 0,1,2,3,2,1,0,1,2,3...
    FORWARD = "forward"  # Forward with wrap: 0,1,2,3,0,1,2,3...


class ImageDisplay:
    """Pygame-based image display with fullscreen and aspect ratio support."""

    def __init__(self, maximize: bool = False, window_title: str = "Image Display"):
        """Initialize the display.

        Args:
            maximize: Whether to use fullscreen mode with aspect ratio preservation
            window_title: Title for the pygame window
        """
        self.maximize = maximize
        self.window_title = window_title

        # Initialize pygame
        pygame.init()
        pygame.font.init()

        # Calculate window dimensions
        if self.maximize:
            self.window_width, self.window_height = self._get_screen_resolution()
            print(f"🖥️ Fullscreen mode: {self.window_width}x{self.window_height}")
        else:
            # Will be set based on first image
            self.window_width = 800
            self.window_height = 600

        # Create window (will be resized if not maximized)
        flags = pygame.FULLSCREEN if self.maximize else 0
        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height), flags
        )
        pygame.display.set_caption(self.window_title)

        # Initialize font for UI text
        self.font = pygame.font.Font(None, 24)

        print(f"✅ Display initialized: {self.window_width}x{self.window_height}")

    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Get the screen resolution using screeninfo with pygame fallback."""
        try:
            # Use screeninfo to get actual screen resolution
            monitor = get_monitors()[0]  # Primary monitor
            return monitor.width, monitor.height
        except Exception as e:
            print(f"❌ Failed to get resolution with screeninfo: {e}")
            # Fallback to pygame detection
            info = pygame.display.Info()
            return info.current_w, info.current_h

    def set_window_size_from_image(self, image: Image.Image):
        """Set window size based on image if not maximized."""
        if not self.maximize:
            self.window_width, self.window_height = image.size
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            print(
                f"🪟 Window resized to image size: {self.window_width}x{self.window_height}"
            )

    def show_image(self, image: Image.Image):
        """Display PIL Image in pygame window with aspect ratio preservation.

        This method reuses the exact scaling logic from animated_navigation.py
        to ensure consistent behavior between tools.
        """
        if image is None:
            return

        # Calculate scaling to fit screen while maintaining aspect ratio
        img_width, img_height = image.size

        # Calculate scale factors for both dimensions
        scale_x = self.window_width / img_width
        scale_y = self.window_height / img_height

        # Use the SMALLER scale factor to ensure image fits completely
        # (touches 2 opposite sides, black borders on other 2)
        scale = min(scale_x, scale_y)

        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize image to calculated dimensions
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center the image in the window
        x_offset = (self.window_width - new_width) // 2
        y_offset = (self.window_height - new_height) // 2

        # Clear window with black background
        self.window.fill((0, 0, 0))

        # Convert PIL image to pygame surface
        mode = resized_image.mode
        size = resized_image.size
        data = resized_image.tobytes()
        py_img = pygame.image.fromstring(data, size, mode)

        # Blit the image to the window
        self.window.blit(py_img, (x_offset, y_offset))

    def draw_text_overlay(
        self,
        text: str,
        position: str = "bottom-left",
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """Draw text overlay on the display.

        Args:
            text: Text to display
            position: Where to position text ("bottom-left", "top-left", "bottom-right", "top-right")
            color: RGB color tuple for the text
        """
        text_surface = self.font.render(text, True, color)
        text_width, text_height = text_surface.get_size()

        # Calculate position based on string
        margin = 10
        if position == "bottom-left":
            x, y = margin, self.window_height - text_height - margin
        elif position == "top-left":
            x, y = margin, margin
        elif position == "bottom-right":
            x, y = (
                self.window_width - text_width - margin,
                self.window_height - text_height - margin,
            )
        elif position == "top-right":
            x, y = self.window_width - text_width - margin, margin
        else:
            x, y = (
                margin,
                self.window_height - text_height - margin,
            )  # Default to bottom-left

        # Draw semi-transparent background
        background_rect = pygame.Rect(x - 5, y - 2, text_width + 10, text_height + 4)
        overlay = pygame.Surface((background_rect.width, background_rect.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.window.blit(overlay, background_rect)

        # Draw text
        self.window.blit(text_surface, (x, y))

    def update(self):
        """Update the display."""
        pygame.display.flip()

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()


class FrameAnimator:
    """Handles frame sequencing and timing for animation playback."""

    def __init__(
        self,
        frames: List[Image.Image],
        fps: int = 24,
        loop_mode: LoopMode = LoopMode.PING_PONG,
    ):
        """Initialize the frame animator.

        Args:
            frames: List of PIL Image frames to animate
            fps: Frames per second for animation timing
            loop_mode: How to loop through frames (ping-pong or forward)
        """
        self.frames = frames
        self.fps = fps
        self.loop_mode = loop_mode
        self.frame_interval = 1.0 / fps  # Time between frames in seconds

        # Animation state
        self.current_index = 0
        self.direction = 1  # 1 for forward, -1 for backward (ping-pong mode)
        self.last_frame_time = time.time()

        # Create frame sequence based on loop mode
        self._setup_frame_sequence()

        print(f"🎬 Frame animator initialized:")
        print(f"   Frames: {len(self.frames)}")
        print(f"   FPS: {self.fps}")
        print(f"   Loop mode: {self.loop_mode.value}")
        print(f"   Sequence length: {len(self.frame_sequence)}")

    def _setup_frame_sequence(self):
        """Set up the frame sequence based on loop mode."""
        if len(self.frames) == 0:
            self.frame_sequence = []
            return

        if self.loop_mode == LoopMode.FORWARD:
            # Simple forward loop: 0,1,2,3,0,1,2,3...
            self.frame_sequence = list(range(len(self.frames)))
        elif self.loop_mode == LoopMode.PING_PONG:
            # Ping-pong pattern: 0,1,2,3,2,1,0,1,2,3,2,1...
            if len(self.frames) == 1:
                self.frame_sequence = [0]
            else:
                forward = list(range(len(self.frames)))
                backward = list(
                    range(len(self.frames) - 2, 0, -1)
                )  # Exclude endpoints to avoid duplication
                self.frame_sequence = forward + backward

    def get_current_frame(self) -> Optional[Image.Image]:
        """Get the current frame, advancing if enough time has passed."""
        if not self.frames:
            return None

        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_interval:
            self._advance_frame()
            self.last_frame_time = current_time

        # Get current frame from sequence
        sequence_index = self.current_index % len(self.frame_sequence)
        frame_index = self.frame_sequence[sequence_index]
        return self.frames[frame_index]

    def _advance_frame(self):
        """Advance to the next frame in the sequence."""
        if not self.frame_sequence:
            return

        self.current_index = (self.current_index + 1) % len(self.frame_sequence)

    def get_frame_info(self) -> str:
        """Get current frame information as a string."""
        if not self.frames:
            return "No frames"

        sequence_index = self.current_index % len(self.frame_sequence)
        frame_index = self.frame_sequence[sequence_index]
        total_frames = len(self.frames)

        return f"Frame {frame_index + 1}/{total_frames} (seq {sequence_index + 1}/{len(self.frame_sequence)})"

    def set_fps(self, fps: int):
        """Change the animation FPS."""
        self.fps = max(1, fps)  # Ensure FPS is at least 1
        self.frame_interval = 1.0 / self.fps
        print(f"⚡ FPS changed to: {self.fps}")


def load_png_frames(directory: str) -> List[Image.Image]:
    """Load PNG files from directory, sorted by filename.

    Args:
        directory: Path to directory containing PNG files

    Returns:
        List of PIL Images sorted by filename

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no PNG files found
    """

    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all PNG files using glob
    png_pattern = os.path.join(directory, "*.png")
    png_files = glob.glob(png_pattern)

    if not png_files:
        raise ValueError(f"No PNG files found in directory: {directory}")

    # Sort files by filename for proper frame order
    png_files.sort()

    print(f"📁 Loading {len(png_files)} PNG files from {directory}")

    # Load images
    frames = []
    for i, filepath in enumerate(png_files):
        try:
            image = Image.open(filepath)
            # Convert to RGB if necessary (in case of RGBA PNGs)
            if image.mode == "RGBA":
                # Create white background for transparency
                background = Image.new("RGB", image.size, (0, 0, 0))
                background.paste(
                    image, mask=image.split()[-1]
                )  # Use alpha channel as mask
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            frames.append(
                image.copy()
            )  # Copy the image to avoid keeping the file handle open
            image.close()  # Close the file handle

            # Print progress for large batches
            if len(png_files) > 10 and (i + 1) % 10 == 0:
                print(f"   Loaded {i + 1}/{len(png_files)} frames...")

        except Exception as e:
            print(f"⚠️ Failed to load {filepath}: {e}")
            continue

    if not frames:
        raise ValueError(f"Failed to load any valid PNG files from {directory}")

    print(f"✅ Successfully loaded {len(frames)} frames")
    return frames
