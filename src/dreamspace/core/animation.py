"""Animation core functionality with rhythm modulators and frame interpolation.

This module provides the foundation for smooth, rhythm-based animations with
artistic modulation patterns that create organic, meditative visual experiences.
"""

import random
import math
import time
from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class RhythmModulator(ABC):
    """Base class for rhythm modulation patterns."""
    
    @abstractmethod
    def next_interval(self) -> float:
        """Return the time in seconds until the next transition."""
        pass


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


class AnimationController:
    """Controls frame animation with rhythm-based transitions and interpolation."""
    
    def __init__(self, rhythm_modulator: RhythmModulator = None):
        self.rhythm_modulator = rhythm_modulator or HeartbeatRhythm(base_bpm=35)
        self.last_transition_time = time.time()
        self.interpolation_enabled = True
        self._current_interval: Optional[float] = None
        
        print(f"🎵 Animation controller initialized with {self.rhythm_modulator.__class__.__name__}")
    
    def set_rhythm_modulator(self, modulator: RhythmModulator):
        """Change the rhythm modulation pattern."""
        self.rhythm_modulator = modulator
        self._current_interval = None  # Reset interval
        print(f"🎵 Rhythm changed to: {modulator.__class__.__name__}")
    
    def should_advance_frame(self) -> bool:
        """Check if it's time to advance to the next frame."""
        current_time = time.time()
        
        # Get the interval for current transition
        if self._current_interval is None:
            self._current_interval = self.rhythm_modulator.next_interval()
        
        # Check if it's time for the next transition
        if current_time - self.last_transition_time >= self._current_interval:
            self.last_transition_time = current_time
            self._current_interval = self.rhythm_modulator.next_interval()
            return True
        
        return False
    
    def get_transition_progress(self) -> float:
        """Get the current progress within the transition interval (0.0 to 1.0)."""
        if self._current_interval is None:
            return 0.0
        
        current_time = time.time()
        time_since_transition = current_time - self.last_transition_time
        return min(time_since_transition / self._current_interval, 1.0)
    
    def interpolate_frames(self, frame1: Image.Image, frame2: Image.Image, alpha: float) -> Image.Image:
        """Create a smooth blend between two frames."""
        if not frame1 or not frame2:
            return frame1 or frame2
        
        if not self.interpolation_enabled:
            return frame1
        
        # Ensure both images are the same size
        if frame1.size != frame2.size:
            frame2 = frame2.resize(frame1.size)
        
        # Use PIL's blend function for smooth interpolation
        return Image.blend(frame1, frame2, alpha)
    
    def smooth_progress(self, progress: float) -> float:
        """Apply smooth interpolation curve (ease-in-out)."""
        return 0.5 - 0.5 * math.cos(progress * math.pi)
    
    def toggle_interpolation(self) -> bool:
        """Toggle image interpolation on/off."""
        self.interpolation_enabled = not self.interpolation_enabled
        status = "enabled" if self.interpolation_enabled else "disabled"
        print(f"🎨 Interpolation {status}")
        return self.interpolation_enabled


class CrossBatchTransition:
    """Handles smooth transitions between animation batches."""
    
    def __init__(self, duration: float = 2.0):
        self.duration = duration
        self.active = False
        self.start_time: Optional[float] = None
        self.old_frame: Optional[Image.Image] = None
        self.new_frame: Optional[Image.Image] = None
    
    def start_transition(self, old_frame: Image.Image, new_frame: Image.Image):
        """Start a cross-batch transition."""
        self.old_frame = old_frame
        self.new_frame = new_frame
        self.start_time = time.time()
        self.active = True
        print(f"🎬 Starting {self.duration}s cross-batch transition")
    
    def get_current_frame(self, animation_controller: AnimationController) -> Optional[Image.Image]:
        """Get the current transition frame, or None if transition is complete."""
        if not self.active or self.start_time is None:
            return None
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        progress = min(elapsed / self.duration, 1.0)
        
        # Check if transition should end
        if progress >= 1.0:
            self.active = False
            self.old_frame = None
            self.new_frame = None
            print(f"🎭 Cross-batch transition completed")
            return self.new_frame
        
        # Apply smooth interpolation
        smooth_progress = animation_controller.smooth_progress(progress)
        return animation_controller.interpolate_frames(self.old_frame, self.new_frame, smooth_progress)
    
    def is_active(self) -> bool:
        """Check if cross-batch transition is currently active."""
        return self.active
