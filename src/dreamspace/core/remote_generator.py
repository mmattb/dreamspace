"""Remote image generation and batch management for animated sequences.

This module handles communication with the image generation server, manages
batches of animation frames, and provides smooth cross-batch transitions.
"""

import time
import random
import base64
import pickle
import gzip
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Optional, Dict, Any
from .animation import (
    RhythmModulator, HeartbeatRhythm, BreathingRhythm, WaveRhythm, 
    AnimationController, CrossBatchTransition
)


class AnimatedRemoteImgGen:
    """Remote image generator with batch animation support and artistic modulation."""
    
    def __init__(self, server_url: str, initial_prompt: str = "a surreal dreamlike forest"):
        self.server_url = server_url.rstrip('/')
        self.prompt = initial_prompt
        self.current_frames: List[Image.Image] = []
        self.frame_order: List[int] = []  # Randomized indices for frame display
        self.frame_index = 0
        self.is_generating = False
        self.current_request_id: Optional[str] = None
        self.cancel_current_request = False
        
        # Animation components
        self.animation_controller = AnimationController()
        self.cross_batch_transition = CrossBatchTransition(duration=2.0)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to the image generation server."""
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
    
    def _create_randomized_order(self):
        """Create a randomized order for frame display."""
        if not self.current_frames:
            return
        
        self.frame_order = list(range(len(self.current_frames)))
        random.shuffle(self.frame_order)
        self.frame_index = 0
        print(f"🎲 Randomized frame order: {len(self.frame_order)} frames")
    
    def generate_animation_batch(self, prompt: str = None, batch_size: int = 32, 
                               request_id: str = None, **kwargs) -> List[Image.Image]:
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
            "num_inference_steps": kwargs.get("num_inference_steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "seed": kwargs.get("seed", 42),
            "noise_magnitude": kwargs.get("noise_magnitude", 0.3),
            "bifurcation_step": kwargs.get("bifurcation_step", 3),
            "output_format": kwargs.get("output_format", "jpeg")
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
                timeout=300
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
            
            # Get output format from request data or metadata
            output_format = request_data.get("output_format", "jpeg")
            server_format = metadata.get("output_format", output_format)
            
            # Convert all images to PIL Images (handle different formats)
            frames = []
            
            if server_format == "tensor":
                # Special handling for tensor format - single tensor contains all images
                if len(result["images"]) > 0:
                    import torch
                    from io import BytesIO
                    
                    tensor_bytes = base64.b64decode(result["images"][0])  # Single tensor for whole batch
                    buffer = BytesIO(tensor_bytes)
                    tensor_batch = torch.load(buffer, map_location='cpu')
                    
                    # tensor_batch should be shape (B, C, H, W)
                    for i in range(tensor_batch.shape[0]):
                        # Check for cancellation during decoding
                        if self.cancel_current_request or self.current_request_id != request_id:
                            print(f"❌ Request {request_id[:8]} cancelled during tensor decoding at frame {i+1}")
                            return []
                        
                        # Extract individual image: (C, H, W) -> (H, W, C)
                        image_tensor = tensor_batch[i].permute(1, 2, 0)
                        image_array = image_tensor.numpy()
                        
                        # Convert to PIL Image (values should be in [0,1] range)
                        if image_array.dtype != np.uint8:
                            image_array = (image_array * 255).astype(np.uint8)
                        
                        image = Image.fromarray(image_array, mode='RGB')
                        frames.append(image)
            else:
                # Handle traditional base64-encoded images (jpeg, png, etc.)
                for i, image_data in enumerate(result["images"]):
                    # Check for cancellation during decoding
                    if self.cancel_current_request or self.current_request_id != request_id:
                        print(f"❌ Request {request_id[:8]} cancelled during decoding at frame {i+1}")
                        return []
                    
                    # Handle as base64-encoded image (jpeg, png, etc.)
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                    frames.append(image)
            
            # Final check before updating frames
            if self.cancel_current_request or self.current_request_id != request_id:
                print(f"❌ Request {request_id[:8]} cancelled before frame update")
                return []
            
            # Set up cross-batch transition if we have existing frames
            if self.current_frames and len(self.current_frames) > 0 and self.frame_order and len(self.frame_order) > 0:
                current_idx = self.frame_order[self.frame_index]
                old_frame = self.current_frames[current_idx]
                new_frame = frames[0] if frames else None  # Use first frame of new batch
                
                if old_frame and new_frame:
                    self.cross_batch_transition.start_transition(old_frame, new_frame)
                    print(f"🎭 Stored old batch last frame for smooth transition")
            
            # Update to new batch
            self.current_frames = frames
            self.frame_index = 0
            self._create_randomized_order()
            
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
    
    def get_current_frame(self) -> Optional[Image.Image]:
        """Get the current animation frame with optional interpolation."""
        if not self.current_frames or not self.frame_order:
            return None
        
        # Handle cross-batch transition FIRST - this takes priority
        transition_frame = self.cross_batch_transition.get_current_frame(self.animation_controller)
        if transition_frame is not None:
            return transition_frame
        
        # Normal frame transitions - advance frame if needed
        if self.animation_controller.should_advance_frame():
            self._advance_to_next_frame()
        
        # Get current and next frames for interpolation
        current_idx = self.frame_order[self.frame_index]
        next_idx = self.frame_order[(self.frame_index + 1) % len(self.frame_order)]
        
        current_frame = self.current_frames[current_idx]
        next_frame = self.current_frames[next_idx]
        
        if not self.animation_controller.interpolation_enabled:
            return current_frame
        
        # Calculate interpolation progress within the transition interval
        progress = self.animation_controller.get_transition_progress()
        smooth_progress = self.animation_controller.smooth_progress(progress)
        
        # Return interpolated frame with stronger blending for smoother transitions
        return self.animation_controller.interpolate_frames(current_frame, next_frame, smooth_progress * 0.7)
    
    def _advance_to_next_frame(self):
        """Advance to the next frame in the randomized sequence."""
        if self.frame_order:
            self.frame_index = (self.frame_index + 1) % len(self.frame_order)
    
    def advance_frame(self):
        """Manual frame advance (kept for compatibility but rhythm-based now)."""
        # This is now handled automatically by get_current_frame()
        pass
    
    def has_frames(self) -> bool:
        """Check if animation frames are available."""
        return len(self.current_frames) > 0
    
    def shuffle_frames(self):
        """Shuffle the frame order for variety."""
        if self.has_frames():
            self._create_randomized_order()
            print("🔀 Frame order reshuffled!")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status information."""
        return {
            "has_frames": self.has_frames(),
            "frame_count": len(self.current_frames),
            "current_frame_index": self.frame_index + 1 if self.has_frames() else 0,
            "is_generating": self.is_generating,
            "cross_batch_active": self.cross_batch_transition.is_active(),
            "interpolation_enabled": self.animation_controller.interpolation_enabled,
            "rhythm_type": self.animation_controller.rhythm_modulator.__class__.__name__
        }
