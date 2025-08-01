"""Keyboard-controlled image space navigation example.

This example demonstrates smooth navigation through image space using keyboard
controls. It uses img2img generation to maintain visual continuity between
frames, creating a dream-like exploration experience.

Controls:
- Arrow Keys: Navigate through parameter space
- Space: Add effects to the prompt
- Escape: Exit

The goal is to create smooth transitions where each keypress causes motion
in image space rather than just flipping through unrelated images.
"""

import sys
import os
import pygame
from PIL import Image

# Add the src directory to the path so we can import dreamspace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dreamspace import ImgGen, Config


def show_image(img: Image.Image, window):
    """Display PIL Image in pygame window."""
    img = img.resize((512, 512))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    py_img = pygame.image.fromstring(data, size, mode)
    window.blit(py_img, (0, 0))
    pygame.display.flip()


def main():
    """Main keyboard navigation loop."""
    # Initialize pygame
    pygame.init()
    win = pygame.display.set_mode((512, 512))
    pygame.display.set_caption("Dreamspace Navigator - Keyboard Control")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
    config = Config(config_path) if os.path.exists(config_path) else Config()
    
    # Initialize image generator with Kandinsky (better for smooth interpolation)
    print("🔮 Loading Kandinsky model...")
    try:
        img_gen = ImgGen(
            backend="kandinsky_local", 
            prompt="a surreal dreamlike forest, ethereal lighting",
            config=config
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install torch torchvision diffusers transformers accelerate")
        return
    
    # Navigation parameters
    current_prompt = img_gen.prompt
    strength = 0.3  # Lower strength for smoother transitions
    strength_delta = 0.05
    
    # Effect modifiers to add variety
    effects = [
        "glowing light",
        "misty atmosphere", 
        "golden hour lighting",
        "ethereal glow",
        "deep shadows",
        "vibrant colors",
        "soft focus",
        "mystical energy"
    ]
    current_effects = []
    
    print("🎨 Generating initial image...")
    try:
        # Generate initial image
        image = img_gen.gen()
        show_image(image, win)
        print("🖼️ Initial image generated!")
    except Exception as e:
        print(f"❌ Failed to generate initial image: {e}")
        return
    
    print("\n🎮 Controls:")
    print("  ← → : Adjust transformation strength")
    print("  ↑ ↓ : Navigate through variations") 
    print("  Space: Add random effects")
    print("  R: Reset effects")
    print("  Escape: Exit")
    print(f"\n🌟 Starting navigation with prompt: '{current_prompt}'")
    
    # Main event loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                
                # Initialize generation parameters
                gen_prompt = current_prompt
                if current_effects:
                    gen_prompt += ", " + ", ".join(current_effects)
                
                try:
                    # Navigation controls
                    if event.key == pygame.K_RIGHT:
                        # Increase strength for more dramatic changes
                        strength = min(1.0, strength + strength_delta)
                        print(f"🔥 Increased strength: {strength:.2f}")
                        image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                    
                    elif event.key == pygame.K_LEFT:
                        # Decrease strength for subtler changes
                        strength = max(0.1, strength - strength_delta)
                        print(f"🌙 Decreased strength: {strength:.2f}")
                        image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                    
                    elif event.key == pygame.K_UP:
                        # Move "forward" in image space with current settings
                        print("⬆️ Moving forward in image space...")
                        image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                    
                    elif event.key == pygame.K_DOWN:
                        # Move "backward" with lower strength for stability
                        print("⬇️ Moving backward in image space...")
                        image = img_gen.gen_img2img(strength=max(0.2, strength * 0.7), prompt=gen_prompt)
                    
                    elif event.key == pygame.K_SPACE:
                        # Add a random effect
                        import random
                        if len(current_effects) < 3:  # Limit effects to avoid prompt overflow
                            new_effect = random.choice([e for e in effects if e not in current_effects])
                            current_effects.append(new_effect)
                            print(f"✨ Added effect: '{new_effect}'")
                            gen_prompt = current_prompt + ", " + ", ".join(current_effects)
                            image = img_gen.gen_img2img(strength=strength, prompt=gen_prompt)
                        else:
                            print("🚫 Maximum effects reached. Press 'R' to reset.")
                            continue
                    
                    elif event.key == pygame.K_r:
                        # Reset effects
                        current_effects.clear()
                        print("🔄 Effects reset")
                        image = img_gen.gen_img2img(strength=strength, prompt=current_prompt)
                    
                    else:
                        continue  # No action for other keys
                    
                    # Display the new image
                    show_image(image, win)
                    
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    continue
    
    # Cleanup
    print("🧹 Cleaning up...")
    if hasattr(img_gen.backend, 'cleanup'):
        img_gen.backend.cleanup()
    
    pygame.quit()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
