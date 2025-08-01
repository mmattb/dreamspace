"""Example of using the remote API backend.

This example shows how to use the remote API backend to generate images
from a server running the Dreamspace Co-Pilot API.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dreamspace import ImgGen, Config
from dreamspace.config.settings import RemoteConfig


def main():
    """Example of remote API usage."""
    print("🌐 Remote API Example")
    print("=" * 50)
    
    # Configuration for remote API
    # Replace with your actual server URL and API key
    api_url = "http://localhost:8000"  # Change to your server URL
    api_key = None  # Set if your server requires authentication
    
    # Create configuration with remote settings
    config = Config()
    config.remote = RemoteConfig(
        api_url=api_url,
        api_key=api_key,
        timeout=120,  # Longer timeout for image generation
        max_retries=3
    )
    
    print(f"🔗 Connecting to: {api_url}")
    
    try:
        # Initialize with remote backend
        img_gen = ImgGen(
            backend="remote",
            prompt="a magical crystalline cave with glowing crystals",
            config=config
        )
        
        # Test connection
        if hasattr(img_gen.backend, 'health_check'):
            if img_gen.backend.health_check():
                print("✅ Server is healthy")
            else:
                print("❌ Server health check failed")
                return
        
        print("🎨 Generating images...")
        
        # Generate first image
        print("1. Generating initial image...")
        image1 = img_gen.gen()
        image1.save("remote_example_1.png")
        print("   💾 Saved as 'remote_example_1.png'")
        
        # Generate with img2img for continuity
        print("2. Generating evolved image...")
        image2 = img_gen.gen_img2img(
            strength=0.4,
            prompt="a magical crystalline cave with golden light streaming in"
        )
        image2.save("remote_example_2.png")
        print("   💾 Saved as 'remote_example_2.png'")
        
        # Try different prompt
        print("3. Generating with new prompt...")
        image3 = img_gen.gen(prompt="an underwater palace with bioluminescent coral")
        image3.save("remote_example_3.png")
        print("   💾 Saved as 'remote_example_3.png'")
        
        # Continue evolution
        print("4. Evolving underwater scene...")
        image4 = img_gen.gen_img2img(
            strength=0.3,
            prompt="an underwater palace with bioluminescent coral and schools of glowing fish"
        )
        image4.save("remote_example_4.png")
        print("   💾 Saved as 'remote_example_4.png'")
        
        print("\n✨ Remote generation complete!")
        print("🖼️ Check the generated images:")
        print("   • remote_example_1.png - Initial magical cave")
        print("   • remote_example_2.png - Cave with golden light")
        print("   • remote_example_3.png - Underwater palace")
        print("   • remote_example_4.png - Palace with glowing fish")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure the server is running:")
        print(f"      python scripts/start_server.py --host 0.0.0.0 --port 8000")
        print("   2. Check the API URL is correct")
        print("   3. Verify API key if authentication is enabled")
        print("   4. Check network connectivity")


if __name__ == "__main__":
    main()
