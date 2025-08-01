"""Multi-server client example.

This example shows how to connect to different Dreamspace Co-Pilot servers
running different models (SD 1.5, SD 2.1, Kandinsky 2.1) and compare their outputs.
"""

import sys
import os
import time
from typing import Dict, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dreamspace import ImgGen, Config
from dreamspace.config.settings import RemoteConfig


def test_server(name: str, api_url: str, api_key: str = None) -> Dict:
    """Test a single server and return results."""
    print(f"\n🔗 Testing {name} server at {api_url}")
    
    try:
        # Configure client
        config = Config()
        config.remote = RemoteConfig(
            api_url=api_url,
            api_key=api_key,
            timeout=120,
            max_retries=2
        )
        
        # Create client
        img_gen = ImgGen("remote", config=config)
        
        # Test health
        if hasattr(img_gen.backend, 'health_check'):
            if not img_gen.backend.health_check():
                return {"status": "unhealthy", "error": "Health check failed"}
        
        print(f"  ✅ {name} server is healthy")
        
        # Test generation
        print(f"  🎨 Generating test image...")
        start_time = time.time()
        
        prompt = "a mystical crystal cave with bioluminescent formations"
        image = img_gen.gen(prompt, num_inference_steps=20)  # Faster for testing
        
        generation_time = time.time() - start_time
        
        # Save image
        filename = f"server_test_{name.lower().replace(' ', '_')}.png"
        image.save(filename)
        
        print(f"  💾 Saved as '{filename}' ({generation_time:.1f}s)")
        
        # Test img2img
        print(f"  🔄 Testing img2img...")
        start_time = time.time()
        
        evolved_image = img_gen.gen_img2img(
            strength=0.4,
            prompt="a mystical crystal cave with golden light streaming in",
            num_inference_steps=15
        )
        
        img2img_time = time.time() - start_time
        
        evolved_filename = f"server_test_{name.lower().replace(' ', '_')}_evolved.png"
        evolved_image.save(evolved_filename)
        
        print(f"  💾 Evolved image saved as '{evolved_filename}' ({img2img_time:.1f}s)")
        
        return {
            "status": "success",
            "generation_time": generation_time,
            "img2img_time": img2img_time,
            "files": [filename, evolved_filename]
        }
        
    except Exception as e:
        print(f"  ❌ Error testing {name}: {e}")
        return {"status": "error", "error": str(e)}


def main():
    """Test multiple servers and compare results."""
    print("🌐 Multi-Server Client Test")
    print("=" * 50)
    
    # Server configurations
    # Adjust URLs and ports to match your server setup
    servers = [
        {
            "name": "Kandinsky 2.1",
            "url": "http://localhost:8000",  # Default Kandinsky server
            "api_key": None  # Set if your server requires auth
        },
        {
            "name": "Stable Diffusion 1.5", 
            "url": "http://localhost:8001",  # SD 1.5 server
            "api_key": None
        },
        {
            "name": "Stable Diffusion 2.1",
            "url": "http://localhost:8002",  # SD 2.1 server  
            "api_key": None
        }
    ]
    
    # Test each server
    results = {}
    for server in servers:
        results[server["name"]] = test_server(
            server["name"], 
            server["url"], 
            server["api_key"]
        )
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    successful_servers = []
    failed_servers = []
    
    for name, result in results.items():
        if result["status"] == "success":
            successful_servers.append(name)
            print(f"✅ {name}")
            print(f"   Generation: {result['generation_time']:.1f}s")
            print(f"   Img2Img: {result['img2img_time']:.1f}s")
            print(f"   Files: {', '.join(result['files'])}")
        else:
            failed_servers.append(name)
            print(f"❌ {name}: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 Summary: {len(successful_servers)}/{len(servers)} servers working")
    
    if successful_servers:
        print("\n🚀 Working servers:")
        for server in successful_servers:
            print(f"  • {server}")
        
        print("\n💡 You can now use these servers in your applications:")
        for i, server in enumerate(servers):
            if server["name"] in successful_servers:
                print(f"""
# Connect to {server["name"]}
config = Config()
config.remote = RemoteConfig(api_url="{server["url"]}")
img_gen = ImgGen("remote", config=config)
""")
    
    if failed_servers:
        print(f"\n⚠️ Failed servers: {', '.join(failed_servers)}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure servers are running:")
        for server in servers:
            if server["name"] in failed_servers:
                backend_name = {
                    "Kandinsky 2.1": "kandinsky21_server",
                    "Stable Diffusion 1.5": "sd15_server", 
                    "Stable Diffusion 2.1": "sd21_server"
                }.get(server["name"], "unknown")
                port = server["url"].split(":")[-1]
                print(f"     python scripts/start_server.py --backend {backend_name} --port {port}")
        print("  2. Check firewall settings")
        print("  3. Verify API keys if authentication is enabled")
    
    print("\n🖼️ Generated images saved in current directory")


if __name__ == "__main__":
    main()
