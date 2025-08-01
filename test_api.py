#!/usr/bin/env python3
"""Simple test client for the Dreamspace API server."""

import requests
import base64
import json
from PIL import Image
from io import BytesIO

def test_generate_image(server_url="http://localhost:8001", save_image=True):
    """Test the /generate endpoint."""
    
    # Simple request
    request_data = {
        "prompt": "a beautiful landscape with mountains and lakes",
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    }
    
    print(f"🚀 Testing image generation...")
    print(f"Server: {server_url}")
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        # Make the request
        response = requests.post(
            f"{server_url}/generate",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 422:
            print("❌ Validation Error (422):")
            print(json.dumps(response.json(), indent=2))
            return None
        elif response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
        
        # Parse response
        result = response.json()
        print("✅ Image generated successfully!")
        print(f"Metadata: {result.get('metadata', {})}")
        
        if save_image and 'image' in result:
            # Decode and save image
            image_data = base64.b64decode(result['image'])
            image = Image.open(BytesIO(image_data))
            
            filename = "test_generated.png"
            image.save(filename)
            print(f"💾 Image saved as: {filename}")
            print(f"Image size: {image.size}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_health_check(server_url="http://localhost:8001"):
    """Test the /health endpoint."""
    try:
        response = requests.get(f"{server_url}/health")
        print(f"Health Check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server healthy: {json.dumps(health_data, indent=2)}")
            return health_data
        else:
            print(f"❌ Health check failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dreamspace API")
    parser.add_argument("--url", default="http://localhost:8001", help="Server URL")
    parser.add_argument("--no-save", action="store_true", help="Don't save generated image")
    
    args = parser.parse_args()
    
    print("🧪 Dreamspace API Test Client")
    print("=" * 40)
    
    # Test health first
    health = test_health_check(args.url)
    if not health:
        print("❌ Server not responding. Is it running?")
        exit(1)
    
    print("\n" + "=" * 40)
    
    # Test image generation
    result = test_generate_image(args.url, not args.no_save)
    
    if result:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Test failed!")
        exit(1)
