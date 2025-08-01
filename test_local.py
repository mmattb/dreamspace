#!/usr/bin/env python3
"""Test the local server creation without running uvicorn."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dreamspace.servers.api_server import create_app
from fastapi.testclient import TestClient

def test_local_server():
    """Test server creation and endpoints locally."""
    
    print("🧪 Testing server creation...")
    
    try:
        # Create app without authentication
        app = create_app(backend_type="kandinsky_local", enable_auth=False, api_key=None)
        client = TestClient(app)
        
        print("✅ Server created successfully")
        
        # Test health endpoint
        print("Testing /health endpoint...")
        health_response = client.get("/health")
        print(f"Health status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health data: {health_response.json()}")
        
        # Test generate endpoint with minimal data
        print("Testing /generate endpoint...")
        generate_data = {"prompt": "test"}
        generate_response = client.post("/generate", json=generate_data)
        print(f"Generate status: {generate_response.status_code}")
        
        if generate_response.status_code == 422:
            print("❌ Validation error details:")
            print(generate_response.json())
        elif generate_response.status_code == 200:
            print("✅ Generate endpoint works!")
            response_data = generate_response.json()
            print(f"Response keys: {list(response_data.keys())}")
        else:
            print(f"❌ Unexpected status: {generate_response.status_code}")
            print(generate_response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_server()
