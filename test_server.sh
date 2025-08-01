#!/bin/bash

# Simple API test commands for the Dreamspace server
SERVER_URL="http://172.28.5.21:8001"

echo "🧪 Testing Dreamspace API Server at $SERVER_URL"
echo "=================================================="

echo "1. Testing health check..."
curl -s "$SERVER_URL/health" | python3 -m json.tool
echo ""

echo "2. Testing simple image generation..."
curl -X POST "$SERVER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains"}' \
  -s | head -c 200
echo ""
echo ""

echo "3. Testing with all parameters..."
curl -X POST "$SERVER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "astronaut riding a horse on mars",
    "width": 512,
    "height": 512,
    "num_inference_steps": 25,
    "guidance_scale": 7.5
  }' \
  -s | jq -r '.image' | base64 -d > test_image.png 2>/dev/null

if [ -f "test_image.png" ]; then
  echo "✅ Image saved as test_image.png"
  echo "Image size: $(file test_image.png)"
else
  echo "❌ Failed to save image"
fi

echo ""
echo "Done! 🎉"
