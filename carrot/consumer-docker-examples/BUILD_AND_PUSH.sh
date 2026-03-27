#!/bin/bash

echo "╔════════════════════════════════════════════╗"
echo "║   BroccoByte Docker Image Builder         ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker found"
echo ""

# Get Docker Hub username
read -p "Enter your Docker Hub username: " USERNAME

if [ -z "$USERNAME" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

echo ""
echo "Available examples:"
echo "  1. hello-gpu       - Simple GPU test"
echo "  2. file-output     - Save results to file"
echo "  3. benchmark       - GPU performance benchmark"
echo "  4. proof-of-work   - Verifiable computation"
echo "  5. custom          - Build from custom Dockerfile"
echo ""

read -p "Choose example (1-5): " CHOICE

case $CHOICE in
    1)
        EXAMPLE="1-hello-gpu"
        IMAGE_NAME="hello-gpu"
        ;;
    2)
        EXAMPLE="2-file-output"
        IMAGE_NAME="file-output"
        ;;
    3)
        EXAMPLE="3-benchmark"
        IMAGE_NAME="gpu-benchmark"
        ;;
    4)
        EXAMPLE="4-proof-of-work"
        IMAGE_NAME="proof-of-work"
        ;;
    5)
        read -p "Enter path to Dockerfile directory: " CUSTOM_PATH
        EXAMPLE="$CUSTOM_PATH"
        read -p "Enter image name: " IMAGE_NAME
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

FULL_IMAGE="$USERNAME/$IMAGE_NAME"

echo ""
echo "Building: $FULL_IMAGE"
echo "From: $EXAMPLE"
echo ""

# Build
echo "Step 1/3: Building Docker image..."
if [ "$CHOICE" == "5" ]; then
    cd "$EXAMPLE"
else
    cd "$(dirname "$0")/$EXAMPLE"
fi

docker build -t "$FULL_IMAGE" .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""

# Test locally
echo "Step 2/3: Testing locally..."
echo "Running: docker run --rm $FULL_IMAGE"
echo ""
docker run --rm "$FULL_IMAGE"

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Test failed, but image was built"
    echo ""
    read -p "Continue with push? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "Step 3/3: Pushing to Docker Hub..."
echo ""

# Check if logged in
docker info 2>/dev/null | grep -q "Username: $USERNAME"
if [ $? -ne 0 ]; then
    echo "Logging in to Docker Hub..."
    docker login
    if [ $? -ne 0 ]; then
        echo "❌ Login failed"
        exit 1
    fi
fi

# Push
docker push "$FULL_IMAGE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Push failed"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   Success! Image Published                 ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Your Docker image is now public:"
echo ""
echo "  📦 Image: $FULL_IMAGE"
echo "  🌐 URL: https://hub.docker.com/r/$FULL_IMAGE"
echo ""
echo "To submit on BroccoByte:"
echo "  1. Go to Consumer Dashboard"
echo "  2. Click 'Post New Job'"
echo "  3. Select 'Docker Image'"
echo "  4. Enter: $FULL_IMAGE"
echo "  5. Set payment and submit"
echo ""
echo "Providers will automatically pull and run your image!"
echo ""
