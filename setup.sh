#!/bin/bash

# Video Analyzer App - Automated Setup Script
# Run this from the Clipper root folder: bash setup.sh

set -e  # Exit on error

echo "=========================================="
echo "Video Analyzer App - Setup Script"
echo "=========================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ Node.js version 16+ required. You have $(node --version)"
    exit 1
fi

echo "✅ Node.js $(node --version) found"
echo "✅ Python $(python3 --version) found"
echo ""

# Step 1: Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Step 2: Install Python packages
echo "📦 Installing Python packages..."
echo "   This may take a few minutes..."

pip install --upgrade pip > /dev/null 2>&1

echo "   Installing Flask and dependencies..."
pip install flask flask-cors > /dev/null 2>&1

echo "   Installing core packages..."
pip install openai opencv-python Pillow python-dotenv requests > /dev/null 2>&1

echo "   Installing scikit-learn (this may take a while)..."
if ! pip install scikit-learn --prefer-binary > /dev/null 2>&1; then
    echo "   Retrying scikit-learn installation..."
    pip install scikit-learn > /dev/null 2>&1
fi

echo "✅ Python packages installed"
echo ""

# Step 3: Install Node.js packages
echo "📦 Installing Node.js packages..."
cd video-analyzer-app

if [ ! -d "node_modules" ]; then
    npm install > /dev/null 2>&1
    echo "✅ Node.js packages installed"
else
    echo "✅ Node.js packages already installed"
fi

cd ..
echo ""

# Step 4: Set up .env file
echo "🔑 Setting up OpenAI API key..."
if [ ! -f ".env" ]; then
    echo ""
    echo "Please enter your OpenAI API key:"
    echo "(You can get one from https://platform.openai.com/api-keys)"
    echo ""
    read -p "API Key: " api_key
    
    if [ -z "$api_key" ]; then
        echo "⚠️  No API key provided. You can add it later by creating a .env file with:"
        echo "   OPENAI_API_KEY=your-key-here"
    else
        echo "OPENAI_API_KEY=$api_key" > .env
        echo "✅ API key saved to .env file"
    fi
else
    echo "✅ .env file already exists"
fi
echo ""

# Step 5: Verify installation
echo "🔍 Verifying installation..."

if python -c "import flask, sklearn, openai" 2>/dev/null; then
    echo "✅ Python packages verified"
else
    echo "❌ Python package verification failed"
    exit 1
fi

if [ -d "video-analyzer-app/node_modules" ]; then
    echo "✅ Node.js packages verified"
else
    echo "❌ Node.js packages not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the app, use:"
echo "  cd video-analyzer-app"
echo "  npm run electron"
echo ""
echo "Or run the quick-start script:"
echo "  bash run.sh"
echo ""

