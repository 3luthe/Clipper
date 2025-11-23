#!/bin/bash

# Video Analyzer App - Quick Start Script
# Run this from the Clipper root folder: bash run.sh

echo "🚀 Starting Video Analyzer App..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first:"
    echo "   bash setup.sh"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. The app may not work without an OpenAI API key."
    echo "   Create a .env file with: OPENAI_API_KEY=your-key-here"
    echo ""
fi

# Start the app
cd video-analyzer-app
npm run electron

