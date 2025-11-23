# Video Analyzer App - Quick Start Script (Windows PowerShell)
# Run this from the Clipper root folder: .\run.ps1

Write-Host "🚀 Starting Video Analyzer App..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found. Please run setup.ps1 first:" -ForegroundColor Red
    Write-Host "   .\setup.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  Warning: .env file not found. The app may not work without an OpenAI API key." -ForegroundColor Yellow
    Write-Host "   Create a .env file with: OPENAI_API_KEY=your-key-here" -ForegroundColor Gray
    Write-Host ""
}

# Start the app
Set-Location video-analyzer-app
npm run electron

