# Video Analyzer App - Automated Setup Script (Windows PowerShell)
# Run this from the Clipper root folder: .\setup.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Video Analyzer App - Setup Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python 3 is not installed. Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

$nodeVersion = (node --version).Substring(1).Split('.')[0]
if ([int]$nodeVersion -lt 16) {
    Write-Host "❌ Node.js version 16+ required. You have $(node --version)" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Node.js $(node --version) found" -ForegroundColor Green
Write-Host "✅ Python $(python --version) found" -ForegroundColor Green
Write-Host ""

# Step 1: Set up Python virtual environment
Write-Host "🐍 Setting up Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Step 2: Install Python packages
Write-Host "📦 Installing Python packages..." -ForegroundColor Yellow
Write-Host "   This may take a few minutes..." -ForegroundColor Gray

python -m pip install --upgrade pip | Out-Null

Write-Host "   Installing Flask and dependencies..." -ForegroundColor Gray
python -m pip install flask flask-cors | Out-Null

Write-Host "   Installing core packages..." -ForegroundColor Gray
python -m pip install openai opencv-python Pillow python-dotenv requests | Out-Null

Write-Host "   Installing scikit-learn (this may take a while)..." -ForegroundColor Gray
python -m pip install scikit-learn --prefer-binary | Out-Null

Write-Host "✅ Python packages installed" -ForegroundColor Green
Write-Host ""

# Step 3: Install Node.js packages
Write-Host "📦 Installing Node.js packages..." -ForegroundColor Yellow
Set-Location video-analyzer-app

if (-not (Test-Path "node_modules")) {
    npm install | Out-Null
    Write-Host "✅ Node.js packages installed" -ForegroundColor Green
} else {
    Write-Host "✅ Node.js packages already installed" -ForegroundColor Green
}

Set-Location ..
Write-Host ""

# Step 4: Set up .env file
Write-Host "🔑 Setting up OpenAI API key..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Please enter your OpenAI API key:" -ForegroundColor Yellow
    Write-Host "(You can get one from https://platform.openai.com/api-keys)" -ForegroundColor Gray
    Write-Host ""
    $apiKey = Read-Host "API Key"
    
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        Write-Host "⚠️  No API key provided. You can add it later by creating a .env file with:" -ForegroundColor Yellow
        Write-Host "   OPENAI_API_KEY=your-key-here" -ForegroundColor Gray
    } else {
        "OPENAI_API_KEY=$apiKey" | Out-File -FilePath .env -Encoding utf8
        Write-Host "✅ API key saved to .env file" -ForegroundColor Green
    }
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}
Write-Host ""

# Step 5: Verify installation
Write-Host "🔍 Verifying installation..." -ForegroundColor Yellow

try {
    python -c "import flask, sklearn, openai" 2>$null
    Write-Host "✅ Python packages verified" -ForegroundColor Green
} catch {
    Write-Host "❌ Python package verification failed" -ForegroundColor Red
    exit 1
}

if (Test-Path "video-analyzer-app/node_modules") {
    Write-Host "✅ Node.js packages verified" -ForegroundColor Green
} else {
    Write-Host "❌ Node.js packages not found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the app, use:" -ForegroundColor Yellow
Write-Host "  cd video-analyzer-app" -ForegroundColor Gray
Write-Host "  npm run electron" -ForegroundColor Gray
Write-Host ""
Write-Host "Or run the quick-start script:" -ForegroundColor Yellow
Write-Host "  .\run.ps1" -ForegroundColor Gray
Write-Host ""

