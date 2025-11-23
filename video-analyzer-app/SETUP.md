# Video Analyzer App - Quick Setup Guide

This guide will help you get the Video Analyzer app running step-by-step.

## ✅ Prerequisites Check

Before starting, verify you have:
- [ ] Node.js 16+ installed (`node --version`)
- [ ] npm 10+ installed (`npm --version`)
- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] OpenAI API key (get one at https://platform.openai.com/api-keys)

## 📦 Installation Steps

### Step 1: Navigate to Project Root
```bash
cd /path/to/Clipper
```

### Step 2: Set Up Python Environment

**Create virtual environment:**
```bash
python3 -m venv .venv
```

**Activate virtual environment:**
```bash
# macOS/Linux:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

**Install Python packages:**
```bash
# Install essential packages
pip install flask flask-cors

# Install core dependencies
pip install openai opencv-python Pillow python-dotenv requests

# Install scikit-learn (for search functionality)
# If this fails, try: pip install scikit-learn --prefer-binary
pip install scikit-learn

# Or install everything from requirements.txt
pip install -r requirements.txt
```

**Verify Python setup:**
```bash
python -c "import flask, sklearn, openai, cv2; print('✓ All packages installed')"
```

### Step 3: Set Up Frontend

**Navigate to app directory:**
```bash
cd video-analyzer-app
```

**Install Node.js dependencies:**
```bash
npm install
```

This will install:
- React, Vite, Electron
- All other frontend dependencies

**Verify frontend setup:**
```bash
npx electron --version  # Should show v39.1.0 or similar
```

### Step 4: Configure API Key

**Create `.env` file in project root:**
```bash
cd ..  # Go back to project root
echo "OPENAI_API_KEY=your-actual-api-key-here" > .env
```

**Or manually create `.env` file:**
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

⚠️ **Important:** Replace `your-actual-api-key-here` with your real OpenAI API key.

## 🚀 Running the App

### Option 1: One-Command Start (Recommended)

From the `video-analyzer-app` directory:
```bash
npm run electron
```

This single command will:
1. ✅ Start Vite dev server (port 5173)
2. ✅ Start Python Flask backend (port 5001)
3. ✅ Wait for both to be ready
4. ✅ Launch Electron desktop app

### Option 2: Manual Start (For Debugging)

**Terminal 1 - Backend:**
```bash
# From project root
source .venv/bin/activate
python src/api_server.py
```

**Terminal 2 - Frontend:**
```bash
# From video-analyzer-app directory
npm run dev
```

**Terminal 3 - Electron:**
```bash
# From video-analyzer-app directory
NODE_ENV=development npx electron .
```

## ✅ Verification Checklist

After starting the app, verify:

- [ ] Backend is running: `curl http://localhost:5001/api/health` returns `{"status":"ok"}`
- [ ] Frontend is running: `curl http://localhost:5173` returns HTML
- [ ] Electron app window opens
- [ ] No error messages in terminal
- [ ] App shows "VIDEO ANALYZER" interface

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```bash
source .venv/bin/activate  # Make sure venv is activated
pip install flask flask-cors
```

### Issue: "Backend API is not running"
**Solution:**
1. Check if backend is running: `curl http://localhost:5001/api/health`
2. If not, start it manually: `python src/api_server.py`
3. Check for port conflicts: `lsof -i :5001`

### Issue: "scikit-learn installation fails"
**Solution:**
```bash
pip install scikit-learn --prefer-binary
# Or install scipy first:
pip install scipy
pip install scikit-learn
```

### Issue: "Port already in use"
**Solution:**
```bash
# Kill process on port 5173 (Vite)
lsof -ti:5173 | xargs kill -9

# Kill process on port 5001 (Flask)
lsof -ti:5001 | xargs kill -9
```

### Issue: Electron app shows blank screen
**Solution:**
1. Open DevTools in Electron (View → Toggle Developer Tools)
2. Check console for errors
3. Verify Vite is running: open `http://localhost:5173` in browser

## 📝 Next Steps

Once the app is running:

1. **Upload Videos:** Click the Upload tab and select video files
2. **Analyze Videos:** Click "Analyze Videos" to process them with OpenAI
3. **Search:** Use the search bar to find specific scenes
4. **Browse:** View analyzed frames in the grid

## 🆘 Still Having Issues?

1. Check all terminal output for error messages
2. Verify all prerequisites are installed correctly
3. Make sure virtual environment is activated
4. Check that ports 5001 and 5173 are not blocked
5. Review the main README.md for more detailed troubleshooting

---

**Last Updated:** Based on successful setup with:
- Node.js v22.15.0
- npm v10.9.2
- Python 3.13.4
- Electron v39.1.0

