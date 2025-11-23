# Video Analyzer App - Setup Instructions

## 🚀 Quick Setup (Recommended)

**For macOS/Linux:**
```bash
bash setup.sh
```

**For Windows (PowerShell):**
```powershell
.\setup.ps1
```

This will automatically:
- ✅ Check prerequisites
- ✅ Set up Python virtual environment
- ✅ Install all Python packages
- ✅ Install all Node.js packages
- ✅ Prompt for OpenAI API key
- ✅ Verify everything is installed correctly

Then to run the app:
```bash
bash run.sh
```

Or on Windows:
```powershell
.\run.ps1
```

---

## 📝 Manual Setup (Alternative)

If you prefer to set up manually, follow these steps:

## Step 1: Check Prerequisites

First, make sure you have the required software installed:

```bash
node --version
```

If this shows an error, install Node.js from https://nodejs.org/ (get version 16 or higher)

```bash
python3 --version
```

If this shows an error, install Python from https://www.python.org/ (get version 3.8 or higher)

## Step 2: Navigate to the Project Folder

Open your terminal and go to the Clipper folder:

```bash
cd /path/to/Clipper
```

(Replace `/path/to/Clipper` with the actual path where you downloaded/cloned the project)

## Step 3: Set Up Python Backend

### Create Python Virtual Environment

```bash
python3 -m venv .venv
```

### Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your terminal prompt.

### Install Python Packages

```bash
pip install flask flask-cors
```

```bash
pip install openai opencv-python Pillow python-dotenv requests
```

```bash
pip install scikit-learn
```

If the last command fails, try this instead:
```bash
pip install scikit-learn --prefer-binary
```

## Step 4: Set Up Frontend (Node.js)

### Go to the Video Analyzer App Folder

```bash
cd video-analyzer-app
```

### Install Node.js Packages

```bash
npm install
```

This will take a minute or two. Wait for it to finish.

### Go Back to Project Root

```bash
cd ..
```

## Step 5: Set Up Your OpenAI API Key

Create a file called `.env` in the Clipper folder:

**On macOS/Linux:**
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**On Windows (PowerShell):**
```powershell
echo "OPENAI_API_KEY=your-api-key-here" | Out-File -FilePath .env -Encoding utf8
```

**Important:** Replace `your-api-key-here` with your actual OpenAI API key. You can get one from https://platform.openai.com/api-keys

If you prefer, you can manually create a file named `.env` in the Clipper folder and put this inside it:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

## Step 6: Run the App

**Quick way:**
```bash
bash run.sh
```

**Or manually:**
```bash
cd video-analyzer-app
npm run electron
```

This will:
- Start the backend server
- Start the frontend server
- Open the Electron app window

**Wait for the app window to open.** This may take 10-30 seconds the first time.

## That's It!

The Video Analyzer app should now be running. You should see a window with "VIDEO ANALYZER" at the top.

## Troubleshooting

### If you see "ModuleNotFoundError: No module named 'flask'"

Make sure the virtual environment is activated. You should see `(.venv)` in your terminal prompt. If not, run:

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

Then try installing Flask again:
```bash
pip install flask flask-cors
```

### If you see "Backend API is not running"

The backend might not have started. Try running it manually in a separate terminal:

**Terminal 1 - Start Backend:**
```bash
cd /path/to/Clipper
source .venv/bin/activate
python src/api_server.py
```

**Terminal 2 - Start Frontend and Electron:**
```bash
cd /path/to/Clipper/video-analyzer-app
npm run dev
```

Then in a third terminal:
```bash
cd /path/to/Clipper/video-analyzer-app
NODE_ENV=development npx electron .
```

### If you see "Port already in use"

Something is already using port 5001 or 5173. Close any other running instances of the app, or kill the processes:

**On macOS/Linux:**
```bash
lsof -ti:5001 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

### If the Electron window is blank

Wait a few seconds for everything to load. If it's still blank after 30 seconds, check the terminal for error messages.

## Need Help?

- Check that all the commands completed without errors
- Make sure you're in the correct folder when running commands
- Verify your OpenAI API key is correct in the `.env` file
- Check the terminal output for any error messages

