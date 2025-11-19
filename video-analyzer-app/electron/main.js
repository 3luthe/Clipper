const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 700,
    backgroundColor: '#1a1a1a',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Load the app
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Start Python backend server
function startPythonBackend() {
  const pythonPath = path.join(__dirname, '../../.venv/bin/python');
  const scriptPath = path.join(__dirname, '../../src/api_server.py');
  const cwd = path.join(__dirname, '../..');

  console.log('Starting Python backend...');
  console.log('Python path:', pythonPath);
  console.log('Script path:', scriptPath);
  console.log('Working dir:', cwd);

  pythonProcess = spawn(pythonPath, [scriptPath], {
    cwd: cwd,
    env: { ...process.env }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python process:', err);
  });
}

app.whenReady().then(() => {
  createWindow();
  // Don't start Python backend - use external server instead
  // startPythonBackend();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

// IPC handlers for communication with renderer
ipcMain.handle('get-videos', async () => {
  // Will communicate with Python backend
  return [];
});

// Handle file selection dialog
ipcMain.handle('select-videos', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'] }
    ],
    title: 'Select Video Files'
  });

  if (result.canceled) {
    return [];
  }

  // Return the full file paths
  return result.filePaths;
});

