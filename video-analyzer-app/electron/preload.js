const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  getVideos: () => ipcRenderer.invoke('get-videos'),
  selectVideos: () => ipcRenderer.invoke('select-videos'),
});

