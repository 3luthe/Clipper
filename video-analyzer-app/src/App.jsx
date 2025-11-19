import { useState, useEffect } from 'react';
import {
  Search,
  Film,
  Upload,
  Grid3x3,
  List,
  Play,
  MapPin,
  Clock,
  Tag,
  Sparkles,
  AlertCircle,
  Trash2,
  CheckCircle,
  XCircle,
  AlertTriangle,
} from 'lucide-react';
import videoService from './api/videoService';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('search'); // 'search', 'upload', 'videos'
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState('grid');
  const [activeFilter, setActiveFilter] = useState('all');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const [error, setError] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Analysis progress state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [videosToAnalyze, setVideosToAnalyze] = useState([]);

  // Library state
  const [libraryVideos, setLibraryVideos] = useState([]);
  const [libraryLoading, setLibraryLoading] = useState(false);

  // Video player state
  const [showVideoPlayer, setShowVideoPlayer] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoMetadata, setVideoMetadata] = useState([]);
  const [currentFrameData, setCurrentFrameData] = useState(null);
  const [videoDuration, setVideoDuration] = useState(0);

  const filters = [
    { id: 'all', label: 'All Results', icon: Sparkles },
    { id: 'landmarks', label: 'Landmarks', icon: MapPin },
    { id: 'people', label: 'People', icon: Tag },
    { id: 'nature', label: 'Nature', icon: Film },
  ];

  // Check API connection on mount
  useEffect(() => {
    checkApiConnection();
    loadInitialResults();
  }, []);

  const checkApiConnection = async () => {
    try {
      const isHealthy = await videoService.healthCheck();
      setApiConnected(isHealthy);
      if (!isHealthy) {
        setError('Could not connect to backend API');
      }
    } catch (err) {
      setApiConnected(false);
      setError('Backend API is not running');
    }
  };

  const loadInitialResults = async () => {
    setLoading(true);
    try {
      const searchResult = await videoService.search('');

      // Filter out results without descriptions (old format or unanalyzed)
      const validResults = searchResult.results.filter(r =>
        r.description && r.description.length > 0
      );

      setResults(validResults);
      setError(null);
    } catch (err) {
      setError('Failed to load results');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!apiConnected) {
      setError('API not connected');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const searchResult = await videoService.search(searchQuery);

      // Filter out results without descriptions (old format or unanalyzed)
      const validResults = searchResult.results.filter(r =>
        r.description && r.description.length > 0
      );

      setResults(validResults);
    } catch (err) {
      setError('Search failed');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadLibraryVideos = async () => {
    setLibraryLoading(true);
    try {
      const videos = await videoService.getAllVideos();
      setLibraryVideos(videos);
    } catch (err) {
      console.error('Failed to load library:', err);
      setError('Failed to load video library');
    } finally {
      setLibraryLoading(false);
    }
  };

  const deleteLibraryVideo = async (videoId, videoName) => {
    if (!confirm(`Are you sure you want to delete "${videoName}"? This will remove all analysis data and thumbnails.`)) {
      return;
    }

    try {
      await videoService.deleteVideo(videoId);
      setLibraryVideos(libraryVideos.filter(v => v.id !== videoId));
      setUploadStatus(`Deleted ${videoName}`);
      setTimeout(() => setUploadStatus(null), 3000);
    } catch (err) {
      setError(`Failed to delete video: ${err.message}`);
    }
  };

  const startAnalysis = async () => {
    try {
      setError(null);

      // Get videos that need analysis
      const response = await videoService.getVideosToAnalyze();

      if (response.count === 0) {
        setUploadStatus('No videos need analysis! All videos are already processed.');
        return;
      }

      // Start the actual analysis on the backend
      await videoService.startAnalysis();

      setVideosToAnalyze(response.videos);
      setIsAnalyzing(true);
      setCurrentVideo({ filename: response.videos[0]?.filename });
      setAnalysisProgress(0);

      // Poll for progress updates
      const pollInterval = setInterval(async () => {
        try {
          const progress = await videoService.getAnalysisProgress();

          if (!progress.running) {
            // Analysis complete
            clearInterval(pollInterval);
            setIsAnalyzing(false);
            setAnalysisProgress(100);
            setUploadStatus(`Successfully analyzed ${response.count} video(s)!`);

            // Refresh results
            setTimeout(() => {
              setActiveTab('search');
              loadInitialResults();
              setUploadStatus(null);
            }, 2000);
          } else {
            // Update progress
            setCurrentVideo({ filename: progress.current_video });
            setAnalysisProgress(progress.progress);
          }
        } catch (err) {
          console.error('Progress poll error:', err);
        }
      }, 1000); // Poll every second

    } catch (err) {
      console.error('Analysis error:', err);
      setError('Failed to start analysis: ' + (err.response?.data?.error || err.message));
      setIsAnalyzing(false);
    }
  };

  const handleFileSelect = async () => {
    try {
      setError(null);

      // Use Electron's dialog API if available
      if (window.electron && window.electron.selectVideos) {
        const filePaths = await window.electron.selectVideos();

        if (filePaths.length === 0) {
          return; // User cancelled
        }

        setUploadStatus(`Selected ${filePaths.length} video(s)`);
        console.log('File paths:', filePaths);

        // Register videos with backend
        setUploadStatus('Registering videos...');
        const result = await videoService.registerVideos(filePaths);

        setUploadStatus(`Successfully registered ${result.count} video(s)! Click "Analyze Videos" to process them.`);

        // Switch to upload tab and show analyze button
        setTimeout(() => {
          setUploadStatus(null);
          setActiveTab('upload');
        }, 2000);
      } else {
        // Fallback for non-Electron environments
        setError('File selection is only available in the Electron app');
      }
    } catch (err) {
      console.error('File select error:', err);
      setError('Failed to select videos: ' + (err.response?.data?.error || err.message));
      setUploadStatus(null);
    }
  };

  const openVideoPlayer = async (result) => {
    try {
      setSelectedVideo(result);
      setShowVideoPlayer(true);

      // Load all metadata for this video
      const response = await videoService.search('', { video_id: result.video_id });
      setVideoMetadata(response.results || []);

      // Set initial frame data to the clicked frame
      setCurrentFrameData(result);
    } catch (err) {
      console.error('Failed to load video metadata:', err);
      setError('Failed to load video data');
    }
  };

  const closeVideoPlayer = () => {
    setShowVideoPlayer(false);
    setSelectedVideo(null);
    setVideoMetadata([]);
    setCurrentFrameData(null);
  };

  const updateFrameData = (currentTime) => {
    if (videoMetadata.length === 0) return;

    // Find the frame closest to current time
    const closestFrame = videoMetadata.reduce((prev, curr) => {
      return Math.abs(curr.timestamp - currentTime) < Math.abs(prev.timestamp - currentTime)
        ? curr
        : prev;
    });

    setCurrentFrameData(closestFrame);
  };

  // If video player is open, show that view instead
  if (showVideoPlayer && selectedVideo) {
    return (
      <div className="app-container video-player-view">
        {/* Video Player Header */}
        <div className="video-player-header">
          <button className="btn btn-secondary" onClick={closeVideoPlayer} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M12 19l-7-7 7-7"/>
            </svg>
            Back to Search
          </button>
          <h2 style={{ flex: 1, textAlign: 'center', fontSize: '16px', fontWeight: 500, color: 'var(--text-primary)' }}>
            {selectedVideo.filename}
          </h2>
          <div style={{ width: 120 }}></div>
        </div>

        {/* Main Video Content */}
        <div className="video-player-main">
          {/* Video Section */}
          <div className="video-display-section">
            <video
              id="main-video"
              controls
              style={{ width: '100%', height: '100%', background: '#000' }}
              onTimeUpdate={(e) => updateFrameData(e.target.currentTime)}
              onLoadedMetadata={(e) => {
                e.target.currentTime = selectedVideo.timestamp;
                setVideoDuration(e.target.duration);
              }}
            >
              <source src={`http://localhost:5001/api/videos/${selectedVideo.video_id}/stream`} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>

          {/* Frame Metadata Sidebar */}
          {currentFrameData && (
            <div className="video-metadata-sidebar">
              <h3>Frame Data @ {currentFrameData.timestamp.toFixed(1)}s</h3>

              <div className="metadata-grid">
                <div className="metadata-item">
                  <strong>Description:</strong>
                  <p>{currentFrameData.description}</p>
                </div>

                {currentFrameData.text_visible && currentFrameData.text_visible.length > 0 && (
                  <div className="metadata-item">
                    <strong>📝 Text Visible:</strong>
                    <p>{currentFrameData.text_visible.join(', ')}</p>
                  </div>
                )}

                {currentFrameData.people && currentFrameData.people.length > 0 && (
                  <div className="metadata-item">
                    <strong>People:</strong>
                    <p>{currentFrameData.people.join(', ')}</p>
                  </div>
                )}

                {currentFrameData.objects && currentFrameData.objects.length > 0 && (
                  <div className="metadata-item">
                    <strong>Objects:</strong>
                    <p>{currentFrameData.objects.slice(0, 10).join(', ')}{currentFrameData.objects.length > 10 ? '...' : ''}</p>
                  </div>
                )}

                {currentFrameData.mood && (
                  <div className="metadata-item">
                    <strong>Mood:</strong>
                    <p>{currentFrameData.mood}</p>
                  </div>
                )}

                {currentFrameData.setting && (
                  <div className="metadata-item">
                    <strong>Setting:</strong>
                    <p>{currentFrameData.setting}</p>
                  </div>
                )}

                {currentFrameData.location_type && (
                  <div className="metadata-item">
                    <strong>Location:</strong>
                    <p>{currentFrameData.location_type}</p>
                  </div>
                )}

                {currentFrameData.actions && currentFrameData.actions.length > 0 && (
                  <div className="metadata-item">
                    <strong>Actions:</strong>
                    <p>{currentFrameData.actions.join(', ')}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Timeline */}
        <div className="video-timeline">
          <div className="timeline-track">
            {videoMetadata.map((frame, idx) => {
              const position = videoDuration > 0 ? (frame.timestamp / videoDuration) * 100 : 0;
              const isActive = currentFrameData && Math.abs(frame.timestamp - currentFrameData.timestamp) < 0.5;

              return (
                <div
                  key={idx}
                  className={`timeline-marker ${isActive ? 'active' : ''}`}
                  style={{ left: `${position}%` }}
                  title={`${frame.timestamp.toFixed(1)}s`}
                  onClick={() => {
                    const video = document.getElementById('main-video');
                    if (video) video.currentTime = frame.timestamp;
                  }}
                >
                  {frame.thumbnail_path && (
                    <img
                      src={`http://localhost:5001${frame.thumbnail_path}`}
                      alt=""
                      className="timeline-thumbnail"
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div
          className={`sidebar-icon ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
          title="Search"
        >
          <Search size={20} />
        </div>
        <div
          className={`sidebar-icon ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
          title="Upload Videos"
        >
          <Upload size={20} />
        </div>
        <div
          className={`sidebar-icon ${activeTab === 'videos' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('videos');
            loadLibraryVideos();
          }}
          title="Video Library"
        >
          <Film size={20} />
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Header */}
        <div className="header">
          <div className="header-title">VIDEO ANALYZER</div>
          <div className="header-actions">
            <div className="view-toggle">
              <button
                className={`view-toggle-btn ${viewMode === 'grid' ? 'active' : ''}`}
                onClick={() => setViewMode('grid')}
              >
                <Grid3x3 size={14} />
              </button>
              <button
                className={`view-toggle-btn ${viewMode === 'list' ? 'active' : ''}`}
                onClick={() => setViewMode('list')}
              >
                <List size={14} />
              </button>
            </div>
          </div>
        </div>

        {/* Search Panel */}
        <div className="search-panel">
          {error && (
            <div style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '12px 16px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              color: '#ef4444',
              maxWidth: '800px',
              margin: '0 auto 16px',
            }}>
              <AlertCircle size={16} />
              {error}
            </div>
          )}
          <div className="search-wrapper">
            <Search className="search-icon" size={18} />
            <input
              type="text"
              className="search-input"
              placeholder="Search by vibe, objects, mood, location... (e.g., 'peaceful nature scene', 'dog running', 'sunset')"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
            <div className="filter-pills">
              {filters.map((filter) => {
                const Icon = filter.icon;
                return (
                  <button
                    key={filter.id}
                    className={`filter-pill ${activeFilter === filter.id ? 'active' : ''}`}
                    onClick={() => setActiveFilter(filter.id)}
                  >
                    <Icon size={12} style={{ display: 'inline', marginRight: 4 }} />
                    {filter.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="content-wrapper">
          <div className="panel-main">
            {activeTab === 'search' && (
              <>
                {loading ? (
                  <div className="empty-state">
                    <div className="loading-spinner"></div>
                  </div>
                ) : results.length === 0 ? (
                  <div className="empty-state">
                    <Sparkles className="empty-state-icon" size={64} />
                    <div className="empty-state-title">No analyzed frames found</div>
                    <div className="empty-state-text">
                      Your videos need to be analyzed with OpenAI Vision API.<br />
                      Go to Upload tab and click "Analyze Videos" to start.
                    </div>
                    <button
                      className="btn btn-primary"
                      style={{ marginTop: 24 }}
                      onClick={() => setActiveTab('upload')}
                    >
                      <Sparkles size={16} />
                      Analyze Videos
                    </button>
                    <div style={{ marginTop: 16, color: 'var(--text-tertiary)', fontSize: 12 }}>
                      Analysis extracts rich metadata from every frame using AI
                    </div>
                  </div>
                ) : (
                  <div className="thumbnail-grid">
                    {results.map((result, idx) => (
                      <div
                        key={`${result.video_id}-${result.timestamp}-${idx}`}
                        className="thumbnail-card"
                        onClick={() => openVideoPlayer(result)}
                      >
                        <div className="thumbnail-image">
                          {result.thumbnail_path ? (
                            <img
                              src={`http://localhost:5001${result.thumbnail_path}`}
                              alt={result.description}
                              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                              onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.parentElement.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; color: var(--text-tertiary);"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg></div>';
                              }}
                            />
                          ) : (
                            <Play size={48} />
                          )}
                        </div>
                        <div className="thumbnail-info">
                          <div className="thumbnail-title">{result.filename}</div>
                          <div className="thumbnail-meta">
                            <Clock size={10} style={{ display: 'inline', marginRight: 4 }} />
                            {result.timestamp.toFixed(1)}s
                            {result.mood && (
                              <>
                                {' · '}
                                <span style={{ color: 'var(--accent-purple)' }}>{result.mood}</span>
                              </>
                            )}
                          </div>
                          <div className="thumbnail-description">{result.description}</div>
                          {result.text_visible && result.text_visible.length > 0 && (
                            <div style={{
                              marginTop: 8,
                              padding: '6px 10px',
                              background: 'rgba(59, 130, 246, 0.1)',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                              borderRadius: '4px',
                              fontSize: '11px',
                              color: 'var(--accent-blue)',
                              fontFamily: 'var(--font-mono)'
                            }}>
                              📝 Text: "{result.text_visible.join(' ')}"
                            </div>
                          )}
                          <div style={{ marginTop: 8 }}>
                            {result.objects && result.objects.slice(0, 3).map((obj, i) => (
                              <span key={`${obj}-${i}`} className="tag">
                                <Tag size={10} />
                                {obj}
                              </span>
                            ))}
                            {result.landmarks && result.landmarks.length > 0 && (
                              <span className="tag" style={{ background: 'var(--accent-blue)', color: 'white' }}>
                                <MapPin size={10} />
                                {result.landmarks[0]}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}

            {activeTab === 'upload' && (
              <div className="empty-state">
                <Upload className="empty-state-icon" size={64} style={{ opacity: 0.5 }} />
                <div className="empty-state-title">Upload Videos for Analysis</div>
                <div className="empty-state-text">
                  Select video files to analyze with OpenAI Vision API.<br />
                  Each video will be split into frames and analyzed for rich metadata.
                </div>
                {uploadStatus && (
                  <div style={{
                    background: 'var(--accent-green)',
                    color: 'white',
                    padding: '12px 24px',
                    borderRadius: '6px',
                    marginTop: '16px',
                    fontWeight: 500,
                  }}>
                    {uploadStatus}
                  </div>
                )}
                <div style={{ display: 'flex', gap: '12px', marginTop: 24 }}>
                  <button
                    className="btn btn-primary"
                    style={{ fontSize: 14, padding: '12px 24px' }}
                    onClick={handleFileSelect}
                    disabled={isAnalyzing}
                  >
                    <Upload size={18} />
                    Select Videos
                  </button>
                  <button
                    className="btn btn-secondary"
                    style={{ fontSize: 14, padding: '12px 24px' }}
                    onClick={startAnalysis}
                    disabled={isAnalyzing}
                  >
                    <Sparkles size={18} />
                    {isAnalyzing ? 'Analyzing...' : 'Analyze Videos'}
                  </button>
                </div>
                <div style={{ marginTop: 16, color: 'var(--text-tertiary)', fontSize: 12 }}>
                  Supported formats: MP4, MOV, MKV, AVI
                </div>
              </div>
            )}

            {activeTab === 'videos' && (
              <>
                {libraryVideos.length === 0 && !libraryLoading ? (
                  <div className="empty-state">
                    <Film className="empty-state-icon" size={64} style={{ opacity: 0.5 }} />
                    <div className="empty-state-title">Video Library Empty</div>
                    <div className="empty-state-text">
                      Upload videos to build your searchable video library
                    </div>
                    <button
                      className="btn btn-primary"
                      style={{ marginTop: 24 }}
                      onClick={() => setActiveTab('upload')}
                    >
                      <Upload size={16} />
                      Upload Videos
                    </button>
                  </div>
                ) : (
                  <div style={{ padding: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                      <h2 style={{ fontSize: '18px', fontWeight: 500, color: 'var(--text-primary)' }}>
                        Video Library ({libraryVideos.length})
                      </h2>
                      <button
                        className="btn btn-secondary"
                        onClick={loadLibraryVideos}
                        disabled={libraryLoading}
                      >
                        {libraryLoading ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {libraryVideos.map((video) => {
                        const status = video.status || (video.metadata_exists ? 'analyzed' : (video.file_exists === false ? 'missing' : 'pending'));

                        return (
                          <div
                            key={video.id}
                            style={{
                              background: 'var(--bg-secondary)',
                              border: status === 'missing' ? '1px solid var(--accent-red)' : '1px solid var(--border-medium)',
                              borderRadius: '8px',
                              padding: '16px',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                            }}
                          >
                            <div style={{ flex: 1 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                                <Film size={20} style={{ color: status === 'missing' ? 'var(--accent-red)' : 'var(--accent-blue)' }} />
                                <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)' }}>
                                  {video.filename}
                                </div>
                                {status === 'analyzed' && (
                                  <CheckCircle size={16} style={{ color: 'var(--accent-green)' }} title="Analyzed" />
                                )}
                                {status === 'pending' && (
                                  <Clock size={16} style={{ color: 'var(--text-tertiary)' }} title="Pending analysis" />
                                )}
                                {status === 'missing' && (
                                  <AlertTriangle size={16} style={{ color: 'var(--accent-red)' }} title="File not found" />
                                )}
                              </div>
                              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                                {video.path}
                              </div>
                              {status === 'analyzed' && (
                                <div style={{ fontSize: '11px', color: 'var(--accent-green)', marginTop: '4px' }}>
                                  ✓ Analyzed
                                </div>
                              )}
                              {status === 'pending' && (
                                <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                                  Not analyzed yet
                                </div>
                              )}
                              {status === 'missing' && (
                                <div style={{ fontSize: '11px', color: 'var(--accent-red)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <AlertTriangle size={12} />
                                  Video file not found - Please re-upload or delete this entry
                                </div>
                              )}
                            </div>
                            <button
                              className="btn btn-icon"
                              onClick={() => deleteLibraryVideo(video.id, video.filename)}
                              style={{ color: 'var(--accent-red)' }}
                              title="Delete video"
                            >
                              <Trash2 size={18} />
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Analysis Progress Bar */}
      {isAnalyzing && (
        <div className="analysis-bar">
          <div className="analysis-bar-content">
            <div className="spinner-small"></div>
            <div className="analysis-info">
              <div className="analysis-title">Analyzing: {currentVideo?.filename || 'Video'}</div>
              <div className="analysis-subtitle">
                Processing frames with OpenAI Vision API...
              </div>
            </div>
            <div className="progress-bar-container">
              <div
                className="progress-bar-fill"
                style={{ width: `${analysisProgress}%` }}
              ></div>
            </div>
            <div className="analysis-stats">
              <div className="analysis-stat">
                <Clock size={14} />
                {Math.round(analysisProgress)}%
              </div>
              <div className="analysis-stat">
                {videosToAnalyze.findIndex(v => v.id === currentVideo?.id) + 1} / {videosToAnalyze.length}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;

