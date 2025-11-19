import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001/api';

class VideoService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Check if the API server is healthy
   */
  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data.status === 'ok';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Get all registered videos
   */
  async getAllVideos() {
    try {
      const response = await this.client.get('/videos');
      return response.data.videos || [];
    } catch (error) {
      console.error('Failed to fetch videos:', error);
      throw error;
    }
  }

  /**
   * Get a specific video's details
   */
  async getVideo(videoId) {
    try {
      const response = await this.client.get(`/videos/${videoId}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch video ${videoId}:`, error);
      throw error;
    }
  }

  /**
   * Get a video's analysis metadata
   */
  async getVideoMetadata(videoId) {
    try {
      const response = await this.client.get(`/videos/${videoId}/metadata`);
      return response.data.metadata || [];
    } catch (error) {
      console.error(`Failed to fetch metadata for ${videoId}:`, error);
      throw error;
    }
  }

  /**
   * Search through all analyzed videos
   */
  async search(query, filters = {}) {
    try {
      const response = await this.client.post('/search', {
        query,
        filters,
      });
      return {
        results: response.data.results || [],
        total: response.data.total || 0,
      };
    } catch (error) {
      console.error('Search failed:', error);
      throw error;
    }
  }

  /**
   * Register new videos for analysis
   */
  async registerVideos(videoPaths) {
    try {
      const response = await this.client.post('/videos/register', {
        paths: videoPaths,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to register videos:', error);
      throw error;
    }
  }

  /**
   * Get videos that need analysis
   */
  async getVideosToAnalyze() {
    try {
      const response = await this.client.post('/videos/analyze');
      return response.data;
    } catch (error) {
      console.error('Failed to get videos to analyze:', error);
      throw error;
    }
  }

  /**
   * Start actual video analysis
   */
  async startAnalysis() {
    try {
      const response = await this.client.post('/videos/start-analysis');
      return response.data;
    } catch (error) {
      console.error('Failed to start analysis:', error);
      throw error;
    }
  }

  /**
   * Get analysis progress
   */
  async getAnalysisProgress() {
    try {
      const response = await this.client.get('/videos/analysis-progress');
      return response.data;
    } catch (error) {
      console.error('Failed to get analysis progress:', error);
      throw error;
    }
  }

  /**
   * Delete a video from library
   */
  async deleteVideo(videoId) {
    try {
      const response = await this.client.delete(`/videos/${videoId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to delete video:', error);
      throw error;
    }
  }
}

export default new VideoService();

