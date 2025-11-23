#!/usr/bin/env python3
"""
Flask API server for Video Analyzer
Provides REST API endpoints for the Electron frontend
"""

from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import os
import json
import sys
import subprocess
import threading
import queue
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.video_cache import iter_registered_videos, get_video_entry, register_videos, get_videos_to_process
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.video_cache import iter_registered_videos, get_video_entry, register_videos, get_videos_to_process

app = Flask(__name__)

# Enable CORS for all routes and origins (development mode)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/api/thumbnails/<video_id>/<filename>', methods=['GET'])
def serve_thumbnail(video_id, filename):
    """Serve thumbnail images"""
    try:
        # Use absolute path from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        thumbnail_path = os.path.join(project_root, 'data', 'thumbnails', video_id, filename)

        if not os.path.exists(thumbnail_path):
            return jsonify({'error': 'Thumbnail not found'}), 404

        return send_file(thumbnail_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos/<video_id>/stream', methods=['GET'])
def stream_video(video_id):
    """Stream video file"""
    try:
        entry = get_video_entry(video_id)
        if not entry:
            return jsonify({'error': 'Video not found'}), 404

        video_path = entry.get('path')
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404

        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get all registered videos"""
    videos = []
    for video_id, entry in iter_registered_videos():
        video_path = entry.get('path')
        metadata_file = entry.get('metadata_file')

        # Check if video file exists
        file_exists = video_path and os.path.exists(video_path)

        # Check if metadata exists
        metadata_exists = metadata_file and os.path.exists(metadata_file)

        videos.append({
            'id': video_id,
            'filename': entry.get('filename'),
            'path': video_path,
            'metadata_file': metadata_file,
            'mtime': entry.get('mtime'),
            'file_exists': file_exists,
            'metadata_exists': metadata_exists,
            'status': 'analyzed' if metadata_exists else ('missing' if not file_exists else 'pending')
        })
    return jsonify({'videos': videos})

@app.route('/api/videos/<video_id>', methods=['GET'])
def get_video(video_id):
    """Get a specific video's details"""
    entry = get_video_entry(video_id)
    if not entry:
        return jsonify({'error': 'Video not found'}), 404
    return jsonify(entry)

@app.route('/api/videos/<video_id>/metadata', methods=['GET'])
def get_video_metadata(video_id):
    """Get a video's analysis metadata"""
    entry = get_video_entry(video_id)
    if not entry:
        return jsonify({'error': 'Video not found'}), 404

    metadata_file = entry.get('metadata_file')
    if not metadata_file or not os.path.exists(metadata_file):
        return jsonify({'error': 'Metadata not found'}), 404

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return jsonify({'metadata': metadata})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_videos():
    """Search through all analyzed videos"""
    data = request.json
    query = data.get('query', '')
    filters = data.get('filters', {})
    video_id_filter = filters.get('video_id') if isinstance(filters, dict) else None

    # Collect all frames from all videos
    all_frames = []

    for video_id, entry in iter_registered_videos():
        # Skip if filtering by video_id and this isn't it
        if video_id_filter and video_id != video_id_filter:
            continue

        metadata_file = entry.get('metadata_file')
        if not metadata_file or not os.path.exists(metadata_file):
            continue

        video_path = entry.get('path')
        filename = entry.get('filename') or video_id

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            for frame_data in metadata:
                # Convert local thumbnail path to API URL
                thumbnail_path = frame_data.get('thumbnail_path')
                thumbnail_url = None
                if thumbnail_path:
                    # Extract video_id and filename from path like "data/thumbnails/VIDEO_ID/frame_X.jpg"
                    parts = thumbnail_path.split('/')
                    if len(parts) >= 3:
                        thumb_video_id = parts[-2]
                        thumb_filename = parts[-1]
                        thumbnail_url = f'/api/thumbnails/{thumb_video_id}/{thumb_filename}'

                frame_result = {
                    'video_id': video_id,
                    'video_path': video_path,
                    'filename': filename,
                    'timestamp': frame_data.get('timestamp', 0),
                    'thumbnail_path': thumbnail_url,  # Now an API URL instead of file path
                    'description': frame_data.get('description', ''),
                    'mood': frame_data.get('mood', ''),
                    'objects': frame_data.get('objects', []),
                    'landmarks': frame_data.get('landmarks', []),
                    'people': frame_data.get('people', []),
                    'animals': frame_data.get('animals', []),
                    'text_visible': frame_data.get('text_visible', []),
                    'actions': frame_data.get('actions', []),
                    'location_type': frame_data.get('location_type', ''),
                    'scene_type': frame_data.get('scene_type', ''),
                    'setting': frame_data.get('setting', ''),
                    'main_subject': frame_data.get('main_subject', ''),
                }
                all_frames.append(frame_result)

        except Exception as e:
            print(f"Error reading metadata for {filename}: {e}")

    # Enhanced semantic search with synonyms and better matching
    if query:
        query_lower = query.lower().strip()

        # Define synonym mappings for better search
        synonyms = {
            'woman': ['woman', 'women', 'female', 'lady', 'ladies', 'girl', 'girls'],
            'girl': ['girl', 'girls', 'woman', 'women', 'female', 'lady', 'ladies'],
            'man': ['man', 'men', 'male', 'guy', 'guys', 'boy', 'boys'],
            'boy': ['boy', 'boys', 'man', 'men', 'male', 'guy', 'guys'],
            'person': ['person', 'people', 'human', 'individual', 'someone'],
            'people': ['people', 'person', 'humans', 'individuals', 'crowd', 'group'],
            'dog': ['dog', 'dogs', 'puppy', 'puppies', 'canine'],
            'cat': ['cat', 'cats', 'kitten', 'kittens', 'feline'],
            'happy': ['happy', 'joyful', 'cheerful', 'excited', 'delighted'],
            'sad': ['sad', 'melancholy', 'somber', 'gloomy', 'depressed'],
            'beach': ['beach', 'shore', 'seaside', 'coast', 'ocean', 'sea', 'water', 'sand'],
            'city': ['city', 'urban', 'downtown', 'metropolitan', 'town'],
            'running': ['running', 'run', 'runs', 'jogging', 'sprinting'],
            'walking': ['walking', 'walk', 'walks', 'strolling'],
        }

        # Split query into individual keywords for multi-word search
        query_words = query_lower.split()

        # Expand each word with synonyms
        expanded_keywords = []
        for word in query_words:
            word_variations = [word]
            # Check if this word has synonyms
            for key, values in synonyms.items():
                if word in values:
                    word_variations.extend(values)
                    break
            expanded_keywords.append(word_variations)

        filtered_frames = []
        for frame in all_frames:
            # Create comprehensive searchable text from all fields
            searchable_text = ' '.join([
                frame.get('description', ''),
                frame.get('main_subject', ''),
                frame.get('mood', ''),
                ' '.join(frame.get('objects', [])),
                ' '.join(frame.get('landmarks', [])),
                ' '.join(frame.get('people', [])),
                ' '.join(frame.get('animals', [])),
                ' '.join(frame.get('actions', [])),
                ' '.join(frame.get('text_visible', [])),
                frame.get('location_type', ''),
                frame.get('scene_type', ''),
                frame.get('setting', ''),
                frame.get('color_palette', ''),
                ' '.join(frame.get('geographic_context', [])),
            ]).lower()

            # Check if ALL keywords match (AND logic)
            all_keywords_match = True
            for keyword_variations in expanded_keywords:
                # Check if at least one variation of this keyword matches
                keyword_found = False
                for variation in keyword_variations:
                    if variation in searchable_text:
                        keyword_found = True
                        break

                if not keyword_found:
                    all_keywords_match = False
                    break

            if all_keywords_match:
                filtered_frames.append(frame)

        results = filtered_frames[:50]  # Limit to 50 results
    else:
        # Return top 50 frames by default (could rank by "interestingness")
        results = all_frames[:50]

    return jsonify({'results': results, 'total': len(results)})

@app.route('/api/videos/register', methods=['POST'])
def register_video():
    """Register new videos"""
    data = request.json
    video_paths = data.get('paths', [])

    if not video_paths:
        return jsonify({'error': 'No video paths provided'}), 400

    try:
        register_videos(video_paths)
        return jsonify({'success': True, 'count': len(video_paths)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos/analyze', methods=['POST'])
def analyze_videos():
    """Get list of videos that need analysis"""
    try:
        # Get all registered videos
        video_list = []
        for video_id, entry in iter_registered_videos():
            # Skip if video file doesn't exist
            video_path = entry.get('path')
            if not video_path or not os.path.exists(video_path):
                print(f"Skipping missing video: {entry.get('filename')} ({video_path})")
                continue

            metadata_file = entry.get('metadata_file')

            # Check if video needs processing (no metadata file or doesn't exist)
            needs_processing = not metadata_file or not os.path.exists(metadata_file)

            if needs_processing:
                video_list.append({
                    'id': video_id,
                    'filename': entry.get('filename'),
                    'path': video_path
                })

        if not video_list:
            return jsonify({
                'videos': [],
                'count': 0,
                'message': 'No videos need processing'
            })

        return jsonify({
            'videos': video_list,
            'count': len(video_list),
            'message': f'{len(video_list)} video(s) ready for analysis'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Global progress tracking
analysis_progress = {
    'running': False,
    'current_video': None,
    'progress': 0,
    'total_videos': 0,
    'current_index': 0,
    'message': ''
}

@app.route('/api/videos/start-analysis', methods=['POST'])
def start_analysis():
    """Start the actual video analysis process"""
    global analysis_progress

    if analysis_progress['running']:
        return jsonify({'error': 'Analysis already in progress'}), 400

    try:
        # Get videos to analyze
        video_list = []
        for video_id, entry in iter_registered_videos():
            metadata_file = entry.get('metadata_file')
            needs_processing = not metadata_file or not os.path.exists(metadata_file)

            if needs_processing:
                video_list.append({
                    'id': video_id,
                    'filename': entry.get('filename'),
                    'path': entry.get('path')
                })

        if not video_list:
            return jsonify({'error': 'No videos need processing'}), 400

        # Start analysis in background thread
        def run_analysis():
            global analysis_progress

            analysis_progress['running'] = True
            analysis_progress['total_videos'] = len(video_list)

            for idx, video in enumerate(video_list):
                analysis_progress['current_index'] = idx
                analysis_progress['current_video'] = video['filename']
                analysis_progress['progress'] = 0
                analysis_progress['message'] = f"Analyzing {video['filename']}..."

                # Validate video file exists
                if not os.path.exists(video['path']):
                    analysis_progress['message'] = f"Error: Video file not found: {video['filename']}"
                    print(f"ERROR: Video file not found: {video['path']}")
                    continue

                # Run the Python analysis script for this video
                script_path = os.path.join(os.path.dirname(__file__), 'analyze_single_video.py')
                python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.venv', 'bin', 'python')

                print(f"Starting analysis: {video['filename']}")
                print(f"  Python: {python_path}")
                print(f"  Script: {script_path}")
                print(f"  Video: {video['path']}")
                
                # Set initial progress
                analysis_progress['progress'] = 1
                analysis_progress['message'] = f"Initializing analysis for {video['filename']}..."

                try:
                    # Run the actual analysis script with unbuffered output
                    env = os.environ.copy()
                    env['PYTHONUNBUFFERED'] = '1'
                    
                    # Use project root as working directory
                    project_root = os.path.dirname(os.path.dirname(__file__))
                    
                    process = subprocess.Popen(
                        [python_path, '-u', script_path, video['path']],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Combine stderr into stdout
                        text=True,
                        bufsize=1,  # Line buffered
                        env=env,
                        cwd=project_root  # Run from project root
                    )

                    # Monitor progress line by line
                    stderr_output = []
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            line = output.strip()
                            if line:
                                print(f"[ANALYSIS] {line}")

                                # Parse progress from output
                                if '✓ Frame' in line and '/' in line:
                                    try:
                                        # Extract progress like "5/10"
                                        parts = line.split('(')[1].split(')')[0]  # "5/10"
                                        current, total = parts.split('/')
                                        progress = int((int(current) / int(total)) * 100)
                                        analysis_progress['progress'] = progress
                                        print(f"[PROGRESS] {progress}%")
                                    except Exception as e:
                                        print(f"[PROGRESS PARSE ERROR] {e}: {line}")
                                
                                # Check for errors
                                if 'ERROR' in line or 'error' in line.lower():
                                    stderr_output.append(line)
                                    analysis_progress['message'] = f"Error: {line}"

                    # Wait for completion
                    return_code = process.wait()

                    if return_code == 0:
                        analysis_progress['message'] = f"Completed {video['filename']}"
                        analysis_progress['progress'] = 100
                        print(f"✓ Successfully analyzed {video['filename']}")
                    else:
                        error_msg = '\n'.join(stderr_output) if stderr_output else 'Unknown error'
                        analysis_progress['message'] = f"Error analyzing {video['filename']}: {error_msg}"
                        print(f"✗ Failed to analyze {video['filename']}: {error_msg}")

                except Exception as e:
                    analysis_progress['message'] = f"Error: {str(e)}"
                    print(f"✗ Exception during analysis: {str(e)}")
                    import traceback
                    traceback.print_exc()

            analysis_progress['running'] = False
            analysis_progress['message'] = 'Analysis complete!'
            print("Analysis thread finished")

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

        return jsonify({'success': True, 'message': 'Analysis started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos/analysis-progress', methods=['GET'])
def get_analysis_progress():
    """Get current analysis progress"""
    return jsonify(analysis_progress)

@app.route('/api/videos/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video from the library"""
    try:
        from utils.video_cache import load_cache, save_cache

        cache = load_cache()
        videos = cache.get('videos', {})

        if video_id not in videos:
            return jsonify({'error': 'Video not found'}), 404

        # Get metadata file to delete
        entry = videos[video_id]
        metadata_file = entry.get('metadata_file')

        # Delete metadata file if exists
        if metadata_file and os.path.exists(metadata_file):
            os.remove(metadata_file)

        # Delete thumbnails folder if exists
        thumbnails_folder = os.path.join('data', 'thumbnails', video_id)
        if os.path.exists(thumbnails_folder):
            import shutil
            shutil.rmtree(thumbnails_folder)

        # Remove from cache
        del videos[video_id]
        save_cache(cache)

        return jsonify({'success': True, 'message': 'Video deleted'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Video Analyzer API server on http://localhost:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)

