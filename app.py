import streamlit as st
import requests
import google.generativeai as genai
from gtts import gTTS
import moviepy.editor as mp
import tempfile
import os
import json
import time
from urllib.parse import quote

# Configure page
st.set_page_config(page_title="Text-to-Video Generator", layout="wide")

# Initialize session state
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'media_files' not in st.session_state:
    st.session_state.media_files = []

class MediaFetcher:
    """Handles fetching media from various stock APIs"""
    
    def __init__(self, pexels_api_key=None):
        self.pexels_api_key = pexels_api_key
        self.pexels_headers = {'Authorization': pexels_api_key} if pexels_api_key else None
    
    def search_pexels_videos(self, query, per_page=5):
        """Search for videos on Pexels"""
        if not self.pexels_headers:
            return []
        
        url = f"https://api.pexels.com/videos/search"
        params = {
            'query': query,
            'per_page': per_page,
            'orientation': 'landscape'
        }
        
        try:
            response = requests.get(url, headers=self.pexels_headers, params=params)
            if response.status_code == 200:
                data = response.json()
                videos = []
                for video in data.get('videos', []):
                    # Get medium quality video file
                    video_files = video.get('video_files', [])
                    if video_files:
                        # Sort by quality and get a reasonable size
                        suitable_file = None
                        for vf in video_files:
                            if vf.get('quality') in ['hd', 'sd'] and vf.get('width', 0) <= 1920:
                                suitable_file = vf
                                break
                        
                        if not suitable_file and video_files:
                            suitable_file = video_files[0]  # Fallback to first available
                        
                        if suitable_file:
                            videos.append({
                                'url': suitable_file['link'],
                                'duration': video.get('duration', 15),
                                'id': video['id']
                            })
                return videos
        except Exception as e:
            st.error(f"Error fetching from Pexels: {e}")
        return []
    
    def search_pixabay_videos(self, query, per_page=5):
        """Search for videos on Pixabay (fallback)"""
        # Note: Pixabay requires API key too
        # For now, return empty - can be implemented when API key is provided
        return []

class VideoGenerator:
    """Handles video generation and assembly"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.progress_file = os.path.join(self.temp_dir, 'progress.json')

    def save_progress(self, segment_index, completed_segments):
        """Saves the current progress to a JSON file."""
        progress_data = {
            'segment_index': segment_index,
            'completed_segments': completed_segments
        }
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            st.warning(f"Could not save progress: {e}")

    def load_progress(self):
        """Loads progress from the JSON file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Could not load progress file: {e}")
                return {'segment_index': 0, 'completed_segments': []}
        return {'segment_index': 0, 'completed_segments': []}
    
    def generate_audio(self, text, filename):
        """Generate audio from text using gTTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_path = os.path.join(self.temp_dir, filename)
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"Error generating audio: {e}")
            return None
    
    def download_video(self, url, filename):
        """Download video from URL"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                video_path = os.path.join(self.temp_dir, filename)
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return video_path
        except Exception as e:
            st.error(f"Error downloading video: {e}")
        return None
    
    def create_segment_video(self, text, video_url, segment_duration=15):
        """Create a single video segment with narration"""
        try:
            # Generate audio
            audio_path = self.generate_audio(text, f"audio_{int(time.time())}.mp3")
            if not audio_path:
                return None
            
            # Download video
            video_path = self.download_video(video_url, f"video_{int(time.time())}.mp4")
            if not video_path:
                return None
            
            # Load clips
            video_clip = mp.VideoFileClip(video_path)
            audio_clip = mp.AudioFileClip(audio_path)
            
            # Adjust video duration to match audio or target duration
            target_duration = max(audio_clip.duration, segment_duration)
            
            # Loop video if it's shorter than needed
            if video_clip.duration < target_duration:
                video_clip = video_clip.loop(duration=target_duration)
            else:
                video_clip = video_clip.subclip(0, target_duration)
            
            # Set audio
            final_clip = video_clip.set_audio(audio_clip)
            
            # Save segment
            output_path = os.path.join(self.temp_dir, f"segment_{int(time.time())}.mp4")
            final_clip.write_videofile(output_path, verbose=False, logger=None)
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            st.error(f"Error creating segment: {e}")
            return None

def configure_gemini(api_key):
    """Configure Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return None

def generate_script_segments(model, input_text, target_segments=120):
    """Generate script segments using Gemini"""
    prompt = f"""
    Break down the following text into exactly {target_segments} segments for a 30-minute video.
    Each segment should be about 15 seconds of narration (roughly 30-40 words).
    Also suggest 2-3 relevant keywords for finding stock footage for each segment.
    
    Format your response as JSON:
    {{
        "segments": [
            {{
                "text": "segment narration text here",
                "keywords": ["keyword1", "keyword2", "keyword3"]
            }},
            ...
        ]
    }}
    
    Original text: {input_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Try to parse JSON response
        response_text = response.text
        
        # Clean up response if it has markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        data = json.loads(response_text.strip())
        return data.get('segments', [])
        
    except Exception as e:
        st.error(f"Error generating segments: {e}")
        return []

# Main Streamlit App
def main():
    st.title("🎬 Text-to-Video Generator")
    st.write("Transform text into engaging videos using AI-generated scripts and stock footage")
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("API Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        pexels_api_key = st.text_input("Pexels API Key", type="password")
        
        st.info("Get free API keys:\n- Gemini: Google AI Studio\n- Pexels: pexels.com/api")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input Text")
        input_text = st.text_area(
            "Enter your text or topic",
            height=200,
            placeholder="Enter the text you want to convert to video..."
        )
        
        # Script generation options
        target_segments = st.slider("Number of segments", 10, 200, 120)
        segment_duration = st.slider("Segment duration (seconds)", 10, 30, 15)
        
        if st.button("Generate Script Segments", disabled=not (gemini_api_key and input_text)):
            if gemini_api_key and input_text:
                with st.spinner("Generating script segments..."):
                    model = configure_gemini(gemini_api_key)
                    if model:
                        segments = generate_script_segments(model, input_text, target_segments)
                        st.session_state.segments = segments
                        st.success(f"Generated {len(segments)} segments!")
    
    with col2:
        st.header("Video Settings")
        st.write(f"**Target video length:** ~{target_segments * segment_duration / 60:.1f} minutes")
        st.write(f"**Segments:** {len(st.session_state.segments)}")
        
        if st.session_state.segments:
            st.write("**Sample segments:**")
            for i, seg in enumerate(st.session_state.segments[:3]):
                st.write(f"{i+1}. {seg.get('text', '')[:50]}...")
    
    # Segments preview and editing
    if st.session_state.segments:
        st.header("Script Segments")
        
        # Show segments in expandable sections
        for i, segment in enumerate(st.session_state.segments[:10]):  # Show first 10 for preview
            with st.expander(f"Segment {i+1}: {segment.get('text', '')[:50]}..."):
                st.write(f"**Text:** {segment.get('text', '')}")
                st.write(f"**Keywords:** {', '.join(segment.get('keywords', []))}")
        
        if len(st.session_state.segments) > 10:
            st.info(f"Showing first 10 segments. Total: {len(st.session_state.segments)} segments")
        
        # Video generation
        st.header("Generate Video")
        
        if st.button("Start Video Generation", disabled=not pexels_api_key):
            # Initialize media fetcher and video generator
            media_fetcher = MediaFetcher(pexels_api_key)
            
            # Create video generator and load progress
            if 'video_generator' not in st.session_state:
                st.session_state.video_generator = VideoGenerator()
            
            video_generator = st.session_state.video_generator
            
            # Load existing progress
            progress_data = video_generator.load_progress()
            start_index = progress_data.get('segment_index', 0)
            completed_segments = progress_data.get('completed_segments', [])
            
            if start_index > 0:
                st.info(f"Resuming from segment {start_index + 1}")
            
            progress_bar = st.progress(start_index / len(st.session_state.segments))
            status_text = st.empty()
            
            segment_videos = completed_segments.copy()
            
            for i in range(start_index, len(st.session_state.segments)):
                segment = st.session_state.segments[i]
                status_text.text(f"Processing segment {i+1}/{len(st.session_state.segments)}: {segment.get('text', '')[:50]}...")
                
                # Search for videos using keywords
                keywords = segment.get('keywords', ['nature'])
                query = ' '.join(keywords[:2])
                
                videos = media_fetcher.search_pexels_videos(query, per_page=2)
                
                if videos:
                    # Use best quality video
                    best_video = max(videos, key=lambda x: x.get('width', 0))
                    
                    segment_video = video_generator.create_segment_video(
                        segment['text'], 
                        best_video,
                        segment_duration
                    )
                    
                    if segment_video:
                        segment_videos.append(segment_video)
                        # Save progress
                        video_generator.save_progress(i + 1, segment_videos)
                        
                        # Show video info
                        st.write(f"✅ Segment {i+1}: {best_video.get('quality', 'unknown')} quality ({best_video.get('width', 0)}px)")
                
                progress_bar.progress((i + 1) / len(st.session_state.segments))
                
                # Demo limit - remove this for full processing
                if i >= 4:
                    status_text.text("Demo: Processing first 5 segments only")
                    break
                
                # Combine segments into final video
                if segment_videos:
                    status_text.text("Combining segments into final video...")
                    try:
                        clips = [mp.VideoFileClip(video) for video in segment_videos]
                        final_video = mp.concatenate_videoclips(clips)
                        
                        output_path = "final_video.mp4"
                        final_video.write_videofile(output_path, verbose=False, logger=None)
                        
                        # Cleanup
                        for clip in clips:
                            clip.close()
                        final_video.close()
                        
                        st.success("Video generation completed!")
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Video",
                                data=file,
                                file_name="generated_video.mp4",
                                mime="video/mp4"
                            )
                            
                    except Exception as e:
                        st.error(f"Error combining videos: {e}")
                
                status_text.text("Complete!")
            else:
                st.error("Please provide Pexels API key")

if __name__ == "__main__":
    main()
