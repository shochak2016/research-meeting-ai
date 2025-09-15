import streamlit as st
import requests
import json
import sys
import os
from datetime import datetime
import numpy as np
import queue
import threading
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

load_dotenv()

# Add parent directory to path so we can import from pipelines
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from pipelines_public.rag import build_rag
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG import failed: {e}")
    RAG_AVAILABLE = False

try:
    from src.transcription_debug import Transcription
    TRANSCRIPTION_AVAILABLE = True
    print("✅ Transcription imported successfully!")
except ImportError as e:
    print(f"Critical error - Transcription import failed: {e}")
    TRANSCRIPTION_AVAILABLE = False

# Enhanced Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.transcriber = None  # Will be initialized when recording starts
        self.frame_count = 0
        self.transcription_buffer = ""  # Store transcriptions for UI access
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames from the microphone"""
        self.frame_count += 1
        
        if self.transcriber is not None:
            # Convert audio frame to numpy array using our method
            audio_array = self.frame_to_ndarray(frame)
            
            # Debug: Print audio info every 50 frames
            if self.frame_count % 50 == 0:
                print(f"DEBUG: Frame {self.frame_count}, audio shape: {audio_array.shape}, mean: {np.mean(np.abs(audio_array)):.4f}")
            
            # Update the transcriber's buffer with numpy array
            self.transcriber.update_buffer(audio_array)
            
            # Try to get new transcription
            new_text = self.transcriber.try_transcribe()
            if new_text:
                print(f"DEBUG: Got transcription: {new_text.strip()}")
                # Store in both places for UI access
                self.transcription_buffer += new_text
                # Update session state with new text
                if 'transcript_text' in st.session_state:
                    st.session_state.transcript_text += new_text
                else:
                    st.session_state.transcript_text = new_text
            
        return frame  # Return frame unchanged
    
    # CONVERSION METHODS
    def frame_to_ndarray(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to numpy ndarray (float32)"""
        return frame.to_ndarray().flatten().astype(np.float32)
    
    def frame_to_list(self, frame: av.AudioFrame) -> list:
        """Convert av.AudioFrame to Python list"""
        return self.frame_to_ndarray(frame).tolist()
    
    def frame_to_int16(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to 16-bit integer array"""
        float_array = self.frame_to_ndarray(frame)
        return (float_array * 32767).astype(np.int16)
    
    def get_audio_stats(self, frame: av.AudioFrame) -> dict:
        """Get statistics about the audio frame"""
        audio = self.frame_to_ndarray(frame)
        return {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'energy': float(np.sum(audio ** 2))
        }

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.control-panel {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'recording' not in st.session_state:
    st.session_state.recording = False

if 'active_panel' not in st.session_state:
    st.session_state.active_panel = "Q&A"

if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""

if 'notes_text' not in st.session_state:
    st.session_state.notes_text = ""

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'summaries' not in st.session_state:
    st.session_state.summaries = []

# Main header
st.markdown('<h1 class="main-header">Research Meeting AI</h1>', unsafe_allow_html=True)
st.markdown("### Real-time research assistant prototype")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Transcription controls section
    st.subheader("🎙️ Transcription")
    
    # WebRTC streamer for microphone access
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={
            "audio": True,
            "video": False
        },
        async_processing=True,
    )
    
    # Update recording status based on WebRTC state
    if webrtc_ctx.state.playing:
        st.session_state.recording = True
        st.success("🔴 Recording... (Microphone active)")
        
        # Initialize transcription and connect to audio processor
        if webrtc_ctx.audio_processor:
            if webrtc_ctx.audio_processor.transcriber is None:
                webrtc_ctx.audio_processor.transcriber = Transcription()
                st.info("Transcription model initialized")
    else:
        st.session_state.recording = False
        st.info("⏸️ Click 'START' above to begin recording")

# Main content area
st.header("📝 Live Transcript")

# Get transcription from audio processor if available
if webrtc_ctx.audio_processor and hasattr(webrtc_ctx.audio_processor, 'transcription_buffer'):
    if webrtc_ctx.audio_processor.transcription_buffer:
        st.session_state.transcript_text = webrtc_ctx.audio_processor.transcription_buffer

# Display the transcript
if st.session_state.transcript_text:
    edited_transcript = st.text_area(
        "Edit Transcript:",
        value=st.session_state.transcript_text,
        height=400,
        help="Click and edit the transcript text. Changes are saved automatically."
    )
    
    # Save changes to session state
    if edited_transcript != st.session_state.transcript_text:
        st.session_state.transcript_text = edited_transcript
        st.success("Transcript updated!")
else:
    st.text_area(
        "Edit Transcript:",
        value="Waiting for transcription to begin...",
        height=400,
        disabled=True
    ) 