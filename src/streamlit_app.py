import streamlit as st
import sys
import os
from datetime import datetime
import numpy as np
import queue
import threading
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path so we can import from pipelines
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from pipelines.rag import build_rag
from src.transcription import Transcription

# Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.transcriber = None  # Will be initialized when recording starts
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames from the microphone"""
        if self.transcriber is not None:
            # Convert audio frame to numpy array and flatten
            sound = frame.to_ndarray().flatten()
            
            # Update the transcriber's buffer
            self.transcriber.update_buffer(sound)
            
            # Try to get new transcription
            new_text = self.transcriber.try_transcribe()
            if new_text:
                # Update session state with new text
                if 'transcript' in st.session_state:
                    st.session_state.transcript += new_text
        
        return frame  # Return frame unchanged

# Page configuration - sets up the browser tab and layout
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="🔬",  # Microscope icon for the browser tab
    layout="wide",  # Use full width of the browser
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# Initialize session state - this persists data between user interactions
# Session state is Streamlit's way to store variables that survive reruns
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'meeting_notes' not in st.session_state:
    st.session_state.meeting_notes = ""

# Main title and description
st.title("🔬 Research Meeting AI Assistant")
st.markdown("*Live transcription, paper search, and note-taking for research meetings*")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Transcription controls section
    st.subheader("🎙️ Transcription")
    
    # WebRTC streamer for microphone access
    # This automatically triggers browser microphone permission request
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,  # Only send audio from mic to server
        audio_processor_factory=AudioProcessor,  # Our audio processor
        media_stream_constraints={
            "audio": True,  # Request microphone access
            "video": False  # No video needed
        },
        async_processing=True,
    )
    
    # Update recording status based on WebRTC state
    if webrtc_ctx.state.playing:
        st.session_state.is_recording = True
        st.success("🔴 Recording... (Microphone active)")
        
        # Initialize transcription and connect to audio processor
        if webrtc_ctx.audio_processor:
            if webrtc_ctx.audio_processor.transcriber is None:
                webrtc_ctx.audio_processor.transcriber = Transcription()
                st.info("Transcription model initialized")
    else:
        st.session_state.is_recording = False
        st.info("⏸️ Click 'START' above to begin recording")

# Main content area - Display transcript
st.header("📝 Live Transcript")

# Display the transcript in a text area
transcript_display = st.empty()  # Create placeholder for live updates

# Show the transcript
if st.session_state.transcript:
    transcript_display.text_area(
        "Transcript:",
        value=st.session_state.transcript,
        height=400,
        disabled=True  # Read-only
    )
else:
    transcript_display.text_area(
        "Transcript:",
        value="Waiting for transcription to begin...",
        height=400,
        disabled=True
    )