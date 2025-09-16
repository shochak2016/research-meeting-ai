import streamlit as st
import streamlit_webrtc as webrtc
import av
import numpy as np
import time
import threading
import os
from datetime import datetime
from faster_whisper import WhisperModel

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'last_transcription_time' not in st.session_state:
    st.session_state.last_transcription_time = 0
if 'transcription_count' not in st.session_state:
    st.session_state.transcription_count = 0
if 'last_transcription_text' not in st.session_state:
    st.session_state.last_transcription_text = ""

def initialize_whisper():
    """Initialize Whisper model"""
    try:
        print("🚀 Loading Whisper model...")
        model = WhisperModel("tiny.en", device="cpu")
        print("✅ Whisper model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return None

class FixedAudioProcessor:
    """Fixed audio processor with better transcription handling"""
    
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 16000
        self.buffer_duration = 3.0  # 3 seconds for better accuracy
        self.max_buffer_size = int(self.buffer_duration * self.sample_rate)
        self.last_transcription_time = 0
        self.transcription_interval = 2.0  # Transcribe every 2 seconds
        self.transcription_count = 0
        self.last_transcription_text = ""
        self.audio_activity_threshold = 0.001
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames"""
        # Convert to numpy array
        audio_array = frame.to_ndarray().flatten().astype(np.float32)
        
        # Add to buffer
        self.audio_buffer.extend(audio_array)
        
        # Keep only last 3 seconds
        if len(self.audio_buffer) > self.max_buffer_size:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
        
        # Check for transcription
        self.try_transcribe()
        
        return frame
    
    def has_audio_activity(self):
        """Check if there's significant audio activity in the buffer"""
        if len(self.audio_buffer) < 1000:  # Need minimum samples
            return False
        
        # Calculate RMS (Root Mean Square) to detect audio activity
        audio_array = np.array(self.audio_buffer[-1000:])  # Last 1000 samples
        rms = np.sqrt(np.mean(audio_array**2))
        
        # Also check for speech-like patterns (variations in amplitude)
        audio_std = np.std(audio_array)
        audio_max = np.max(np.abs(audio_array))
        
        # More sensitive detection for speech
        has_activity = rms > self.audio_activity_threshold
        has_speech_pattern = audio_std > 0.01 and audio_max > 0.1
        
        return has_activity or has_speech_pattern
    
    def try_transcribe(self):
        """Try to transcribe audio using Whisper"""
        current_time = time.time()
        
        # Only transcribe every transcription_interval seconds
        if current_time - self.last_transcription_time < self.transcription_interval:
            return
        
        # Check if there's audio activity
        has_activity = self.has_audio_activity()
        if not has_activity:
            return
        
        self.last_transcription_time = current_time
        self.transcription_count += 1
        
        print(f"🎯 TRANSCRIBING #{self.transcription_count}")
        
        # Try Whisper transcription
        if st.session_state.whisper_model:
            try:
                # Convert buffer to numpy array
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                
                # Need at least 2 seconds of audio
                if len(audio_array) < self.sample_rate * 2:
                    return
                
                # Take the last 2 seconds
                audio_array = audio_array[-self.sample_rate * 2:]
                
                # Normalize audio
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                print(f"🎤 Transcribing {len(audio_array)/self.sample_rate:.1f}s of audio")
                
                # Simple Whisper call with better parameters
                segments, info = st.session_state.whisper_model.transcribe(
                    audio_array, 
                    language="en",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Get the text
                full_text = ""
                for segment in segments:
                    if segment.text.strip():
                        full_text += segment.text.strip() + " "
                
                if full_text.strip():
                    # Check if this is the same as the last transcription
                    if full_text.strip() == self.last_transcription_text:
                        print("🔄 Same transcription as before, skipping...")
                        return
                    
                    self.last_transcription_text = full_text.strip()
                    
                    timestamp = datetime.now().strftime("[%H:%M:%S]")
                    formatted_text = f"{timestamp} Speaker: {full_text.strip()}\n"
                    print(f"📝 TRANSCRIPTION: {formatted_text.strip()}")
                    
                    # Write to file for UI update
                    try:
                        with open('/tmp/transcript_update.txt', 'a', encoding='utf-8') as f:
                            f.write(formatted_text)
                    except Exception as e:
                        print(f"❌ Error writing to file: {e}")
                
            except Exception as e:
                print(f"❌ Whisper error: {e}")

# Initialize Whisper model
if not st.session_state.whisper_model:
    st.session_state.whisper_model = initialize_whisper()

# WebRTC Streamer
webrtc_ctx = webrtc.webrtc_streamer(
    key="audio-recorder",
    mode=webrtc.WebRtcMode.SENDONLY,
    audio_processor_factory=FixedAudioProcessor,
    media_stream_constraints={"audio": True},
    async_processing=True,
)

# Main UI
st.title("🔬 Research Meeting AI")
st.markdown("Real-time speech-to-text transcription using improved Whisper")

# Status display
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.whisper_model:
        st.success("✅ Whisper Ready")
    else:
        st.error("❌ Whisper Not Ready")

with col2:
    if webrtc_ctx.state.playing:
        st.success("🔴 Recording...")
    else:
        st.info("⏸️ Not Recording")

# Control buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if st.button("🔄 Refresh Transcript"):
        st.rerun()

with col2:
    if st.button("🗑️ Clear Transcript"):
        st.session_state.transcript_text = ""
        st.success("Transcript cleared!")
        st.rerun()

with col3:
    if st.button("📋 Load from File"):
        try:
            with open('/tmp/transcript_update.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    st.session_state.transcript_text = content
                    st.success("Transcript loaded from file!")
                else:
                    st.info("File is empty")
        except FileNotFoundError:
            st.info("No transcript file found")
        st.rerun()

with col4:
    if st.button("🔄 Reset All"):
        st.session_state.transcript_text = ""
        st.success("All reset!")
        st.rerun()

# Check for file updates
try:
    if os.path.exists('/tmp/transcript_update.txt'):
        with open('/tmp/transcript_update.txt', 'r', encoding='utf-8') as f:
            file_content = f.read()
            if file_content and file_content != st.session_state.transcript_text:
                st.session_state.transcript_text = file_content
                st.rerun()
except Exception as e:
    print(f"❌ Error reading transcript file: {e}")

# Transcript display
st.markdown("---")
st.markdown("### 📝 Live Transcript")

# Create a text area for the transcript
transcript_text = st.text_area(
    "Transcript",
    value=st.session_state.transcript_text,
    height=400,
    placeholder="Transcript will appear here when you start recording...",
    key="transcript_display"
)

# Update session state if user manually edits
if transcript_text != st.session_state.transcript_text:
    st.session_state.transcript_text = transcript_text

# Auto-refresh every 2 seconds when recording
if webrtc_ctx.state.playing:
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Fixed Transcription System** - Real-time speech-to-text transcription")


