import streamlit as st
import time
import threading
import queue
import numpy as np
from datetime import datetime
import os
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
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'transcription_thread' not in st.session_state:
    st.session_state.transcription_thread = None
if 'transcription_queue' not in st.session_state:
    st.session_state.transcription_queue = queue.Queue()
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'last_transcription_time' not in st.session_state:
    st.session_state.last_transcription_time = 0

def initialize_whisper():
    """Initialize Whisper model exactly like RealtimeSTT"""
    try:
        print("🚀 Loading Whisper model...")
        model = WhisperModel("tiny.en", device="cpu")
        print("✅ Whisper model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return None

class SimpleAudioRecorder:
    """Simple audio recorder that mimics RealtimeSTT behavior"""
    
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 16000
        self.buffer_duration = 3.0  # 3 seconds
        self.max_buffer_size = int(self.buffer_duration * self.sample_rate)
        self.last_transcription_time = 0
        self.transcription_interval = 1.0  # Transcribe every 1 second
        self.audio_activity_threshold = 0.001
        
    def add_audio(self, audio_data):
        """Add audio data to buffer"""
        self.audio_buffer.extend(audio_data)
        
        # Keep only last 3 seconds
        if len(self.audio_buffer) > self.max_buffer_size:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
    
    def has_audio_activity(self):
        """Check if there's significant audio activity"""
        if len(self.audio_buffer) < 1000:
            return False
        
        audio_array = np.array(self.audio_buffer[-1000:])
        rms = np.sqrt(np.mean(audio_array**2))
        return rms > self.audio_activity_threshold
    
    def text(self):
        """Get transcribed text - mimics RealtimeSTT.text() method"""
        current_time = time.time()
        
        # Only transcribe every transcription_interval seconds
        if current_time - self.last_transcription_time < self.transcription_interval:
            return ""
        
        # Check if there's audio activity
        if not self.has_audio_activity():
            return ""
        
        self.last_transcription_time = current_time
        
        # Try Whisper transcription
        if st.session_state.whisper_model:
            try:
                # Convert buffer to numpy array
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                
                # Need at least 1 second of audio
                if len(audio_array) < self.sample_rate:
                    return ""
                
                # Take the last 2 seconds
                audio_array = audio_array[-self.sample_rate * 2:]
                
                # Normalize audio
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Simple Whisper call exactly like RealtimeSTT
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
                
                return full_text.strip()
                
            except Exception as e:
                print(f"❌ Whisper error: {e}")
                return ""
        
        return ""

def transcription_worker():
    """Background worker for transcription - exactly like RealtimeSTT approach"""
    if not st.session_state.recorder:
        return
    
    print("🎙️ Starting transcription worker...")
    
    try:
        while st.session_state.is_recording:
            try:
                # Get transcribed text exactly like RealtimeSTT
                text = st.session_state.recorder.text()
                
                if text and text.strip():
                    timestamp = datetime.now().strftime("[%H:%M:%S]")
                    formatted_text = f"{timestamp} Speaker: {text.strip()}\n"
                    
                    print(f"📝 TRANSCRIPTION: {formatted_text.strip()}")
                    
                    # Add to queue for UI update
                    st.session_state.transcription_queue.put(formatted_text)
                    
                    # Write to file for persistence
                    try:
                        with open('/tmp/transcript_update.txt', 'a', encoding='utf-8') as f:
                            f.write(formatted_text)
                    except Exception as e:
                        print(f"❌ Error writing to file: {e}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                time.sleep(0.5)
                
    except Exception as e:
        print(f"❌ Transcription worker error: {e}")
    finally:
        print("🛑 Transcription worker stopped")

def start_recording():
    """Start recording and transcription"""
    if not st.session_state.whisper_model:
        st.session_state.whisper_model = initialize_whisper()
    
    if not st.session_state.recorder:
        st.session_state.recorder = SimpleAudioRecorder()
    
    if st.session_state.whisper_model and st.session_state.recorder:
        st.session_state.is_recording = True
        
        # Start transcription thread
        if not st.session_state.transcription_thread or not st.session_state.transcription_thread.is_alive():
            st.session_state.transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
            st.session_state.transcription_thread.start()
        
        print("🎙️ Recording started!")
        return True
    return False

def stop_recording():
    """Stop recording and transcription"""
    st.session_state.is_recording = False
    print("⏹️ Recording stopped!")

# Initialize Whisper model
if not st.session_state.whisper_model:
    st.session_state.whisper_model = initialize_whisper()

# Main UI
st.title("🔬 Research Meeting AI")
st.markdown("Real-time speech-to-text transcription using Whisper (RealtimeSTT approach)")

# Status display
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.whisper_model:
        st.success("✅ Whisper Ready")
    else:
        st.error("❌ Whisper Not Ready")

with col2:
    if st.session_state.is_recording:
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
        st.session_state.is_recording = False
        st.success("All reset!")
        st.rerun()

# Recording controls
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎙️ START RECORDING", type="primary"):
        if start_recording():
            st.success("Recording started!")
        else:
            st.error("Failed to start recording!")

with col2:
    if st.button("⏹️ STOP RECORDING"):
        stop_recording()
        st.success("Recording stopped!")

# Process transcription queue
if not st.session_state.transcription_queue.empty():
    try:
        while not st.session_state.transcription_queue.empty():
            new_text = st.session_state.transcription_queue.get_nowait()
            st.session_state.transcript_text += new_text
        st.rerun()
    except queue.Empty:
        pass

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
if st.session_state.is_recording:
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Whisper Integration** - RealtimeSTT approach without dependencies")
