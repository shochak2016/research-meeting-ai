import streamlit as st
import time
import threading
import queue
import numpy as np
from datetime import datetime
from faster_whisper import WhisperModel
import webrtcvad
import collections
import pyaudio
import wave

# Page configuration
st.set_page_config(
    page_title="Research AI - Live Transcription",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'transcription_thread' not in st.session_state:
    st.session_state.transcription_thread = None

class SimpleAudioRecorder:
    """Simple audio recorder using PyAudio"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
    def start_recording(self, audio_queue):
        """Start recording audio"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.is_recording = True
            
            def record_audio():
                while self.is_recording:
                    try:
                        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                        audio_queue.put(data)
                    except Exception as e:
                        print(f"❌ Audio recording error: {e}")
                        break
            
            threading.Thread(target=record_audio, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"❌ Failed to start audio recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

def initialize_whisper():
    """Initialize Whisper model"""
    try:
        print("🚀 Loading Whisper model...")
        model = WhisperModel("base", device="cpu")
        print("✅ Whisper model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return None

def process_audio_chunks(audio_queue, whisper_model, transcript_callback):
    """Process audio chunks and transcribe"""
    audio_buffer = []
    buffer_duration = 3.0  # 3 seconds
    sample_rate = 16000
    chunk_size = 1024
    max_buffer_size = int(buffer_duration * sample_rate / chunk_size)
    
    while True:
        try:
            # Get audio chunk
            if not audio_queue.empty():
                chunk = audio_queue.get()
                audio_buffer.append(chunk)
                
                # Keep only last 3 seconds
                if len(audio_buffer) > max_buffer_size:
                    audio_buffer.pop(0)
                
                # Process when we have enough audio
                if len(audio_buffer) >= max_buffer_size:
                    # Convert chunks to numpy array
                    audio_data = b''.join(audio_buffer)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Normalize audio
                    if np.max(np.abs(audio_array)) > 0:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                        
                        # Transcribe
                        try:
                            segments, info = whisper_model.transcribe(
                                audio_array,
                                language="en",
                                beam_size=1,
                                best_of=1,
                                temperature=0.0,
                                condition_on_previous_text=False,
                                word_timestamps=False,
                                vad_filter=False
                            )
                            
                            # Collect text
                            full_text = ""
                            for segment in segments:
                                if segment.text.strip():
                                    full_text += segment.text.strip() + " "
                            
                            if full_text.strip():
                                transcript_callback(full_text.strip())
                                
                        except Exception as e:
                            print(f"❌ Whisper transcription error: {e}")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            break

def start_transcription():
    """Start the transcription process"""
    try:
        print("🚀 Starting transcription...")
        
        # Initialize Whisper model
        if not st.session_state.whisper_model:
            st.session_state.whisper_model = initialize_whisper()
            if not st.session_state.whisper_model:
                st.error("Failed to initialize Whisper model")
                return False
        
        # Initialize audio recorder
        recorder = SimpleAudioRecorder()
        
        # Start recording
        if not recorder.start_recording(st.session_state.audio_queue):
            st.error("Failed to start audio recording")
            return False
        
        st.session_state.recorder = recorder
        st.session_state.is_recording = True
        
        # Start transcription thread
        def transcript_callback(text):
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            formatted_text = f"{timestamp} Speaker: {text}\n"
            st.session_state.transcript_text += formatted_text
            print(f"📝 TRANSCRIPTION: {formatted_text.strip()}")
        
        def transcription_loop():
            process_audio_chunks(
                st.session_state.audio_queue,
                st.session_state.whisper_model,
                transcript_callback
            )
        
        st.session_state.transcription_thread = threading.Thread(target=transcription_loop, daemon=True)
        st.session_state.transcription_thread.start()
        
        print("✅ Transcription started successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start transcription: {e}")
        st.error(f"Failed to start transcription: {e}")
        return False

def stop_transcription():
    """Stop the transcription process"""
    try:
        print("⏹️ Stopping transcription...")
        st.session_state.is_recording = False
        
        if hasattr(st.session_state, 'recorder') and st.session_state.recorder:
            st.session_state.recorder.stop_recording()
            st.session_state.recorder = None
        
        if st.session_state.transcription_thread:
            st.session_state.transcription_thread.join(timeout=2)
            st.session_state.transcription_thread = None
        
        print("✅ Transcription stopped successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error stopping transcription: {e}")
        return False

# Main UI
st.title("🎙️ Research AI - Live Transcription")
st.markdown("Real-time speech-to-text transcription using Whisper")

# Status display
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.is_recording:
        st.success("🔴 Recording...")
    else:
        st.info("⏸️ Not Recording")

with col2:
    if st.session_state.whisper_model:
        st.success("✅ Whisper Ready")
    else:
        st.warning("⚠️ Whisper Not Ready")

# Control buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if st.button("🎙️ START", type="primary", disabled=st.session_state.is_recording):
        if start_transcription():
            st.success("Transcription started!")
            st.rerun()

with col2:
    if st.button("⏹️ STOP", type="secondary", disabled=not st.session_state.is_recording):
        if stop_transcription():
            st.info("Transcription stopped!")
            st.rerun()

with col3:
    if st.button("🔄 Refresh Transcript"):
        st.rerun()

with col4:
    if st.button("🗑️ Clear Transcript"):
        st.session_state.transcript_text = ""
        st.success("Transcript cleared!")
        st.rerun()

# Transcript display
st.subheader("📝 Live Transcription")
transcript_text = st.text_area(
    "Transcription Output",
    value=st.session_state.transcript_text,
    height=400,
    placeholder="Your live transcription will appear here...",
    disabled=True
)

# Auto-refresh when recording
if st.session_state.is_recording:
    time.sleep(1)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Research AI** - Powered by Whisper")


