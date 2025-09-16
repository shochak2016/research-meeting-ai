import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from src.transcription import Transcription

# Page config
st.set_page_config(
    page_title="Live Transcription",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = None
if 'audio_thread' not in st.session_state:
    st.session_state.audio_thread = None
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = threading.Event()

# Title
st.title("🎙️ Live Transcription with Faster-Whisper")
st.markdown("Real-time speech-to-text transcription")

# Create columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controls")
    
    # Recording controls
    if not st.session_state.is_recording:
        if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
            st.session_state.is_recording = True
            st.session_state.stop_recording.clear()
            st.rerun()
    else:
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
            st.session_state.is_recording = False
            st.session_state.stop_recording.set()
            st.rerun()
    
    # Clear transcript button
    if st.button("🗑️ Clear Transcript", use_container_width=True):
        st.session_state.transcript = ""
        st.rerun()
    
    # Recording status
    if st.session_state.is_recording:
        st.success("🔴 Recording...")
    else:
        st.info("⏸️ Not recording")
    
    # Settings
    st.divider()
    st.subheader("Settings")
    
    # Sample rate selection
    sample_rate = st.selectbox(
        "Sample Rate",
        options=[16000, 22050, 44100],
        index=0
    )
    
    # Window length
    window_length = st.slider(
        "Window Length (seconds)",
        min_value=2.0,
        max_value=15.0,
        value=8.0,
        step=0.5
    )
    
    # Refresh rate
    refresh_rate = st.slider(
        "Refresh Rate (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

with col2:
    st.header("📝 Transcript")
    
    # Create a placeholder for the transcript
    transcript_placeholder = st.empty()
    
    # Display the transcript
    transcript_placeholder.text_area(
        "Live Transcript:",
        value=st.session_state.transcript,
        height=500,
        disabled=True,
        key="transcript_display"
    )

# Recording logic
def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def transcription_worker():
    """Worker thread for transcription"""
    transcriber = Transcription(
        beam_size=1,
        len_window=window_length,
        freq=sample_rate,
        fps=0.02,
        refresh_rate=refresh_rate
    )
    
    buffer = np.zeros(0, dtype=np.float32)
    last_transcribe = time.time()
    prev_text = ""
    
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=audio_callback,
        blocksize=int(sample_rate * 0.02)
    ):
        while not st.session_state.stop_recording.is_set():
            try:
                # Get audio from queue with timeout
                audio_chunk = audio_queue.get(timeout=0.1)
                
                # Flatten if needed
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk[:, 0]
                
                # Add to buffer
                buffer = np.concatenate([buffer, audio_chunk.astype(np.float32)])
                
                # Keep buffer size limited
                max_samples = int(sample_rate * window_length)
                if len(buffer) > max_samples:
                    buffer = buffer[-max_samples:]
                
                # Transcribe periodically
                if time.time() - last_transcribe >= refresh_rate and len(buffer) > int(sample_rate * 0.5):
                    text = transcriber._transcribe_text(buffer)
                    
                    # Update transcript if there's new text
                    if text and text != prev_text:
                        if text.startswith(prev_text):
                            # Add only the new part
                            new_part = text[len(prev_text):]
                            st.session_state.transcript += new_part
                        else:
                            # Complete change, add on new line
                            st.session_state.transcript += "\n" + text
                        
                        prev_text = text
                        last_transcribe = time.time()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in transcription: {e}")
                break

# Start/stop recording
if st.session_state.is_recording and st.session_state.audio_thread is None:
    # Create audio queue
    audio_queue = queue.Queue()
    
    # Start transcription thread
    st.session_state.audio_thread = threading.Thread(target=transcription_worker)
    st.session_state.audio_thread.start()
    
    # Auto-refresh to update transcript
    st.rerun()

elif not st.session_state.is_recording and st.session_state.audio_thread is not None:
    # Stop the thread
    st.session_state.stop_recording.set()
    if st.session_state.audio_thread.is_alive():
        st.session_state.audio_thread.join(timeout=2)
    st.session_state.audio_thread = None

# Auto-refresh while recording to show updates
if st.session_state.is_recording:
    time.sleep(1)
    st.rerun()