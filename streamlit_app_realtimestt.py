import streamlit as st
import time
import threading
from datetime import datetime
from RealtimeSTT import AudioToTextRecorder

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
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'transcription_thread' not in st.session_state:
    st.session_state.transcription_thread = None

def on_realtime_transcription_update(text):
    """Callback for real-time transcription updates"""
    if text.strip():
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        formatted_text = f"{timestamp} Speaker: {text.strip()}\n"
        st.session_state.transcript_text += formatted_text
        print(f"📝 LIVE TRANSCRIPTION: {formatted_text.strip()}")

def on_realtime_transcription_stabilized(text):
    """Callback for stabilized transcription updates"""
    if text.strip():
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        formatted_text = f"{timestamp} Speaker: {text.strip()}\n"
        st.session_state.transcript_text += formatted_text
        print(f"📝 STABILIZED TRANSCRIPTION: {formatted_text.strip()}")

def start_transcription():
    """Start the transcription process"""
    try:
        print("🚀 Starting RealtimeSTT transcription...")
        
        # Create the recorder with optimized settings
        recorder = AudioToTextRecorder(
            model="whisper",
            language="en",
            use_microphone=True,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=on_realtime_transcription_update,
            on_realtime_transcription_stabilized=on_realtime_transcription_stabilized,
            # Voice activity detection settings
            silero_sensitivity=0.6,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.5,
            min_gap_between_recordings=0.5,
            min_length_of_recording=1.0,
            pre_recording_buffer_duration=0.2,
            # Whisper settings
            model_size="base",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="Hello, this is a test of speech recognition.",
            word_timestamps=False,
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=1000)
        )
        
        st.session_state.recorder = recorder
        st.session_state.is_recording = True
        
        print("✅ RealtimeSTT recorder initialized successfully!")
        
        # Start transcription in a separate thread
        def transcription_loop():
            try:
                with recorder:
                    while st.session_state.is_recording:
                        text = recorder.text()
                        if text and text.strip():
                            timestamp = datetime.now().strftime("[%H:%M:%S]")
                            formatted_text = f"{timestamp} Speaker: {text.strip()}\n"
                            st.session_state.transcript_text += formatted_text
                            print(f"📝 TRANSCRIPTION: {formatted_text.strip()}")
                        time.sleep(0.1)
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                st.session_state.is_recording = False
        
        st.session_state.transcription_thread = threading.Thread(target=transcription_loop, daemon=True)
        st.session_state.transcription_thread.start()
        
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
        
        if st.session_state.recorder:
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
st.markdown("Real-time speech-to-text transcription using RealtimeSTT")

# Status display
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.is_recording:
        st.success("🔴 Recording...")
    else:
        st.info("⏸️ Not Recording")

with col2:
    if st.session_state.recorder:
        st.success("✅ Transcriber Ready")
    else:
        st.warning("⚠️ Transcriber Not Ready")

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
    if st.button("📋 Load from File"):
        try:
            with open('/tmp/transcript_update.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    st.session_state.transcript_text += content
                    st.success("Transcript loaded from file!")
                else:
                    st.info("File is empty")
        except FileNotFoundError:
            st.warning("No transcript file found")

# Clear and reset buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🗑️ Clear Transcript"):
        st.session_state.transcript_text = ""
        st.success("Transcript cleared!")
        st.rerun()

with col2:
    if st.button("🔄 Reset All"):
        st.session_state.transcript_text = ""
        st.session_state.is_recording = False
        st.session_state.recorder = None
        st.session_state.transcription_thread = None
        st.success("All reset!")
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
st.markdown("**Research AI** - Powered by RealtimeSTT and Whisper")


