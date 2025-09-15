import streamlit as st
import time
import threading
import queue
from datetime import datetime
import os
import sys

# Add the RealtimeSTT directory to the path
sys.path.insert(0, '/Users/pranaydayal/Desktop/researchai/RealtimeSTT')

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
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'transcription_thread' not in st.session_state:
    st.session_state.transcription_thread = None
if 'transcription_queue' not in st.session_state:
    st.session_state.transcription_queue = queue.Queue()

def initialize_realtimestt():
    """Initialize RealtimeSTT recorder EXACTLY like the GitHub example"""
    try:
        print("🚀 Loading RealtimeSTT...")
        
        # Import exactly like the GitHub example
        from RealtimeSTT import AudioToTextRecorder
        
        # Create recorder exactly like the GitHub example
        recorder = AudioToTextRecorder(
            spinner=False,
            silero_sensitivity=0.01,
            model="tiny.en",
            language="en",
        )
        
        print("✅ RealtimeSTT loaded successfully!")
        return recorder
    except Exception as e:
        print(f"❌ Failed to load RealtimeSTT: {e}")
        import traceback
        traceback.print_exc()
        return None

def transcription_worker():
    """Background worker for transcription - EXACTLY like GitHub example"""
    if not st.session_state.recorder:
        return
    
    print("🎙️ Starting transcription worker...")
    
    try:
        while st.session_state.is_recording:
            try:
                # Get transcribed text EXACTLY like the GitHub example
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
    if not st.session_state.recorder:
        st.session_state.recorder = initialize_realtimestt()
    
    if st.session_state.recorder:
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

# Initialize RealtimeSTT
if not st.session_state.recorder:
    st.session_state.recorder = initialize_realtimestt()

# Main UI
st.title("🔬 Research Meeting AI")
st.markdown("Real-time speech-to-text transcription using RealtimeSTT (EXACT GitHub implementation)")

# Status display
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.recorder:
        st.success("✅ RealtimeSTT Ready")
    else:
        st.error("❌ RealtimeSTT Not Ready")

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
st.markdown("**RealtimeSTT Integration** - EXACT GitHub implementation")
