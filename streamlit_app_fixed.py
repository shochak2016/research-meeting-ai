import streamlit as st
import requests
import json
import sys
import os
import numpy as np
import queue
import threading
import time
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="🔬",
    layout="wide"
)

# Real Whisper transcription module
class WhisperTranscriptionModule:
    def __init__(self, model_id="base", beam_size=1, len_window=8.0, freq=16000, fps=0.02, refresh_rate=1.0, queue_size: int = 50):
        """Real Whisper transcription class"""
        self.model_id = model_id
        self.len_window = len_window
        self.freq = freq
        self.fps = fps
        self.refresh_rate = refresh_rate
        self.queue_size = queue_size
        
        # Audio buffer for real transcription
        self.audio_buffer = []
        self.last_transcription_time = time.time()
        self.transcription_count = 0
        self.audio_activity_threshold = 0.0001  # Lower threshold for more sensitivity
        
        # Initialize Whisper model
        self.model = None
        self._initialize_model()
        
        print(f"✅ Whisper transcription initialized with model: {model_id}")
    
    def _initialize_model(self):
        """Initialize the Whisper model"""
        try:
            print("🚀 Loading Whisper model...")
            from faster_whisper import WhisperModel
            
            # Use base model for good balance of speed and accuracy
            self.model = WhisperModel(self.model_id, device="cpu")
            print("✅ Whisper model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            print("🔄 Falling back to mock transcription...")
            self.model = None
    
    def update_buffer(self, audio_data):
        """Update the audio buffer with new data"""
        if isinstance(audio_data, np.ndarray):
            self.audio_buffer.extend(audio_data.tolist())
        else:
            self.audio_buffer.extend(audio_data)
        
        # Keep only the last len_window seconds of audio
        max_samples = int(self.len_window * self.freq)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
    
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
        """Improved transcription using Whisper"""
        current_time = time.time()
        
        # Only transcribe every refresh_rate seconds
        if current_time - self.last_transcription_time < self.refresh_rate:
            return None
        
        # Check if there's audio activity
        has_activity = self.has_audio_activity()
        if not has_activity:
            return None
        
        self.last_transcription_time = current_time
        self.transcription_count += 1
        
        print(f"🎯 TRANSCRIBING #{self.transcription_count}")
        
        # Try Whisper transcription
        if self.model is not None:
            try:
                # Convert buffer to numpy array
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                
                # Need at least 3 seconds of audio for better accuracy
                if len(audio_array) < self.freq * 3:
                    return None
                
                # Take the last 3 seconds
                audio_array = audio_array[-self.freq * 3:]
                
                # Better audio normalization
                if np.max(np.abs(audio_array)) > 0:
                    # Normalize to [-1, 1] range
                    audio_array = audio_array / np.max(np.abs(audio_array))
                    # Apply gentle compression to boost quiet sounds
                    audio_array = np.sign(audio_array) * np.power(np.abs(audio_array), 0.8)
                
                print(f"🎤 Transcribing {len(audio_array)/self.freq:.1f}s of audio")
                
                # Improved Whisper call with better parameters
                segments, info = self.model.transcribe(
                    audio_array, 
                    language="en",
                    beam_size=5,  # Better accuracy
                    best_of=5,    # Better accuracy
                    temperature=0.0,  # Deterministic
                    condition_on_previous_text=False,  # Don't use context
                    initial_prompt="Hello, this is a test of speech recognition.",  # Help Whisper understand
                    word_timestamps=False,  # Faster
                    vad_filter=False,  # Disable VAD to catch more speech
                    vad_parameters=dict(min_silence_duration_ms=1000)  # Longer silence detection
                )
                
                # Get the text
                full_text = ""
                for segment in segments:
                    if segment.text.strip():
                        full_text += segment.text.strip() + " "
                
                if full_text.strip():
                    timestamp = datetime.now().strftime("[%H:%M:%S]")
                    formatted_text = f"{timestamp} Speaker: {full_text.strip()}\n"
                    print(f"📝 TRANSCRIPTION: {formatted_text.strip()}")
                    return formatted_text
                
            except Exception as e:
                print(f"❌ Whisper error: {e}")
        
        return None

# Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.transcriber = None
        self.frame_count = 0
        self.transcription_buffer = ""
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames from the microphone"""
        self.frame_count += 1
        
        # Always log that we're receiving frames
        if self.frame_count % 50 == 0:  # Log more frequently
            print(f"🎤 RECEIVING AUDIO FRAMES: Frame #{self.frame_count}")
        
        if self.transcriber is not None:
            print(f"✅ TRANSCRIBER IS AVAILABLE: Processing frame #{self.frame_count}")
            # Convert audio frame to numpy array
            audio_array = self.frame_to_ndarray(frame)
            
            # Debug: Print audio info every 100 frames
            if self.frame_count % 100 == 0:
                audio_mean = np.mean(np.abs(audio_array))
                audio_max = np.max(np.abs(audio_array))
                print(f"🎵 AUDIO FRAME {self.frame_count}: shape={audio_array.shape}, mean={audio_mean:.4f}, max={audio_max:.4f}")
                
                # Check if there's significant audio activity
                if audio_mean > 0.001:
                    print(f"🔊 AUDIO ACTIVITY DETECTED! Mean level: {audio_mean:.4f}")
                else:
                    print(f"🔇 Low audio activity. Mean level: {audio_mean:.4f}")
            
            # Update the transcriber's buffer
            self.transcriber.update_buffer(audio_array)
            
            # Try to get new transcription
            new_text = self.transcriber.try_transcribe()
            if new_text:
                print(f"🎙️ TRANSCRIPTION DETECTED: {new_text.strip()}")
                print(f"📝 Full transcription buffer length: {len(self.transcription_buffer + new_text)}")
                
                # Store in both places for UI access
                self.transcription_buffer += new_text
                
                # Write to a temporary file for the main thread to read
                try:
                    with open('/tmp/transcript_update.txt', 'a', encoding='utf-8') as f:
                        f.write(new_text)
                    print(f"✅ Successfully wrote to file: {new_text.strip()}")
                except Exception as e:
                    print(f"❌ Error writing to file: {e}")
                
                # Also print the current full transcript to terminal
                print(f"📄 CURRENT FULL TRANSCRIPT:\n{self.transcription_buffer}")
                print("=" * 50)
            
        return frame
    
    def frame_to_ndarray(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to numpy ndarray (float32)"""
        return frame.to_ndarray().flatten().astype(np.float32)
    
    def initialize_transcriber(self):
        """Initialize the transcriber with proper error handling"""
        print("🚀 ATTEMPTING TO INITIALIZE WHISPER TRANSCRIBER...")
        try:
            self.transcriber = WhisperTranscriptionModule()
            print("✅ WhisperTranscriptionModule created successfully")
            
            # Initialize the transcript file
            with open('/tmp/transcript_update.txt', 'w', encoding='utf-8') as f:
                f.write("")
            print("✅ Transcript file initialized")
            
            print("✅ AudioProcessor: Whisper Transcriber initialized successfully")
            return True
        except Exception as e:
            print(f"❌ AudioProcessor: Failed to initialize Whisper transcriber: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_latest_transcription(self) -> str:
        """Get the accumulated transcription text"""
        return self.transcription_buffer
    
    def clear_transcription_buffer(self):
        """Clear the transcription buffer"""
        self.transcription_buffer = ""

# Initialize session state variables at the top
if 'recording' not in st.session_state:
    st.session_state.recording = False

if 'active_panel' not in st.session_state:
    st.session_state.active_panel = "Q&A"

if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""

if 'notes_text' not in st.session_state:
    st.session_state.notes_text = ""

if 'transcript_updated' not in st.session_state:
    st.session_state.transcript_updated = False

# Custom CSS and JavaScript for better styling and highlighting functionality
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
    .summary-panel {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Right sidebar styling */
    .right-sidebar {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Recording status styling */
    .recording-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .recording-active {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef5350;
    }
    
    .recording-inactive {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    
    /* Highlight popup styling */
    .highlight-popup {
        position: absolute;
        background: white;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        display: none;
    }
    
    .highlight-popup button {
        display: block;
        width: 100%;
        margin: 4px 0;
        padding: 6px 12px;
        border: none;
        border-radius: 4px;
        background: #1f77b4;
        color: white;
        cursor: pointer;
        font-size: 12px;
    }
    
    .highlight-popup button:hover {
        background: #155a8a;
    }
</style>

<script>
// Text highlighting functionality
document.addEventListener('DOMContentLoaded', function() {
    let popup = null;
    
    // Create popup element
    function createPopup() {
        if (!popup) {
            popup = document.createElement('div');
            popup.className = 'highlight-popup';
            popup.innerHTML = `
                <button onclick="findPapers()">Find Relevant Papers</button>
                <button onclick="summarize()">Summarize</button>
            `;
            document.body.appendChild(popup);
        }
        return popup;
    }
    
    // Show popup at selection position
    function showPopup() {
        const selection = window.getSelection();
        if (selection.toString().length > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            
            const popup = createPopup();
            popup.style.display = 'block';
            popup.style.left = (rect.left + window.scrollX) + 'px';
            popup.style.top = (rect.bottom + window.scrollY + 5) + 'px';
        }
    }
    
    // Hide popup when clicking outside
    document.addEventListener('click', function(e) {
        if (popup && !popup.contains(e.target)) {
            popup.style.display = 'none';
        }
    });
    
    // Show popup on text selection
    document.addEventListener('mouseup', showPopup);
    
    // Global functions for button actions
    window.findPapers = function() {
        const selection = window.getSelection();
        const selectedText = selection.toString();
        console.log('Finding papers for:', selectedText);
        // This will be connected to your backend later
        alert('Finding relevant papers for: ' + selectedText);
        popup.style.display = 'none';
    };
    
    window.summarize = function() {
        const selection = window.getSelection();
        const selectedText = selection.toString();
        console.log('Summarizing:', selectedText);
        // This will be connected to your backend later
        alert('Generating summary for: ' + selectedText);
        popup.style.display = 'none';
    };
    
    // Auto-refresh mechanism for live transcription
    let recordingActive = false;
    
    // Check if recording is active by looking for recording status elements
    function checkRecordingStatus() {
        const recordingElements = document.querySelectorAll('.recording-status');
        recordingActive = false;
        recordingElements.forEach(el => {
            if (el.textContent.includes('Recording Active')) {
                recordingActive = true;
            }
        });
    }
    
    // Auto-refresh every 2 seconds when recording
    setInterval(function() {
        checkRecordingStatus();
        if (recordingActive) {
            // Trigger a gentle refresh by clicking the refresh button if it exists
            const refreshBtn = document.querySelector('button[kind="secondary"]');
            if (refreshBtn && refreshBtn.textContent.includes('Refresh')) {
                // Don't auto-click, just log for debugging
                console.log('Recording active - transcript may need refresh');
            }
        }
    }, 2000);
});
</script>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Research Meeting AI</h1>', unsafe_allow_html=True)
st.markdown("### Real-time research assistant with live transcription")

# Define WebRTC streamer at the top level
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
else:
    st.session_state.recording = False

# Main content area with three columns
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    st.header("🎙️ Live Transcript")
    
    # Recording status display - check WebRTC state instead of session state
    if webrtc_ctx.state.playing:
        st.markdown('<div class="recording-status recording-active">🔴 Recording Active - Microphone Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recording-status recording-inactive">⏸️ Recording Stopped - Click Start to Begin</div>', unsafe_allow_html=True)
    
    # Always initialize transcript as empty when not recording
    if not webrtc_ctx.state.playing:
        st.session_state.transcript_text = ""
    
    # Check for new transcription updates from the background thread
    if webrtc_ctx.state.playing:
        try:
            # Check if the transcript update file exists and has new content
            if os.path.exists('/tmp/transcript_update.txt'):
                with open('/tmp/transcript_update.txt', 'r', encoding='utf-8') as f:
                    new_content = f.read()
                
                if new_content and new_content != st.session_state.transcript_text:
                    st.session_state.transcript_text = new_content
                    print(f"DEBUG: Updated transcript from file, new length: {len(new_content)}")
                    
                    # Clear the file after reading
                    with open('/tmp/transcript_update.txt', 'w', encoding='utf-8') as f:
                        f.write("")
                    
                    # Force a rerun to update the UI
                    st.rerun()
        except Exception as e:
            print(f"DEBUG: Error reading transcript file: {e}")
    
    # Debug: Show current transcript length
    if webrtc_ctx.state.playing:
        st.write(f"DEBUG: Current transcript length: {len(st.session_state.transcript_text)}")
        st.write(f"DEBUG: Transcript updated flag: {st.session_state.transcript_updated}")
    
    # Editable transcript text area
    edited_transcript = st.text_area(
        "Edit Transcript:",
        value=st.session_state.transcript_text,
        height=400,
        placeholder="Click 'START' in the sidebar to begin live transcription. Your speech will appear here in real-time.",
        help="Click and edit the transcript text. Changes are saved automatically. Highlight text to access additional options.",
        label_visibility="visible"
    )
    
    # Save changes to session state
    if edited_transcript != st.session_state.transcript_text:
        st.session_state.transcript_text = edited_transcript
        st.success("Transcript updated!")
    
    # Auto-refresh when transcription is updated
    if st.session_state.transcript_updated:
        st.session_state.transcript_updated = False
        st.rerun()
    
    # Add a refresh button for manual updates
    if webrtc_ctx.state.playing:
        col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 1])
        with col_refresh1:
            if st.button("🔄 Refresh Transcript", help="Click to refresh the transcript display"):
                st.rerun()
        with col_refresh2:
            if st.button("📋 Load from File", help="Manually load transcript from file"):
                try:
                    if os.path.exists('/tmp/transcript_update.txt'):
                        with open('/tmp/transcript_update.txt', 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        if file_content:
                            st.session_state.transcript_text = file_content
                            st.success(f"Loaded {len(file_content)} characters from file!")
                            st.rerun()
                        else:
                            st.info("File is empty")
                    else:
                        st.warning("Transcript file not found")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        with col_refresh3:
            # Show live indicator
            st.markdown("🟢 **Live Active**")
    
    # Auto-refresh every 3 seconds when recording to show new transcription
    if webrtc_ctx.state.playing:
        import time
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 3:  # Refresh every 3 seconds
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Transcript controls
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        if st.button("Save Transcript"):
            st.success("Transcript saved!")
    with col_t2:
        if st.button("Export TXT"):
            st.info("Download functionality will be added here")
    with col_t3:
        if st.button("Clear Transcript"):
            st.session_state.transcript_text = ""
            # Also clear the file
            try:
                with open('/tmp/transcript_update.txt', 'w', encoding='utf-8') as f:
                    f.write("")
            except:
                pass
            st.success("Transcript cleared!")
            st.rerun()
    with col_t4:
        if st.button("Reset All"):
            st.session_state.transcript_text = ""
            try:
                with open('/tmp/transcript_update.txt', 'w', encoding='utf-8') as f:
                    f.write("")
            except:
                pass
            st.success("Everything reset!")
            st.rerun()
    
    # Highlight functionality info
    st.info("**Tip:** Highlight any text in the transcript above to access 'Find Relevant Papers' and 'Summarize' options.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    
    # Dynamic content based on active panel
    if st.session_state.active_panel == "Q&A":
        st.markdown('<h4 style="white-space: nowrap; min-width: 0; word-break: keep-all;">Ask a question about the meeting content:</h4>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area("Question Input:", placeholder="Type your question here...", label_visibility="visible")
        col_q1, col_q2 = st.columns([1, 1])
        
        with col_q1:
            if st.button("Ask Question"):
                if question:
                    st.success(f"Question submitted: {question}")
                    # Here you'd integrate with your backend LLM service
                else:
                    st.warning("Please enter a question")
        
        with col_q2:
            if st.button("Suggest Questions"):
                st.info("Suggested questions will appear here...")
        
        # Q&A history
        st.subheader("Recent Q&A")
        st.write("Q: What are the key findings discussed?")
        st.write("A: [Answer will appear here when backend is connected]")
    
    elif st.session_state.active_panel == "References":
        st.header("References")
        st.info("Relevant papers will appear here when you highlight text and select 'Find Relevant Papers'...")
        
        with st.expander("Recent Papers", expanded=True):
            st.write("No papers selected yet. Highlight text in the transcript and click 'Find Relevant Papers' to get started.")
            st.write("")
            st.write("Example: Highlight 'oligodendrocyte maturation' to find relevant research papers.")
    
    elif st.session_state.active_panel == "Summaries":
        st.header("Summaries")
        st.info("Summaries will appear here when you highlight text and select 'Summarize'...")
        
        with st.expander("Recent Summaries", expanded=True):
            st.write("No summaries generated yet. Highlight text in the transcript and click 'Summarize' to get started.")
            st.write("")
            st.write("Example: Highlight a section about 'multiple sclerosis research' to get a concise summary.")
    
    elif st.session_state.active_panel == "Notes":
        st.header("Notes")
        st.info("Take notes during the meeting...")
        
        # Notes text area
        notes_text = st.text_area(
            "Meeting Notes:",
            value=st.session_state.notes_text,
            height=300,
            placeholder="Type your notes here...",
            help="Take notes during the meeting. Use markdown for formatting.",
            label_visibility="visible"
        )
        
        # Save notes to session state
        if notes_text != st.session_state.notes_text:
            st.session_state.notes_text = notes_text
        
        # Notes controls
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            if st.button("Save Notes"):
                st.success("Notes saved!")
        with col_n2:
            if st.button("Clear Notes"):
                if st.button("Confirm Clear"):
                    st.session_state.notes_text = ""
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    # Right sidebar menu (similar to session controls but smaller)
    st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
    st.header("Menu")
    
    # Menu buttons
    if st.button("Q&A", key="qa_btn", help="Q&A Panel"):
        st.session_state.active_panel = "Q&A"
        st.rerun()
    
    if st.button("References", key="ref_btn", help="References"):
        st.session_state.active_panel = "References"
        st.rerun()
    
    if st.button("Summaries", key="sum_btn", help="Summaries"):
        st.session_state.active_panel = "Summaries"
        st.rerun()
    
    if st.button("Notes", key="notes_btn", help="Notes"):
        st.session_state.active_panel = "Notes"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🎙️ Microphone Status")
    
    # Show recording status based on WebRTC state
    if webrtc_ctx.state.playing:
        st.success("🔴 Recording... (Microphone active)")
        print("🎙️ WEBRTC STATE: Playing = True")
        
        # Initialize transcription and connect to audio processor
        if webrtc_ctx.audio_processor:
            print("✅ Audio processor is available")
            if webrtc_ctx.audio_processor.transcriber is None:
                print("🔄 Transcriber is None, attempting to initialize...")
                if webrtc_ctx.audio_processor.initialize_transcriber():
                    st.info("✅ Transcription model initialized")
                    print("✅ UI: Transcription model initialized")
                else:
                    st.error("❌ Failed to initialize transcription")
                    print("❌ UI: Failed to initialize transcription")
            else:
                print("✅ Transcriber already initialized")
        else:
            print("❌ Audio processor is not available")
            st.error("❌ Audio processor not available")
    else:
        st.info("⏸️ Click 'START' above to begin recording")
        print("⏸️ WEBRTC STATE: Playing = False")
    
    # Session info
    st.header("Session Settings")
    session_title = st.text_input("Meeting Title/Topic (optional)")
    num_speakers = st.number_input("Number of Speakers", min_value=1, max_value=10, value=2)
    
    # Privacy settings
    st.header("Privacy Settings")
    retention = st.selectbox(
        "Data Retention",
        ["Keep data (default)", "Opt out for this session"],
        index=0
    )
    
    # Audio settings
    st.header("Audio Settings")
    mic_source = st.selectbox("Microphone", ["Default", "Built-in", "External"])
    
    # Transcription settings
    st.header("Transcription Settings")
    st.success("✅ Transcription available")
    st.info("Using simple transcription for demonstration")

# Footer
st.markdown("---")
st.markdown("*Research Meeting AI - Live Transcription Prototype v0.3*")
