import streamlit as st
import requests
import json
import sys
import os
import numpy as np
import queue
import threading
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# Add the researchai directory to the path so we can import transcription modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'researchai', 'src'))

# Import the simple transcription module
try:
    from transcription_simple import TranscriptionModule
    TRANSCRIPTION_AVAILABLE = True
    print("✅ Transcription module imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import transcription module: {e}")
    TRANSCRIPTION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="🔬",
    layout="wide"
)

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
        
        if self.transcriber is not None:
            # Convert audio frame to numpy array
            audio_array = self.frame_to_ndarray(frame)
            
            # Debug: Print audio info every 100 frames
            if self.frame_count % 100 == 0:
                print(f"DEBUG: Frame {self.frame_count}, audio shape: {audio_array.shape}, mean: {np.mean(np.abs(audio_array)):.4f}")
            
            # Update the transcriber's buffer
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
            
        return frame
    
    def frame_to_ndarray(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to numpy ndarray (float32)"""
        return frame.to_ndarray().flatten().astype(np.float32)
    
    def initialize_transcriber(self):
        """Initialize the transcriber with proper error handling"""
        if not TRANSCRIPTION_AVAILABLE:
            print("❌ Transcription not available")
            return False
        
        try:
            self.transcriber = TranscriptionModule()
            print("✅ AudioProcessor: Transcriber initialized successfully")
            return True
        except Exception as e:
            print(f"❌ AudioProcessor: Failed to initialize transcriber: {e}")
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
});
</script>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Research Meeting AI</h1>', unsafe_allow_html=True)
st.markdown("### Real-time research assistant with live transcription")

# Main content area with three columns
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    st.header("🎙️ Live Transcript")
    
    # Recording status display
    if st.session_state.recording:
        st.markdown('<div class="recording-status recording-active">🔴 Recording Active - Microphone Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recording-status recording-inactive">⏸️ Recording Stopped - Click Start to Begin</div>', unsafe_allow_html=True)
    
    # Initialize transcript with sample data if empty and not recording
    if not st.session_state.transcript_text and not st.session_state.recording:
        st.session_state.transcript_text = """[00:00] Speaker 1: Welcome everyone to today's seminar on oligodendrocyte maturation and its implications for neurological disorders.

[00:15] Speaker 1: Today we'll be discussing recent findings in cell differentiation, particularly focusing on the molecular mechanisms that regulate oligodendrocyte development.

[00:30] Speaker 2: Thank you for the introduction. I'd like to add that we've seen remarkable progress in understanding how transcription factors like Olig1 and Olig2 control the differentiation process.

[00:45] Audience: Could you elaborate on the implications for multiple sclerosis research? How do these findings relate to demyelination?

[01:00] Speaker 1: Excellent question. The connection is quite direct - oligodendrocytes are the cells that produce myelin, the protective sheath around nerve fibers. In multiple sclerosis, the immune system attacks this myelin, leading to nerve damage.

[01:15] Speaker 2: Building on that, our recent work has shown that promoting oligodendrocyte maturation could potentially help repair damaged myelin in MS patients. We've identified several key signaling pathways that could be therapeutic targets.

[01:30] Audience: What about the role of microglia in this process? I've read some conflicting studies about their involvement in remyelination.

[01:45] Speaker 1: That's a great point. The microglia story is complex - they can both help and hinder remyelination depending on their activation state. Recent research suggests they play a crucial role in clearing debris and promoting the recruitment of oligodendrocyte precursor cells."""
    
    # Editable transcript text area
    edited_transcript = st.text_area(
        "Edit Transcript:",
        value=st.session_state.transcript_text,
        height=400,
        help="Click and edit the transcript text. Changes are saved automatically. Highlight text to access additional options.",
        label_visibility="visible"
    )
    
    # Save changes to session state
    if edited_transcript != st.session_state.transcript_text:
        st.session_state.transcript_text = edited_transcript
        st.success("Transcript updated!")
    
    # Transcript controls
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        if st.button("Save Transcript"):
            st.success("Transcript saved!")
    with col_t2:
        if st.button("Export TXT"):
            st.info("Download functionality will be added here")
    with col_t3:
        if st.button("Clear Transcript"):
            if st.button("Confirm Clear"):
                st.session_state.transcript_text = ""
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
    st.header("🎙️ Microphone Controls")
    
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
                if webrtc_ctx.audio_processor.initialize_transcriber():
                    st.info("✅ Transcription model initialized")
                else:
                    st.error("❌ Failed to initialize transcription")
    else:
        st.session_state.recording = False
        st.info("⏸️ Click 'START' above to begin recording")
    
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
    if TRANSCRIPTION_AVAILABLE:
        st.success("✅ Transcription available")
        st.info("Using mock transcription for demonstration")
    else:
        st.error("❌ Transcription not available")
        st.info("Check console for error details")

# Footer
st.markdown("---")
st.markdown("*Research Meeting AI - Live Transcription Prototype v0.2*")


