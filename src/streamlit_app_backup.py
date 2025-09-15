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
    from src.transcription import Transcription
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    RAG_AVAILABLE = False

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
                if 'transcript_text' in st.session_state:
                    st.session_state.transcript_text += new_text
        
        return frame  # Return frame unchanged

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state variables at the top
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

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

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
        // Send to Streamlit backend
        const event = new CustomEvent('findPapers', { detail: { text: selectedText } });
        document.dispatchEvent(event);
        popup.style.display = 'none';
    };
    
    window.summarize = function() {
        const selection = window.getSelection();
        const selectedText = selection.toString();
        console.log('Summarizing:', selectedText);
        // Send to Streamlit backend
        const event = new CustomEvent('summarize', { detail: { text: selectedText } });
        document.dispatchEvent(event);
        popup.style.display = 'none';
    };
});
</script>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Research Meeting AI</h1>', unsafe_allow_html=True)
st.markdown("### Real-time research assistant prototype")

# Initialize RAG system (with proper configuration)
@st.cache_resource
def get_rag_system():
    if not RAG_AVAILABLE:
        return None
    
    try:
        # You'll need to configure these parameters based on your setup
        query = "research papers"  # Default query
        index_name = "research-papers"  # Your Pinecone index name
        return build_rag(query, index_name)
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

rag_system = get_rag_system()

# Main content area with three columns
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    st.header("Live Transcript")
    
    if st.session_state.recording:
        st.info("Recording in progress...")
        
        # Initialize transcript with sample data if empty
        if not st.session_state.transcript_text:
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
            help="Click and edit the transcript text. Changes are saved automatically. Highlight text to access additional options."
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
        
    else:
        st.info("Click 'Start Recording' to begin capturing the meeting")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    
    # Dynamic content based on active panel
    if st.session_state.active_panel == "Q&A":
        st.markdown('<h4 style="white-space: nowrap; min-width: 0; word-break: keep-all;">Ask a question about the meeting content:</h4>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area("", placeholder="Type your question here...")
        col_q1, col_q2 = st.columns([1, 1])
        
        with col_q1:
            if st.button("Ask Question"):
                if question and rag_system:
                    with st.spinner("Generating answer..."):
                        try:
                            # Use RAG system to answer the question
                            result = rag_system(question)
                            answer = result.get('results', [{}])[0].get('summary', 'No answer available')
                            st.session_state.qa_history.append({
                                "question": question,
                                "answer": answer,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.success(f"Question submitted: {question}")
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
                elif question:
                    st.warning("RAG system not available. Please check configuration.")
                else:
                    st.warning("Please enter a question")
        
        with col_q2:
            if st.button("Suggest Questions"):
                suggested_questions = [
                    "What are the key findings discussed?",
                    "What are the main research questions?",
                    "What methodologies were mentioned?",
                    "What are the implications of this research?"
                ]
                st.info("Suggested questions:")
                for q in suggested_questions:
                    st.write(f"• {q}")
        
        # Q&A history
        st.subheader("Recent Q&A")
        if st.session_state.qa_history:
            for qa in reversed(st.session_state.qa_history[-5:]):  # Show last 5
                st.write(f"**Q ({qa['timestamp']}):** {qa['question']}")
                st.write(f"**A:** {qa['answer']}")
                st.write("---")
        else:
            st.write("No questions asked yet.")
    
    elif st.session_state.active_panel == "References":
        st.header("References")
        st.info("Relevant papers will appear here when you highlight text and select 'Find Relevant Papers'...")
        
        with st.expander("Recent Papers", expanded=True):
            if st.session_state.search_results:
                for i, result in enumerate(st.session_state.search_results[-5:]):  # Show last 5
                    st.write(f"**{i+1}.** {result.get('title', 'Unknown Title')}")
                    st.write(f"Authors: {result.get('authors', 'Unknown')}")
                    st.write(f"Abstract: {result.get('abstract', 'No abstract available')[:200]}...")
                    st.write("---")
            else:
                st.write("No papers selected yet. Highlight text in the transcript and click 'Find Relevant Papers' to get started.")
                st.write("")
                st.write("Example: Highlight 'oligodendrocyte maturation' to find relevant research papers.")
    
    elif st.session_state.active_panel == "Summaries":
        st.header("Summaries")
        st.info("Summaries will appear here when you highlight text and select 'Summarize'...")
        
        with st.expander("Recent Summaries", expanded=True):
            if st.session_state.summaries:
                for i, summary in enumerate(st.session_state.summaries[-5:]):  # Show last 5
                    st.write(f"**Summary {i+1}:**")
                    st.write(summary.get('text', ''))
                    st.write(f"*Source: {summary.get('source', 'Unknown')}*")
                    st.write("---")
            else:
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
            help="Take notes during the meeting. Use markdown for formatting."
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
    st.header("Session Controls")
    
    # Session info
    session_title = st.text_input("Meeting Title/Topic (optional)")
    num_speakers = st.number_input("Number of Speakers", min_value=1, max_value=10, value=2)
    
    # Recording controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", type="primary"):
            st.session_state.recording = True
            st.success("Recording started!")
    
    with col2:
        if st.button("Stop Recording"):
            st.session_state.recording = False
            st.info("Recording stopped!")
    
    # WebRTC streamer for microphone access (integrated with transcription)
    st.subheader("🎙️ Live Transcription")
    
    # WebRTC streamer for microphone access
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
        st.success("🔴 Live transcription active")
        
        # Initialize transcription and connect to audio processor
        if webrtc_ctx.audio_processor:
            if webrtc_ctx.audio_processor.transcriber is None:
                webrtc_ctx.audio_processor.transcriber = Transcription()
                st.info("Transcription model initialized")
    else:
        st.info("⏸️ Click 'START' above to begin live transcription")
    
    # Privacy settings
    st.header("Privacy Settings")
    retention = st.selectbox(
        "Data Retention",
        ["Keep data (default)", "Opt out for this session"],
        index=0
    )
    
    # Mic source (placeholder for now)
    st.header("Audio Source")
    mic_source = st.selectbox("Microphone", ["Default", "Built-in", "External"])

# Handle highlight actions (placeholder for now - would need JavaScript integration)
# This would be implemented with proper JavaScript-Streamlit communication
def handle_find_papers(selected_text):
    """Handle finding papers for selected text"""
    if rag_system:
        with st.spinner("Searching for relevant papers..."):
            try:
                # Use RAG system to find relevant papers
                result = rag_system(f"Find papers related to: {selected_text}")
                st.session_state.search_results.append({
                    "query": selected_text,
                    "results": result.get('results', []),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.success(f"Found papers for: {selected_text}")
            except Exception as e:
                st.error(f"Error searching papers: {e}")
    else:
        st.warning("RAG system not available")

def handle_summarize(selected_text):
    """Handle summarizing selected text"""
    if rag_system:
        with st.spinner("Generating summary..."):
            try:
                # Use RAG system to summarize
                result = rag_system(f"Summarize this: {selected_text}")
                summary = result.get('results', [{}])[0].get('summary', 'No summary available')
                st.session_state.summaries.append({
                    "text": summary,
                    "source": selected_text[:100] + "...",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.success(f"Generated summary for selected text")
            except Exception as e:
                st.error(f"Error generating summary: {e}")
    else:
        st.warning("RAG system not available")

# Footer
st.markdown("---")
st.markdown("*Research Meeting AI - Enhanced Prototype v0.2*")
