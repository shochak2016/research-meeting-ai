import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="",
    layout="wide"
)

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
st.markdown("### Real-time research assistant prototype")

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

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    st.header("Live Transcript")
    
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    
    if st.session_state.recording:
        st.info("Recording in progress...")
        
        # Initialize transcript in session state if not exists
        if 'transcript_text' not in st.session_state:
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
    st.markdown('<h4 style="white-space: nowrap; min-width: 0; word-break: keep-all;">Ask a question about the meeting content:</h4>', unsafe_allow_html=True)
    
    # Question input
    question = st.text_area("", placeholder="Type your question here...")
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom section for references, summaries, and notes
st.markdown("---")

# Initialize expanded state
if 'expanded_column' not in st.session_state:
    st.session_state.expanded_column = None



# Render based on expanded state
if st.session_state.expanded_column is None:
    # Normal three-column layout
    col_ref1, col_ref2, col_ref3 = st.columns([1, 1, 1])
    
    with col_ref1:
        st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
        st.header("References")
        st.info("Relevant papers will appear here when you highlight text and select 'Find Relevant Papers'...")
        
        # Placeholder for citations
        with st.expander("Recent Papers", expanded=True):
            st.write("No papers selected yet. Highlight text in the transcript and click 'Find Relevant Papers' to get started.")
            st.write("")
            st.write("Example: Highlight 'oligodendrocyte maturation' to find relevant research papers.")
        
        # Expand button
        if st.button("Expand to Full View", key="expand_ref"):
            st.session_state.expanded_column = "References"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_ref2:
        st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
        st.header("Summaries")
        st.info("Summaries will appear here when you highlight text and select 'Summarize'...")
        
        # Placeholder for summaries
        with st.expander("Recent Summaries", expanded=True):
            st.write("No summaries generated yet. Highlight text in the transcript and click 'Summarize' to get started.")
            st.write("")
            st.write("Example: Highlight a section about 'multiple sclerosis research' to get a concise summary.")
        
        # Expand button
        if st.button("Expand to Full View", key="expand_sum"):
            st.session_state.expanded_column = "Summaries"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_ref3:
        st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
        st.header("Notes")
        st.info("Take notes during the meeting...")
        
        # Initialize notes in session state if not exists
        if 'notes_text' not in st.session_state:
            st.session_state.notes_text = ""
        
        # Notes text area
        notes_text = st.text_area(
            "Meeting Notes:",
            value=st.session_state.notes_text,
            height=200,
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
        
        # Expand button
        if st.button("Expand to Full View", key="expand_notes"):
            st.session_state.expanded_column = "Notes"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.expanded_column == "References":
    # Only References expanded - hide Summaries and Technical Terms completely
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    
    # Close button (similar to sidebar)
    col_close, col_title = st.columns([0.1, 0.9])
    with col_close:
        if st.button("X", key="close_expanded"):
            st.session_state.expanded_column = None
            st.rerun()
    with col_title:
        st.header("References - Full View")
    
    # Render only References content
    st.info("Relevant papers will appear here when you highlight text and select 'Find Relevant Papers'...")
    
    with st.expander("Recent Papers", expanded=True):
        st.write("No papers selected yet. Highlight text in the transcript and click 'Find Relevant Papers' to get started.")
        st.write("")
        st.write("Example: Highlight 'oligodendrocyte maturation' to find relevant research papers.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.expanded_column == "Summaries":
    # Only Summaries expanded - hide References and Notes completely
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    
    # Close button (similar to sidebar)
    col_close, col_title = st.columns([0.1, 0.9])
    with col_close:
        if st.button("X", key="close_expanded"):
            st.session_state.expanded_column = None
            st.rerun()
    with col_title:
        st.header("Summaries - Full View")
    
    # Render only Summaries content
    st.info("Summaries will appear here when you highlight text and select 'Summarize'...")
    
    with st.expander("Recent Summaries", expanded=True):
        st.write("No summaries generated yet. Highlight text in the transcript and click 'Summarize' to get started.")
        st.write("")
        st.write("Example: Highlight a section about 'multiple sclerosis research' to get a concise summary.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.expanded_column == "Notes":
    # Only Notes expanded - hide References and Summaries completely
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    
    # Close button (similar to sidebar)
    col_close, col_title = st.columns([0.1, 0.9])
    with col_close:
        if st.button("X", key="close_expanded"):
            st.session_state.expanded_column = None
            st.rerun()
    with col_title:
        st.header("Notes - Full View")
    
    # Rich notes editor in full view
    st.info("Take comprehensive notes during the meeting...")
    
    # Notes formatting options
    col_format1, col_format2, col_format3, col_format4 = st.columns(4)
    
    with col_format1:
        font_size = st.selectbox("Font Size:", ["12px", "14px", "16px", "18px", "20px"], key="font_size")
    
    with col_format2:
        text_style = st.selectbox("Style:", ["Normal", "Bold", "Italic", "Code"], key="text_style")
    
    with col_format3:
        list_type = st.selectbox("List:", ["None", "Bullet", "Numbered"], key="list_type")
    
    with col_format4:
        if st.button("Add Image"):
            st.info("Image upload functionality will be added here")
    
    # Large notes text area for full view
    notes_text_full = st.text_area(
        "Meeting Notes:",
        value=st.session_state.notes_text,
        height=500,
        placeholder="Type your comprehensive notes here... Use markdown for formatting: **bold**, *italic*, `code`, - bullets, 1. numbered lists",
        help="Take detailed notes during the meeting. Use markdown for rich formatting."
    )
    
    # Save notes to session state
    if notes_text_full != st.session_state.notes_text:
        st.session_state.notes_text = notes_text_full
    
    # Notes controls in full view
    col_control1, col_control2, col_control3, col_control4 = st.columns(4)
    with col_control1:
        if st.button("Save Notes", key="save_notes_full"):
            st.success("Notes saved!")
    with col_control2:
        if st.button("Export Notes", key="export_notes"):
            st.info("Export functionality will be added here")
    with col_control3:
        if st.button("Clear Notes", key="clear_notes_full"):
            if st.button("Confirm Clear", key="confirm_clear_full"):
                st.session_state.notes_text = ""
                st.rerun()
    with col_control4:
        if st.button("Format Notes", key="format_notes"):
            st.info("Auto-formatting will be added here")
    
    # Markdown preview
    if st.session_state.notes_text:
        st.markdown("---")
        st.subheader("Notes Preview:")
        st.markdown(st.session_state.notes_text)
    
    st.markdown('</div>', unsafe_allow_html=True)



# Footer
st.markdown("---")
st.markdown("*Research Meeting AI - Prototype v0.1*")

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False 