import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import json
import numpy as np
import threading
import time
import os
import queue
import sounddevice as sd
import torch
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.transcription import Transcription, SENT_END_RE

# CPU stability - limit threads to prevent thrash
torch.set_num_threads(min(4, os.cpu_count() or 4))

# Track how many sentence ends have been committed so far
if 'sentence_count' not in st.session_state:
    st.session_state.sentence_count = 0

# Page configuration
st.set_page_config(
    page_title="Research Meeting AI",
    page_icon="",
    layout="wide"
)

# Initialize session state variables at the top
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'active_panel' not in st.session_state:
    st.session_state.active_panel = "Q&A"
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""
if 'live_partial' not in st.session_state:
    st.session_state.live_partial = ""
if 'notes_text' not in st.session_state:
    st.session_state.notes_text = ""
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'last_transcription_time' not in st.session_state:
    st.session_state.last_transcription_time = 0

# ---------------------------
# Transcription (Whisper side)
# ---------------------------

# --------------------------------
# Helper Functions
# --------------------------------

def stitch_with_overlap(parts, max_olap=40):
    out = ""
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        k = min(len(out), len(p), max_olap)
        while k > 0 and not out.endswith(p[:k]):
            k -= 1
        out += p[k:]
    return out

def normalize_punctuation_spacing(text: str) -> str:
    # Ensure a space after ., !, ?, … if not already followed by space or newline
    return re.sub(r'([.!?…])(?!\s)', r'\1 ', text)

def insert_paragraph_breaks(text: str, start_count: int, step: int = 10):
    """
    Walks through text, and every time the cumulative sentence-end count hits a multiple of step, inserts two newlines.
    Returns (new_text, new_total_count).
    """
    out = []
    i = 0
    count = start_count
    for m in SENT_END_RE.finditer(text):
        end = m.end()
        out.append(text[i:end])  # include the sentence-ending punctuation
        count += 1
        if count % step == 0:
            out.append("\n\n")  # paragraph break after every 10 sentences
        i = end
    out.append(text[i:])  # remainder
    return ''.join(out), count

# --------------------------------
# Transcription Class

# --------------------------------
# Streamlit Transcription Manager
# --------------------------------

class StreamlitTranscriptionManager:
    def __init__(self):
        self.transcriber = None
        self.is_running = False
        self.thread = None
        self.audio_stream = None

        # Throttle live file writes
        self._last_live_write = 0.0
        self._last_live_payload = ""

    def start_transcription(self):
        """Start transcription in a background thread"""
        if self.is_running:
            return

        self.is_running = True
        # Tuned parameters for normal speech speed
        self.transcriber = Transcription(
            beam_size=2,           # Better accuracy (default: 1)
            len_window=15.0,       # Reduced buffer for faster processing (default: 30.0)
            refresh_rate=0.6,      # More time between updates (default: 0.4)
            chunk_sec=4.5,         # Faster commitment (default: 7.5)
            vad_silence_ms=1200    # Longer pauses to avoid splitting sentences (default: 500)
        )
        self.transcriber.is_running = True  # make callback live

        # 🔁 Reset session counters for a fresh run
        st.session_state.sentence_count = 0
        self.transcriber.committed_upto_time = 0.0
        self.transcriber.next_commit_boundary = self.transcriber.CHUNK_SEC
        self.transcriber.pending_segments = []

        # Reset markers so nothing carries over from previous sessions
        self.transcriber.committed_text = ""
        self.transcriber.last_final_hyp = ""

        # Clear any leftover files from previous sessions
        for p in ("/tmp/transcript_update.txt", "/tmp/transcript_live.txt"):
            try:
                with open(p, "w") as f:
                    f.write("")
            except Exception:
                pass

        self.thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.thread.start()

        print("🎤 Transcription started")

    def stop_transcription(self, timeout: float = 3.0) -> dict:
        status = {
            "set_is_running_false": False,
            "audio_stream_aborted": False,
            "audio_stream_stopped": False,
            "audio_stream_closed": False,
            "queue_cleared": False,
            "thread_joined": False,
            "thread_alive_after_join": None,
            "transcriber_marked_stopped": False,
            "transcript_file_cleared": False,
            "errors": [],
        }

        try:
            self.is_running = False
            if self.transcriber is not None:
                self.transcriber.is_running = False
            status["set_is_running_false"] = True
        except Exception as e:
            status["errors"].append(f"is_running flag: {e}")

        stream = getattr(self, "audio_stream", None)
        if stream is not None:
            try:
                if hasattr(stream, "abort"):
                    stream.abort()
                status["audio_stream_aborted"] = True
            except Exception as e:
                status["errors"].append(f"audio_stream.abort(): {e}")

            try:
                if hasattr(stream, "stop"):
                    stream.stop()
                status["audio_stream_stopped"] = True
            except Exception as e:
                status["errors"].append(f"audio_stream.stop(): {e}")

            try:
                if hasattr(stream, "close"):
                    stream.close()
                status["audio_stream_closed"] = True
            except Exception as e:
                status["errors"].append(f"audio_stream.close(): {e}")
            finally:
                self.audio_stream = None

        try:
            if getattr(self, "transcriber", None) and hasattr(self.transcriber, "queue"):
                q = self.transcriber.queue
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break
                status["queue_cleared"] = True
        except Exception as e:
            status["errors"].append(f"queue clear: {e}")

        try:
            if getattr(self, "thread", None) is not None:
                self.thread.join(timeout=timeout)
                status["thread_joined"] = True
                status["thread_alive_after_join"] = self.thread.is_alive()
                if not self.thread.is_alive():
                    self.thread = None
        except Exception as e:
            status["errors"].append(f"thread.join(): {e}")

        # Flush any remaining live text on stop
        try:
            tm = self.transcriber
            if tm and tm.pending_segments:
                tail = stitch_with_overlap(
                    [s["text"] for s in tm.pending_segments if s["end"] > tm.committed_upto_time]
                ).strip()
                if tail:
                    with open("/tmp/transcript_update.txt", "a") as f:
                        f.write(tail)
                tm.pending_segments = []
        except Exception as e:
            status["errors"].append(f"final flush: {e}")

        try:
            with open("/tmp/transcript_update.txt", "w") as f:
                f.write("")
            status["transcript_file_cleared"] = True
        except Exception as e:
            status["errors"].append(f"clear transcript file: {e}")

        critical_ok = (
            status["set_is_running_false"]
            and getattr(self, "audio_stream", None) is None
            and (status["thread_alive_after_join"] in (False, None))
        )

        return {"ok": critical_ok and not status["errors"], "details": status}

    def _transcription_loop(self):
        try:
            # Use the user-selected input device (or system default if None)
            selected_dev = st.session_state.get("input_device_index", None)
            try:
                device_info = sd.query_devices(selected_dev, 'input')  # works with index or None
            except Exception as e:
                # fallback to system default input if the selected device is unavailable
                device_info = sd.query_devices(None, 'input')
                selected_dev = None
                print(f"⚠️ Falling back to system default input: {e}")

            device_sample_rate = int(device_info['default_samplerate'])
            print(f"🎤 Using audio device: {device_info['name']} (index={selected_dev})")
            print(f"🎤 Device sample rate: {device_sample_rate} Hz")

            stream = sd.InputStream(
                device=selected_dev,  # <-- key line: respect explicit selection (or None for default)
                samplerate=device_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(device_sample_rate * 0.01),  # ~10ms blocks
                callback=self.transcriber.audio_processing
            )
            self.audio_stream = stream
            stream.start()

            while True:
                if not self.is_running:
                    break

                try:
                    frame = self.transcriber.queue.get(timeout=0.1)
                except queue.Empty:
                    if not self.is_running:
                        break
                    continue

                if not self.is_running:
                    break

                if frame.ndim > 1:
                    frame = frame.flatten()
                frame = frame.astype(np.float32)

                self.transcriber.update_buffer(frame, device_sample_rate)

                tick = self.transcriber.try_transcribe()
                if tick is not None and self.is_running:
                    segs = tick["segments"]
                    audio_time = tick["audio_time"]
                    base_time = tick["base_time"]
                    now = time.time()

                    # Convert to absolute timeline
                    abs_segs = [
                        {"start": base_time + s["start"], "end": base_time + s["end"], "text": s["text"]}
                        for s in segs
                    ]

                    # 1) Merge absolute segments into pending
                    self.transcriber._merge_pending(abs_segs)

                    # 2) LIVE PREVIEW: newest hypothesis only (after last commit)
                    live_text_parts = [
                        s["text"] for s in self.transcriber.pending_segments
                        if s["end"] > self.transcriber.committed_upto_time
                    ]
                    live_text = stitch_with_overlap(live_text_parts)  # de-dup across segment edges

                    if live_text != self._last_live_payload and (now - self._last_live_write) >= 0.25:
                        try:
                            with open("/tmp/transcript_live.txt", "w") as lf:
                                lf.write(live_text)
                            self._last_live_write = now
                            self._last_live_payload = live_text
                        except Exception as e:
                            print(f"❌ Live write error: {e}")

                    # 3) FINALIZE in fixed chunks (still using absolute times)
                    while audio_time >= self.transcriber.next_commit_boundary:
                        commit_until = self.transcriber.next_commit_boundary
                        to_commit = [
                            s for s in self.transcriber.pending_segments
                            if s["end"] <= commit_until and s["end"] > self.transcriber.committed_upto_time
                        ]
                        chunk_text = stitch_with_overlap([s["text"] for s in to_commit]).strip()

                        if chunk_text:
                            try:
                                with open("/tmp/transcript_update.txt", "a") as f:
                                    # no visible separator; we'll handle spacing in the UI
                                    f.write(chunk_text)
                            except Exception as e:
                                print(f"❌ File write error: {e}")

                        # Advance cursors for the next chunk
                        self.transcriber.committed_upto_time = commit_until
                        self.transcriber.next_commit_boundary += self.transcriber.CHUNK_SEC

                        # Drop anything we just committed so it won't appear in live again
                        self.transcriber.pending_segments = [
                            s for s in self.transcriber.pending_segments
                            if s["end"] > self.transcriber.committed_upto_time
                        ]

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            self.is_running = False
        finally:
            try:
                if self.audio_stream is not None:
                    try:
                        if hasattr(self.audio_stream, "abort"):
                            self.audio_stream.abort()
                    except Exception:
                        pass
                    try:
                        self.audio_stream.stop()
                    except Exception:
                        pass
                    try:
                        self.audio_stream.close()
                    except Exception:
                        pass
            finally:
                self.audio_stream = None

# Initialize transcription manager ONCE and keep it across reruns
if "transcription_manager" not in st.session_state:
    st.session_state.transcription_manager = StreamlitTranscriptionManager()

transcription_manager = st.session_state.transcription_manager

# ---------------------------
# Debug helper (optional)
# ---------------------------

def _dump_audio_state(tag="state"):
    mgr = transcription_manager
    st.write({
        tag: {
            "manager_is_running": mgr.is_running,
            "has_stream": mgr.audio_stream is not None,
            "thread_exists": mgr.thread is not None,
            "thread_alive": (mgr.thread.is_alive() if mgr.thread else None),
            "transcriber_exists": mgr.transcriber is not None,
            "transcriber_running": (mgr.transcriber.is_running if mgr.transcriber else None),
            "queue_size": (mgr.transcriber.queue.qsize() if (mgr.transcriber and hasattr(mgr.transcriber, "queue")) else None),
        }
    })

# ---------------------------
# Custom CSS/JS (UI helpers)
# ---------------------------

st.markdown("""
<style>
.main-header {
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 2.5rem;
    font-weight: bold;
}

.summary-panel {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: black;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}

.control-panel {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.transcript-scroll {
    height: 400px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    background: #ffffff;
    color: #000000;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.95rem;
    line-height: 1.35rem;
    white-space: pre-wrap;
    word-break: break-word;
}

.stButton > button {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.stTextInput > div > div > input {
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}

.stTextArea > div > div > textarea {
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}

.stSelectbox > div > div > select {
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}

.stNumberInput > div > div > input {
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}

.sidebar .stButton > button {
    width: 100%;
    margin: 0.25rem 0;
}

/* Popup styles */
.popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    z-index: 1000;
    max-width: 500px;
    width: 90%;
}

.popup-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
}

.popup h3 {
    margin-top: 0;
    color: #333;
}

.popup button {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    margin: 0.25rem;
}

.popup button:hover {
    background: #5a6fd8;
}
</style>

<script>
// Popup functionality
function showPopup(title, content) {
    const overlay = document.createElement('div');
    overlay.className = 'popup-overlay';
    overlay.onclick = () => hidePopup();
    
    const popup = document.createElement('div');
    popup.className = 'popup';
    popup.innerHTML = `
        <h3>${title}</h3>
        <p>${content}</p>
        <button onclick="hidePopup()">Close</button>
    `;
    
    document.body.appendChild(overlay);
    document.body.appendChild(popup);
}

function hidePopup() {
    const overlay = document.querySelector('.popup-overlay');
    const popup = document.querySelector('.popup');
    if (overlay) overlay.remove();
    if (popup) popup.remove();
}

// Auto-hide alerts after 3 seconds
setTimeout(() => {
    const alerts = document.querySelectorAll('.stAlert');
    alerts.forEach(alert => {
        if (alert.querySelector('.stSuccess, .stInfo, .stWarning')) {
            alert.style.transition = 'opacity 0.5s';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }
    });
}, 3000);
</script>
""", unsafe_allow_html=True)

# ---------------------------
# Main UI
# ---------------------------

st.markdown('<h1 class="main-header">Research Meeting AI</h1>', unsafe_allow_html=True)
st.markdown("### Real-time research assistant prototype")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
    st.header("Live Transcript")

    # --- Always show the transcript UI ---
    is_rec = st.session_state.recording

    # Status banner
    if is_rec:
        st.info("Recording in progress…")
    else:
        st.warning("Recording stopped. You can still edit and save the transcript below.")

    # --- Final, append-only transcript (read-only, scrollable) ---
    # Build the visible text = committed + (optional blank line) + live preview
    committed = st.session_state.transcript_text or ""
    live = st.session_state.live_partial or ""

    if committed and live and not committed.endswith((" ", "\n")):
        box_content = committed + " " + live
    else:
        box_content = committed + live

    st.markdown("""
    <style>
    .transcript-scroll {
      height: 400px;
      overflow-y: auto;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 0.75rem 0.9rem;
      background: #ffffff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.95rem;
      line-height: 1.35rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div id="tx-box" class="transcript-scroll">' + box_content + '</div>', unsafe_allow_html=True)

    # Optional: autoscroll to bottom while live updating
    st.markdown("""
    <script>
    const el = document.getElementById("tx-box");
    if (el) { el.scrollTop = el.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)

    # Controls row (available whether recording or not)
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        if st.button("Save Transcript"):
            st.success("Transcript saved!")
    with col_t2:
        if st.button("Export TXT"):
            st.info("Download functionality will be added here")
    with col_t3:
        if st.button("Clear Live Preview"):
            st.session_state.live_partial = ""
            st.rerun()

    st.info("**Tip:** The newest line stays at the bottom automatically while you're at the bottom. If you scroll up, autoscroll pauses until you return to the bottom.")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="summary-panel">', unsafe_allow_html=True)

    if st.session_state.active_panel == "Q&A":
        st.markdown('<h4 style="white-space: nowrap; min-width: 0; word-break: keep-all;">Ask a question about the meeting content:</h4>', unsafe_allow_html=True)
        question = st.text_area("Question Input", placeholder="Type your question here...", label_visibility="collapsed")
        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            if st.button("Ask Question"):
                if question:
                    st.success(f"Question submitted: {question}")
                else:
                    st.warning("Please enter a question")
        with col_q2:
            if st.button("Suggest Questions"):
                st.info("Suggested questions will appear here...")

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
        notes_text = st.text_area(
            "Meeting Notes:",
            value=st.session_state.notes_text,
            height=300,
            placeholder="Type your notes here...",
            help="Take notes during the meeting. Use markdown for formatting."
        )
        if notes_text != st.session_state.notes_text:
            st.session_state.notes_text = notes_text

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

# Horizontal menu bar under the two columns
st.markdown("---")
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
col_menu1, col_menu2, col_menu3, col_menu4, col_menu5 = st.columns([1, 1, 1, 1, 1])

with col_menu1:
    if st.button("References", key="menu_ref", help="View References"):
        st.session_state.active_panel = "References"
        st.rerun()

with col_menu2:
    if st.button("Summaries", key="menu_sum", help="View Summaries"):
        st.session_state.active_panel = "Summaries"
        st.rerun()

with col_menu3:
    if st.button("Notes", key="menu_notes", help="View Notes"):
        st.session_state.active_panel = "Notes"
        st.rerun()

with col_menu4:
    if st.button("Q&A", key="menu_qa", help="View Q&A"):
        st.session_state.active_panel = "Q&A"
        st.rerun()

with col_menu5:
    if st.button("☰", key="menu_toggle", help="Toggle between 2-column and 3-column view"):
        if st.session_state.active_panel == "Q&A":
            st.session_state.active_panel = "References"
        else:
            st.session_state.active_panel = "Q&A"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---- Audio device helpers ----

@st.cache_data(show_spinner=False)
def list_input_devices():
    """Return [(index, label)] for all input-capable devices."""
    devices = sd.query_devices()
    options = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            # e.g. "[2] Yeti Stereo Microphone (USB Audio)"
            label = f"[{i}] {d.get('name', 'Unknown')} — {int(d.get('default_samplerate', 16000))} Hz"
            options.append((i, label))
    return options

# ---------------------------
# Sidebar (controls)
# ---------------------------

with st.sidebar:
    st.header("Session Controls")
    session_title = st.text_input("Meeting Title/Topic (optional)")
    num_speakers = st.number_input("Number of Speakers", min_value=1, max_value=10, value=2)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Start Recording", type="primary"):
            st.session_state.recording = True
            transcription_manager.start_transcription()
            st.success("Recording started!")

    with colB:
        if st.button("Stop Recording"):
            st.session_state.recording = False
            try:
                result = transcription_manager.stop_transcription(timeout=3.0)
            except Exception as e:
                result = {"ok": False, "details": {"errors": [f"stop_transcription() exception: {e}"]}}

            # Belt & suspenders: stop any lingering PortAudio streams
            try:
                sd.stop()
            except Exception:
                pass

            # Clear live partial so it doesn't look "stuck"
            st.session_state.live_partial = ""

            # Debug: show state after stopping (optional)
            _dump_audio_state("after_stop")

            if result.get("ok"):
                st.success("✅ Recording stopped and cleaned up.")
            else:
                st.error("⚠️ Tried to stop, but some components may still be running.")
                with st.expander("See shutdown details"):
                    st.write(result.get("details"))
                st.info("If the mic indicator stays on, click Stop again or reload the page.")

            st.rerun()

    st.header("Privacy Settings")
    retention = st.selectbox(
        "Data Retention",
        ["Keep data (default)", "Opt out for this session"],
        index=0
    )

    st.header("Audio Source")
    # Build the list of real input devices
    device_list = list_input_devices()
    labels = ["System default"] + [label for _, label in device_list]
    choice = st.selectbox("Input device", labels, index=0, help="Pick the exact mic you want.")

    # Store the selected device index in session_state so the manager can use it
    if choice == "System default":
        st.session_state.input_device_index = None
    else:
        sel_idx = labels.index(choice) - 1  # offset because of "System default"
        st.session_state.input_device_index = device_list[sel_idx][0]

    st.subheader("Mic level")
    # Read the current RMS level from the running transcriber
    rms = 0.0
    tm = st.session_state.get("transcription_manager")
    if tm and tm.transcriber:
        rms = float(getattr(tm.transcriber, "last_rms", 0.0))

    # Simple gain so the bar moves nicely (tweak the multiplier if it's too hot/quiet)
    meter = max(0.0, min(1.0, rms * 20.0))  # scale RMS to 0..1 (20x is a sensible default)
    st.progress(int(meter * 100), text="Listening…")

# ---------------------------
# Transcript update polling
# ---------------------------

def check_transcript_updates():
    """Check for transcript updates and update UI"""
    try:
        # 1) Append any finalized chunks to transcript_text
        if os.path.exists("/tmp/transcript_update.txt"):
            with open("/tmp/transcript_update.txt", "r") as f:
                new_content = f.read()

            if new_content and new_content.strip():
                nc = new_content.strip()
                # normalize punctuation spacing first
                nc = normalize_punctuation_spacing(nc)
                # 👉 insert paragraph breaks based on cumulative sentence count
                nc_with_breaks, new_total = insert_paragraph_breaks(
                    nc,
                    st.session_state.sentence_count,
                    step=10
                )

                prev = st.session_state.transcript_text or ""
                # stitch (keeping a space if needed)
                if prev and not prev.endswith((" ", "\n")) and nc_with_breaks and not nc_with_breaks.startswith((" ", "\n")):
                    st.session_state.transcript_text = prev + " " + nc_with_breaks
                else:
                    st.session_state.transcript_text = prev + nc_with_breaks

                # bump the committed sentence counter
                st.session_state.sentence_count = new_total

                # clear file so we don't re-append
                with open("/tmp/transcript_update.txt", "w") as f:
                    f.write("")
                print("📝 Appended FINAL transcript from file (with paragraph breaks)")

        # 2) Read live tail directly (file contains only the tail now)
        if os.path.exists("/tmp/transcript_live.txt"):
            with open("/tmp/transcript_live.txt", "r") as lf:
                live_tail = lf.read()
            st.session_state.live_partial = live_tail

    except Exception as e:
        print(f"❌ Error checking transcript updates: {e}")

# Poll the transcript more frequently for partials
if st.session_state.recording:
    st_autorefresh(interval=300, key="poll-transcript")  # was 1000

# Quick UI refresh for the mic meter (250ms) while recording
if st.session_state.recording:
    st_autorefresh(interval=250, key="mic-meter-refresh")


check_transcript_updates()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("*Research Meeting AI - Prototype v0.1*")