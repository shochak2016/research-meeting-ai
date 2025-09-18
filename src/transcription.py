# pip install faster-whisper soundfile librosa

from faster_whisper import WhisperModel
import torch
import queue, sys, time
import numpy as np
import sounddevice as sd
import re

# Count ".", "!", "?", "…" as sentence ends when followed by space or end of text
SENT_END_RE = re.compile(r'[.!?…](?=\s|$)')

class Transcription:
    def __init__(self, beam_size=1, len_window=10.0, freq=16000, fps=0.02, refresh_rate=0.4, queue_size: int = 50):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.freq = freq
        self.fps = fps
        self.len_window = 30.0  # give more headroom so segments don't age out
        self.refresh_rate = refresh_rate
        self.beam_size = beam_size

        # Initialize faster-whisper model
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel("base", device=self.device, compute_type=compute_type)

        # Bounded queue to avoid backpressure hanging the stream
        self.queue_size = queue_size
        self.queue = queue.Queue(maxsize=self.queue_size)
        self.buffer = np.zeros(0, dtype=float)
        self.blocksize = int(self.freq * self.fps)
        self.last_emit: float = time.time()
        self.prev_text: str = ""

        # The manager sets this as well; default True while active
        self.is_running: bool = True

        # --- streaming segment tracking (finalize-only appends) ---
        self.committed_text = ""  # what we've already appended to the UI/file
        self.live_hypothesis = ""  # current unfixed hypothesis (not yet appended)
        self.last_change_time = time.time()
        self.last_final_hyp = ""  # last hypothesis string we finalized (window-scoped)

        # time-chunk finalization settings
        self.CHUNK_SEC = 7.5  # 7.5-second chunks
        self.committed_upto_time = 0.0
        self.next_commit_boundary = self.CHUNK_SEC

        # Keep uncommitted segments here
        self.pending_segments = []  # list[dict(start, end, text)]

        self.last_rms = 0.0  # live mic level (RMS, smoothed)

        # sample-based gating to avoid unnecessary transcribes
        self.samples_seen = 0
        self.samples_since_last_tx = 0

    def audio_processing(self, indata, frames=None, time_info=None, status=None):
        # No-op if we're stopping/stopped
        if not getattr(self, "is_running", False):
            return

        if status:
            print(f"Audio status: {status}")
            return

        if indata is None or len(indata) == 0:
            return

        audio_data = indata.flatten().astype(np.float32)

        # --- add this live level calc (lightweight) ---
        try:
            rms = float(np.sqrt(np.mean(audio_data**2))) if audio_data.size else 0.0
            # Smooth a bit so the bar isn't jumpy
            self.last_rms = 0.8 * getattr(self, "last_rms", 0.0) + 0.2 * rms
        except Exception:
            pass
        # --- end add ---

        try:
            self.queue.put_nowait(audio_data)  # don't block
        except queue.Full:
            # Drop frame if full to avoid backpressure
            pass

    def iter_helper(self, prev: str, cur: str):
        if cur[:len(prev)] == prev:
            return cur[len(prev):], cur
        return "\n" + cur + "\n", cur

    def _transcribe_text(self, audio_buffer):
        """Transcribe audio buffer and return segments with timestamps."""
        try:
            if len(audio_buffer) < 1000:
                return []  # return list, not string

            audio_buffer = audio_buffer.astype(np.float32)
            max_abs = np.max(np.abs(audio_buffer)) if audio_buffer.size else 0.0
            if max_abs > 0:
                audio_buffer = audio_buffer / max_abs

            segments, info = self.model.transcribe(
                audio_buffer,
                language="en",
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                no_speech_threshold=0.25,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.8,
                temperature=0.0,
                condition_on_previous_text=False
            )

            out = []
            for s in segments:
                # keep timestamps for chunking
                out.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
            return out

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return []

    def update_buffer(self, audio_frame, sample_rate=None):
        """Add audio frame to buffer and maintain max window size"""
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate and sample_rate != self.freq and len(audio_frame) > 1:
            ratio = self.freq / sample_rate
            new_length = int(len(audio_frame) * ratio)
            audio_frame = np.interp(
                np.linspace(0, len(audio_frame), new_length, endpoint=False),
                np.arange(len(audio_frame)),
                audio_frame
            )

        # Append to rolling buffer
        self.buffer = np.concatenate([self.buffer, audio_frame.astype(np.float32)])

        max_samples = int(self.freq * self.len_window)
        if self.buffer.size > max_samples:
            self.buffer = self.buffer[-max_samples:]

        # Track samples for gating
        added = int(len(audio_frame))
        self.samples_seen += added
        self.samples_since_last_tx += added

    def try_transcribe(self):
        """Return segments and audio time when it's time to refresh."""
        min_new = int(self.freq * 0.20)
        if self.samples_since_last_tx < min_new:
            return None

        if time.time() - self.last_emit < self.refresh_rate:
            return None

        if self.buffer.size <= int(self.freq * 0.2):
            return None

        segs = self._transcribe_text(self.buffer)  # list of {start, end, text}
        self.last_emit = time.time()
        self.samples_since_last_tx = 0

        # current audio time (seconds) from samples we've seen
        audio_time = self.samples_seen / float(self.freq)
        cur_buf_sec = self.buffer.size / float(self.freq)  # seconds currently in buffer
        base_time = audio_time - cur_buf_sec  # absolute time at buffer[0]

        return {"segments": segs, "audio_time": audio_time, "base_time": base_time}

    def _merge_pending(self, new_segments):
        """
        Merge new segments into pending by replacing any older segments that overlap in time.
        This keeps only the *latest* hypothesis for a span.
        """
        if not new_segments:
            return

        # Small rounding to damp minor timestamp jitter
        def norm(s):
            return {
                "start": round(float(s["start"]), 2),
                "end": round(float(s["end"]), 2),
                "text": s["text"].strip()
            }

        incoming = [norm(s) for s in new_segments]
        updated = []

        for seg in incoming:
            s0, s1 = seg["start"], seg["end"]

            # Drop any existing pending segment that overlaps this new one
            pruned = []
            for old in self.pending_segments:
                o0, o1 = old["start"], old["end"]
                overlap = not (o1 <= s0 or o0 >= s1)  # (o0,o1) intersects (s0,s1)
                if not overlap:
                    pruned.append(old)

            self.pending_segments = pruned
            # Append the new one
            self.pending_segments.append(seg)

        # Keep pending sorted by start time for nice stitching/reading
        self.pending_segments.sort(key=lambda x: x["start"])

        # Also drop anything already committed (saves memory)
        if self.committed_upto_time:
            self.pending_segments = [s for s in self.pending_segments if s["end"] > self.committed_upto_time]

    # Simplified run method for standalone use
    def run(self):
        print(f"Using device: {self.device}")
        with sd.InputStream(
            samplerate=self.freq,
            channels=1,
            dtype="float",
            blocksize=self.blocksize,
            callback=self.audio_processing,
        ):
            try:
                while True:
                    # Pull one block from the producer
                    frame = self.queue.get()
                    frame = frame[:, 0] if frame.ndim == 2 else frame
                    
                    # Update buffer with new audio
                    self.update_buffer(frame)
                    
                    # Try to transcribe
                    result = self.try_transcribe()
                    if result and result.get("segments"):
                        # Simple output for standalone use
                        for seg in result["segments"]:
                            sys.stdout.write(seg["text"] + " ")
                        sys.stdout.flush()
            except KeyboardInterrupt:
                print("\nStopped.")