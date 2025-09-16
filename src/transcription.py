# pip install faster-whisper soundfile librosa

from faster_whisper import WhisperModel
import torch
import queue, sys, time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
import frame

load_dotenv()

class TranscriptionModule(): 
    def __init__(self, model_id="nvidia/canary-180m-flash", beam_size=1, len_window = 8.0, freq = 16000, fps = 0.02, refresh_rate=0.5, queue_size: int = 50):        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.freq = freq 
        self.fps = fps
        self.len_window = len_window
        self.refresh_rate = refresh_rate
        self.beam_size = beam_size #increase if gpu is enabled and if ok with rewrites
        
        # Initialize faster-whisper model
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel("base", device=self.device, compute_type=compute_type)

        self.queue = queue.Queue()
        self.buffer = np.zeros(0, dtype=float)
        self.blocksize = int(self.freq * self.fps)
        self.queue_size = queue_size

        self.last_emit: float = time.time()
        self.prev_text: str = ""

    def convert(self, input: av.AudioFrame) -> np.ndarray:
        return frame.to_ndarray(input)

    def audio_processing(self, input, status):
        if status:
            print(status)
        else:
            self.queue.put(input.copy())

    def iter_helper(self, prev: str, cur: str):
        if cur[:len(prev)] == prev:
            return cur[len(prev):], cur
        return "\n" + cur + "\n", cur #cur, prev = diff_suffix(prev, cur)


    
    def _transcribe_text(self, audio_buffer):
        """Transcribe audio buffer using the model"""
        # Faster-whisper expects normalized float32 audio
        audio_buffer = audio_buffer.astype(np.float32)
        
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio_buffer,
            beam_size=self.beam_size,
            language="en",
            vad_filter=True
        )
        
        # Collect all segments into text
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    
    def update_buffer(self, audio_frame):
        """Add audio frame to buffer and maintain max window size"""
        # Append to rolling buffer
        self.buffer = np.concatenate([self.buffer, audio_frame.astype(np.float32)])
        max_samples = int(self.freq * self.len_window)
        if self.buffer.size > max_samples:
            self.buffer = self.buffer[-max_samples:]
    
    def try_transcribe(self):
        """Attempt transcription if enough time has passed and buffer has enough audio"""
        if time.time() - self.last_emit >= self.refresh_rate and self.buffer.size > int(self.freq * 0.5):
            text = self._transcribe_text(self.buffer)
            new, self.prev_text = self.iter_helper(self.prev_text, text)
            self.last_emit = time.time()
            return new
        return None
    
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
                    new_text = self.try_transcribe()
                    if new_text is not None:
                        sys.stdout.write(new_text)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                print("Stopped.")