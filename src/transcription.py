# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install nemo_toolkit[asr] soundfile librosa

from nemo.collections.asr.models import EncDecMultiTaskModel
import torch
import queue, sys, time
import numpy as np
import sounddevice as sd

class Transcription():
    def __init__(self, model_id="nvidia/canary-180m-flash", beam_size=1, len_window = 8.0, freq = 16000, fps = 0.02, refresh_rate=0.5, queue_size: int = 50):
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.freq = freq 
        self.fps = fps
        self.len_window = len_window
        self.refresh_rate = refresh_rate
        self.cfg = self.model.cfg.decoding
        self.beam_size = beam_size #increase if gpu is enabled and if ok with rewrites
        self.model.change_decoding_strategy(self.cfg)

        self.queue = queue.Queue()

        self.model = EncDecMultiTaskModel.from_pretrained(str(model_id)).to(self.device).eval()
        self.buffer = np.zeros(0, dtype=float)
        self.blocksize=int(self.preq * self.fps)
        self.queue_size = queue_size

        self.last_emit: float = time.time()
        self.prev_text: str = ""

    def audio_processing(self, input, status):
        if status:
            print(status)
        else:
            self.queue.put(input.copy())

    def iter_helper(prev: str, cur: str):
        if cur[len(prev):] == prev:
            return cur[len(prev):], cur
        return "\n" + cur + "\n", cur #cur, prev = diff_suffix(prev, cur)
    
    def run(self):
        print(f"Using device: {self.device}")
        with sd.InputStream(
            samplerate=self.freq,
            channels=1,
            dtype="float",
            blocksize=self.blocksize,
            callback=self.audio_cb,
        ):
            try:
                while True:
                    # Pull one blok from the producer
                    frame = self.q.get()
                    frame = frame[:, 0] if frame.ndim == 2 else frame
                    # Append to rolling buffer
                    self.buf = np.concatenate([self.buf, frame.astype(np.float32)])
                    max_samples = int(self.freq * self.len_window)
                    if self.buf.size > max_samples:
                        self.buf = self.buf[-max_samples:]

                    # keep retranscribing
                    if time.time() - self.last_emit >= self.refresh_rate and self.buf.size > int(self.freq * 0.5):
                        text = self._transcribe_text(self.buf)
                        new, self.prev_text = self.diff_suffix(self.prev_text, text)
                        sys.stdout.write(new)
                        sys.stdout.flush()
                        self.last_emit = time.time()
            except KeyboardInterrupt:
                print("Stopped.")