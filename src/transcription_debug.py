import numpy as np
import time
from datetime import datetime

class Transcription:
    def __init__(self, model_id="debug", beam_size=1, len_window=8.0, freq=16000, fps=0.02, refresh_rate=0.5, queue_size: int = 50):
        """Debug transcription class that always triggers"""
        self.model_id = model_id
        self.len_window = len_window
        self.freq = freq
        self.fps = fps
        self.refresh_rate = 1.0  # Force 1 second refresh
        self.queue_size = queue_size
        
        # Debug buffer for audio data
        self.audio_buffer = []
        self.last_transcription_time = time.time()
        self.transcription_count = 0
        self.audio_frames_received = 0
        
        print(f"✅ Debug transcription initialized with model: {model_id}")
    
    def update_buffer(self, audio_data):
        """Update the audio buffer with new data"""
        self.audio_frames_received += 1
        self.audio_buffer.extend(audio_data)
        
        # Keep only the last len_window seconds of audio
        max_samples = int(self.len_window * self.freq)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        
        # Debug: Print every 100 frames
        if self.audio_frames_received % 100 == 0:
            print(f"DEBUG: Received {self.audio_frames_received} audio frames, buffer size: {len(self.audio_buffer)}")
    
    def try_transcribe(self):
        """Debug transcription that always returns text"""
        current_time = time.time()
        
        # Only transcribe every refresh_rate seconds
        if current_time - self.last_transcription_time < self.refresh_rate:
            return None
        
        self.last_transcription_time = current_time
        self.transcription_count += 1
        
        # Always return transcription (no audio activity check)
        debug_texts = [
            f"DEBUG: Audio frame #{self.audio_frames_received} processed",
            f"DEBUG: Buffer size: {len(self.audio_buffer)} samples",
            f"DEBUG: Transcription #{self.transcription_count}",
            f"DEBUG: Time: {datetime.now().strftime('%H:%M:%S')}",
            f"DEBUG: Mock transcription working - frame {self.audio_frames_received}",
            f"DEBUG: Audio processing active - {self.transcription_count} transcriptions",
            f"DEBUG: WebRTC stream connected - {len(self.audio_buffer)} samples buffered",
            f"DEBUG: Live transcription test #{self.transcription_count}",
            f"DEBUG: Microphone data flowing - frame {self.audio_frames_received}",
            f"DEBUG: Real-time processing working - transcription {self.transcription_count}"
        ]
        
        # Cycle through debug texts
        text = debug_texts[self.transcription_count % len(debug_texts)]
        
        # Add timestamp
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        formatted_text = f"{timestamp} DEBUG: {text}\n"
        
        print(f"DEBUG: Returning transcription: {formatted_text.strip()}")
        return formatted_text
