import numpy as np
import time
from datetime import datetime

class TranscriptionModule:
    def __init__(self, model_id="mock", beam_size=1, len_window=8.0, freq=16000, fps=0.02, refresh_rate=0.5, queue_size: int = 50):
        """Mock transcription class for testing"""
        self.model_id = model_id
        self.len_window = len_window
        self.freq = freq
        self.fps = fps
        self.refresh_rate = refresh_rate
        self.queue_size = queue_size
        
        # Mock buffer for audio data
        self.audio_buffer = []
        self.last_transcription_time = time.time()
        self.transcription_count = 0
        self.audio_activity_threshold = 0.001  # Threshold for detecting audio activity
        
        print(f"✅ Mock transcription initialized with model: {model_id}")
    
    def update_buffer(self, audio_data):
        """Update the audio buffer with new data"""
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
        return rms > self.audio_activity_threshold
    
    def try_transcribe(self):
        """Mock transcription that returns sample text based on audio activity"""
        current_time = time.time()
        
        # Only transcribe every refresh_rate seconds
        if current_time - self.last_transcription_time < self.refresh_rate:
            return None
        
        # Check if there's audio activity
        if not self.has_audio_activity():
            return None
        
        self.last_transcription_time = current_time
        self.transcription_count += 1
        
        # Return mock transcription text
        mock_texts = [
            "Hello, this is a test of the transcription system.",
            "The microphone is working correctly and picking up audio.",
            "Live transcription is now active and processing speech.",
            "You can speak and see text appear here in real-time.",
            "This demonstrates the transcription functionality working.",
            "The audio is being processed and converted to text.",
            "Mock transcription is working as expected.",
            "You should see this text appear in the transcript area.",
            "The system is ready for live transcription and note-taking.",
            "This shows that the microphone connection is successful.",
            "Real-time speech-to-text is functioning properly.",
            "The transcription service is active and monitoring audio.",
            "You can now speak and see your words transcribed.",
            "The audio processing pipeline is working correctly.",
            "Live transcription is capturing and displaying speech."
        ]
        
        # Cycle through mock texts
        text = mock_texts[self.transcription_count % len(mock_texts)]
        
        # Add timestamp
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        formatted_text = f"{timestamp} Speaker: {text}\n"
        
        return formatted_text
