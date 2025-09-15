import numpy as np
import streamlit as st
from streamlit_webrtc import AudioProcessorBase
import av

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.transcriber = None
        self.latest_audio_data = None
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Required method - must return av.AudioFrame"""
        if self.transcriber is not None:
            # Convert to your preferred format
            audio_array = self.frame_to_ndarray(frame)
            
            # Store for other uses
            self.latest_audio_data = audio_array
            
            # Process transcription
            self.transcriber.update_buffer(audio_array)
            new_text = self.transcriber.try_transcribe()
            if new_text:
                if 'transcript_text' in st.session_state:
                    st.session_state.transcript_text += new_text
        
        return frame  # Must return av.AudioFrame
    
    # YOUR CUSTOM CONVERSION METHODS:
    
    def frame_to_ndarray(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to numpy ndarray"""
        return frame.to_ndarray().flatten().astype(np.float32)
    
    def frame_to_list(self, frame: av.AudioFrame) -> list:
        """Convert av.AudioFrame to Python list"""
        return frame.to_ndarray().flatten().tolist()
    
    def frame_to_int16(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to 16-bit integer array (common for audio files)"""
        float_array = frame.to_ndarray().flatten()
        # Convert from float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
        return (float_array * 32767).astype(np.int16)
    
    def frame_to_bytes(self, frame: av.AudioFrame) -> bytes:
        """Convert av.AudioFrame to raw bytes"""
        int16_array = self.frame_to_int16(frame)
        return int16_array.tobytes()
    
    def frame_to_dict(self, frame: av.AudioFrame) -> dict:
        """Convert av.AudioFrame to dictionary with metadata"""
        audio_array = self.frame_to_ndarray(frame)
        return {
            'audio_data': audio_array,
            'sample_rate': frame.sample_rate,
            'channels': frame.layout.nb_channels,
            'samples': frame.samples,
            'format': str(frame.format),
            'duration_ms': (len(audio_array) / frame.sample_rate) * 1000,
            'max_amplitude': np.max(np.abs(audio_array)),
            'rms_level': np.sqrt(np.mean(audio_array ** 2))
        }
    
    def frame_to_wav_format(self, frame: av.AudioFrame) -> dict:
        """Convert to WAV-compatible format"""
        return {
            'data': self.frame_to_int16(frame),
            'sample_rate': frame.sample_rate,
            'channels': frame.layout.nb_channels,
            'bit_depth': 16
        }
    
    # UTILITY METHODS:
    
    def get_latest_audio_as_ndarray(self) -> np.ndarray:
        """Get the most recent audio as ndarray"""
        return self.latest_audio_data
    
    def get_audio_stats(self, frame: av.AudioFrame) -> dict:
        """Get statistics about the audio frame"""
        audio = self.frame_to_ndarray(frame)
        return {
            'mean': np.mean(audio),
            'std': np.std(audio),
            'min': np.min(audio),
            'max': np.max(audio),
            'rms': np.sqrt(np.mean(audio ** 2)),
            'peak_to_peak': np.max(audio) - np.min(audio)
        } 