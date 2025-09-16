import numpy as np
import av
from typing import Dict, List, Union
import json

class AudioConverter:
    """Utility class for converting av.AudioFrame to various formats"""
    
    @staticmethod
    def to_ndarray(frame: av.AudioFrame) -> np.ndarray:
        """Convert to numpy ndarray (float32)"""
        return frame.to_ndarray().flatten().astype(np.float32)
    
    @staticmethod
    def to_list(frame: av.AudioFrame) -> List[float]:
        """Convert to Python list"""
        return AudioConverter.to_ndarray(frame).tolist()
    
    @staticmethod
    def to_int16(frame: av.AudioFrame) -> np.ndarray:
        """Convert to 16-bit integer array (common for audio processing)"""
        float_array = AudioConverter.to_ndarray(frame)
        return (float_array * 32767).astype(np.int16)
    
    @staticmethod
    def to_bytes(frame: av.AudioFrame) -> bytes:
        """Convert to raw bytes"""
        return AudioConverter.to_int16(frame).tobytes()
    
    @staticmethod
    def to_dict(frame: av.AudioFrame) -> Dict:
        """Convert to dictionary with full metadata"""
        audio_data = AudioConverter.to_ndarray(frame)
        return {
            'audio_data': audio_data.tolist(),  # Convert to list for JSON serialization
            'sample_rate': frame.sample_rate,
            'channels': frame.layout.nb_channels,
            'samples': frame.samples,
            'format': str(frame.format),
            'duration_ms': (len(audio_data) / frame.sample_rate) * 1000,
            'statistics': AudioConverter.get_stats(frame)
        }
    
    @staticmethod
    def to_json(frame: av.AudioFrame) -> str:
        """Convert to JSON string"""
        return json.dumps(AudioConverter.to_dict(frame))
    
    @staticmethod
    def to_wav_compatible(frame: av.AudioFrame) -> Dict:
        """Convert to format compatible with WAV files"""
        return {
            'data': AudioConverter.to_int16(frame),
            'sample_rate': frame.sample_rate,
            'channels': frame.layout.nb_channels,
            'bit_depth': 16
        }
    
    @staticmethod
    def get_stats(frame: av.AudioFrame) -> Dict:
        """Get statistical information about the audio"""
        audio = AudioConverter.to_ndarray(frame)
        return {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak_to_peak': float(np.max(audio) - np.min(audio)),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0)),
            'energy': float(np.sum(audio ** 2))
        }

# Usage Example:
def example_usage():
    """Example of how to use the AudioConverter"""
    
    class ExampleAudioProcessor(AudioProcessorBase):
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # Convert to different formats
            as_ndarray = AudioConverter.to_ndarray(frame)
            as_list = AudioConverter.to_list(frame)
            as_int16 = AudioConverter.to_int16(frame)
            as_bytes = AudioConverter.to_bytes(frame)
            as_dict = AudioConverter.to_dict(frame)
            as_json = AudioConverter.to_json(frame)
            stats = AudioConverter.get_stats(frame)
            
            # Use them however you want
            print(f"Audio as ndarray shape: {as_ndarray.shape}")
            print(f"Audio stats: {stats}")
            
            return frame 