# Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.transcriber = None  # Will be initialized when recording starts
        self.frame_count = 0
        self.transcription_buffer = ""  # Store transcriptions for UI access
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames from the microphone"""
        self.frame_count += 1
        
        if self.transcriber is not None:
            # Convert audio frame to numpy array using our method
            audio_array = self.frame_to_ndarray(frame)
            
            # Debug: Print audio info every 50 frames
            if self.frame_count % 50 == 0:
                print(f"DEBUG: Frame {self.frame_count}, audio shape: {audio_array.shape}, mean: {np.mean(np.abs(audio_array)):.4f}")
            
            # Update the transcriber's buffer with numpy array
            self.transcriber.update_buffer(audio_array)
            
            # Try to get new transcription
            new_text = self.transcriber.try_transcribe()
            if new_text:
                print(f"DEBUG: Got transcription: {new_text.strip()}")
                # Store in both places for UI access
                self.transcription_buffer += new_text
                # Update session state with new text
                if 'transcript_text' in st.session_state:
                    st.session_state.transcript_text += new_text
                else:
                    st.session_state.transcript_text = new_text
            
        return frame  # Return frame unchanged
    
    # CONVERSION METHODS
    def frame_to_ndarray(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to numpy ndarray (float32)"""
        return frame.to_ndarray().flatten().astype(np.float32)
    
    def frame_to_list(self, frame: av.AudioFrame) -> list:
        """Convert av.AudioFrame to Python list"""
        return self.frame_to_ndarray(frame).tolist()
    
    def frame_to_int16(self, frame: av.AudioFrame) -> np.ndarray:
        """Convert av.AudioFrame to 16-bit integer array"""
        float_array = self.frame_to_ndarray(frame)
        return (float_array * 32767).astype(np.int16)
    
    def get_audio_stats(self, frame: av.AudioFrame) -> dict:
        """Get statistics about the audio frame"""
        audio = self.frame_to_ndarray(frame)
        return {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'energy': float(np.sum(audio ** 2))
        }
    
    # TRANSCRIPTION METHODS
    def initialize_transcriber(self, transcription_class):
        """Initialize the transcriber with proper error handling"""
        try:
            self.transcriber = transcription_class()
            print(f"✅ AudioProcessor: Transcriber initialized successfully")
            return True
        except Exception as e:
            print(f"❌ AudioProcessor: Failed to initialize transcriber: {e}")
            return False
    
    def get_latest_transcription(self) -> str:
        """Get the accumulated transcription text"""
        return self.transcription_buffer
    
    def clear_transcription_buffer(self):
        """Clear the transcription buffer"""
        self.transcription_buffer = "" 