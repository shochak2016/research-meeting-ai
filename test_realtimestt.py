#!/usr/bin/env python3

import sys
import time
from datetime import datetime

def test_realtimestt():
    """Test RealtimeSTT functionality"""
    try:
        print("🚀 Testing RealtimeSTT import...")
        from RealtimeSTT import AudioToTextRecorder
        print("✅ RealtimeSTT imported successfully!")
        
        print("🚀 Testing RealtimeSTT initialization...")
        
        # Test with minimal settings
        recorder = AudioToTextRecorder(
            model="whisper",
            language="en",
            use_microphone=True,
            enable_realtime_transcription=True,
            model_size="base",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=False
        )
        
        print("✅ RealtimeSTT recorder created successfully!")
        
        # Test recording for 5 seconds
        print("🎙️ Testing recording for 5 seconds...")
        print("Please speak now...")
        
        with recorder:
            start_time = time.time()
            while time.time() - start_time < 5:
                text = recorder.text()
                if text and text.strip():
                    timestamp = datetime.now().strftime("[%H:%M:%S]")
                    print(f"📝 {timestamp}: {text.strip()}")
                time.sleep(0.1)
        
        print("✅ RealtimeSTT test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RealtimeSTT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing RealtimeSTT...")
    success = test_realtimestt()
    if success:
        print("🎉 RealtimeSTT is working!")
        sys.exit(0)
    else:
        print("💥 RealtimeSTT test failed!")
        sys.exit(1)


