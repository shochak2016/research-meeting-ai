#!/usr/bin/env python3

import os
import sys

# Add the RealtimeSTT directory to the path
sys.path.insert(0, '/Users/pranaydayal/Desktop/researchai/RealtimeSTT')

def test_realtimestt():
    """Test RealtimeSTT functionality exactly like the GitHub example"""
    try:
        print("🚀 Testing RealtimeSTT import...")
        from RealtimeSTT import AudioToTextRecorder
        print("✅ RealtimeSTT imported successfully!")
        
        print("🚀 Testing RealtimeSTT initialization...")
        
        # Test with minimal settings exactly like the GitHub example
        recorder = AudioToTextRecorder(
            spinner=False,
            silero_sensitivity=0.01,
            model="tiny.en",
            language="en",
        )
        
        print("✅ RealtimeSTT recorder created successfully!")
        
        print("🎙️ Testing transcription...")
        print("Say something...")
        
        # Test transcription exactly like the GitHub example
        for i in range(5):  # Test 5 times
            try:
                text = recorder.text()
                if text and text.strip():
                    print(f"Detected text: {text}")
                else:
                    print("No text detected")
            except Exception as e:
                print(f"❌ Transcription error: {e}")
            
            import time
            time.sleep(1)
        
        print("✅ RealtimeSTT test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RealtimeSTT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_realtimestt()


