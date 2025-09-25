#!/usr/bin/env python3
import sys, time, argparse, queue, threading
from pathlib import Path
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from transcription import Transcription

def list_input_devices():
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            print(f'[{i}] {d["name"]} (in:{d["max_input_channels"]}, default_sr:{int(d["default_samplerate"])} Hz)')

def run(device_idx=None, samplerate=None, block_ms=20, save_path=None):
    print("=" * 60)
    print("LIVE MICROPHONE TRANSCRIPTION TEST")
    print("=" * 60)

    try:
        transcriber = Transcription()
        print("✓ Model loaded")
        print(f"  - Sample rate: {transcriber.freq} Hz")
        print(f"  - Window: {transcriber.len_window}s")
        print(f"  - Device: {transcriber.device}")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return

    sr = int(samplerate or transcriber.freq)
    bs = max(1, int(sr * (block_ms / 1000.0)))
    if device_idx is not None:
        try:
            sd.check_input_settings(device=device_idx, samplerate=sr, channels=1, dtype='float32')
        except Exception as e:
            print(f"✗ Device/settings not supported: {e}")
            return

    print("\nAvailable audio devices (inputs only):")
    print("-" * 30)
    list_input_devices()

    print("\n" + "=" * 60)
    print("STARTING CONTINUOUS RECORDING")
    print("Speak clearly…  Ctrl+C to stop")
    print("=" * 60 + "\n")

    q = queue.Queue(maxsize=64)
    stop_evt = threading.Event()
    transcription_text = []
    start_time = time.time()

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(f"[Audio Status]: {status}", flush=True)
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass  # drop if behind

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype='float32',
            blocksize=bs,
            callback=audio_cb,
            device=device_idx,
            latency='low',
        ):
            while not stop_evt.is_set():
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    chunk = None
                if chunk is not None:
                    mono = chunk[:, 0] if chunk.ndim == 2 else chunk
                    mono = mono.astype(np.float32, copy=False)
                    transcriber.update_buffer(mono)
                new_text = transcriber.try_transcribe()
                if new_text:
                    sys.stdout.write(new_text)
                    sys.stdout.flush()
                    transcription_text.append(new_text)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n✗ Error during recording: {e}")
        return
    finally:
        stop_evt.set()

    dur = time.time() - start_time
    print("\n" + "=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    full_text = ''.join(transcription_text).replace('\n\n', '\n').strip()
    if full_text:
        print("\nFULL TRANSCRIPTION:")
        print("-" * 40)
        print(full_text or transcriber.prev_text)
        print("-" * 40)
    else:
        print("\nNo transcription captured.")

    print("\nSTATISTICS:")
    print(f"  - Buffer size: {len(transcriber.buffer)} samples")
    print(f"  - Recording duration: {dur:.2f} s")
    print(f"  - Final accumulated text: {transcriber.prev_text}")

    if save_path:
        try:
            Path(save_path).write_text(full_text, encoding='utf-8')
            print(f"\nSaved transcript to: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save transcript: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="Live mic transcription tester")
    ap.add_argument("--device", type=int, default=None, help="Input device index")
    ap.add_argument("--rate", type=int, default=None, help="Sample rate (defaults to model rate)")
    ap.add_argument("--block-ms", type=int, default=20, help="Block size in ms (default 20)")
    ap.add_argument("--save", type=str, default=None, help="Path to save final transcript")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    while True:
        run(device_idx=args.device, samplerate=args.rate, block_ms=args.block_ms, save_path=args.save)
        ans = input("\nRun another test? (y/n): ").strip().lower()
        if ans != 'y':
            break
