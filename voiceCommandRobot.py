#!/usr/bin/env python3
"""
voiceCommandRobot
------------------
An agent skill that simulates a robot receiving voice from the microphone
and using off-line OpenAI Whisper to transcribe it into text.

Design (two simple parts):
  Part 1 — Get voice from the microphone and save it as a WAV file.
  Part 2 — Use Whisper offline ASR to transcribe the WAV into text,
           then write the transcription to a text file.

Usage:
  python3 voiceCommandRobot.py                   # one-shot: record + transcribe
  python3 voiceCommandRobot.py --loop           # continuous loop (Ctrl+C to stop)
  python3 voiceCommandRobot.py --loop --max 5   # loop 5 times then stop
  python3 voiceCommandRobot.py --model tiny      # choose Whisper model size
  python3 voiceCommandRobot.py --duration 5      # record for N seconds
  python3 voiceCommandRobot.py --language zh     # hint language to Whisper
  python3 voiceCommandRobot.py --backend alsa   # force ALSA backend (RK3588)
  python3 voiceCommandRobot.py --list-devices   # show available microphones
  python3 voiceCommandRobot.py --optimize       # optimize for speed
"""

import argparse
import datetime
import json
import os
import sys
import datetime
import argparse
import numpy as np
import queue
import threading
import time
from collections import deque


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_to_file(text: str, filepath: str):
    mode = "a" if os.path.exists(filepath) else "w"
    line = f"[{timestamp()}] {text}\n"
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(line)
    return filepath


ACTION_COMMANDS = {
    "前进": ["前进", "向前", "往前", "走", "up", "forward"],
    "后退": ["后退", "向后", "往后", "退", "down", "backward"],
    "左转": ["左转", "向左转", "往左", "左", "left"],
    "右转": ["右转", "向右转", "往右", "右", "right"],
    "停止": ["停止", "停", "停下", "站住", "stop", "halt"],
    "抓取": ["抓取", "抓", "抓住", "拿", "grab", "pick"],
    "释放": ["释放", "放", "放开", "放下", "release", "drop"],
}


def match_command(text: str):
    if not text:
        return ""
    text_clean = text.strip().replace(" ", "").replace("，", "").replace("。", "")
    for cmd, aliases in ACTION_COMMANDS.items():
        for alias in aliases:
            if alias in text_clean or text_clean in alias:
                return cmd
    return ""


class VoiceRobot:
    def __init__(
        self,
        model_name: str = "tiny",
        model_dir: str = None,
        output_file: str = "robot_transcripts.txt",
        device: int = None,
        language: str = "zh",
        chunk_duration: float = 2.0,
        sample_rate: int = 16000,
        enable_commands: bool = False,
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.output_file = os.path.abspath(output_file)
        self.device = device
        self.language = language
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.enable_commands = enable_commands
        
        self._model = None
        self._audio_queue = queue.Queue()
        self._running = False
        
        print(f"[VoiceRobot] Model: {self.model_name}, Output: {self.output_file}")

    def _load_model(self):
        if self._model is not None:
            return
        
        print(f"[VoiceRobot] Loading Faster-Whisper model '{self.model_name}'...")
        
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            cache_path = os.path.join(
                self.model_dir,
                f"models--Systran--faster-whisper-{self.model_name.replace('.', '--')}"
            )
            
            if os.path.exists(cache_path):
                snapshots_dir = os.path.join(cache_path, "snapshots")
                if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
                    model_path = os.path.join(snapshots_dir, snapshots[0])
                    print(f"[VoiceRobot] Using LOCAL model: {model_path}")
                else:
                    from faster_whisper import download_model
                    print(f"[VoiceRobot] Downloading via hf-mirror.com...")
                    model_path = download_model(self.model_name, cache_dir=self.model_dir)
                    print(f"[VoiceRobot] Model saved to: {model_path}")
            else:
                from faster_whisper import download_model
                print(f"[VoiceRobot] Downloading via hf-mirror.com...")
                model_path = download_model(self.model_name, cache_dir=self.model_dir)
                print(f"[VoiceRobot] Model saved to: {model_path}")
        else:
            model_path = self.model_name
        
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_path, device="cpu", compute_type="int8")
        print("[VoiceRobot] Model loaded successfully!")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[Audio] Status: {status}", file=sys.stderr)
        self._audio_queue.put(indata.copy())

    def _capture_loop(self):
        import sounddevice as sd
        print(f"[Audio] Capturing at {self.sample_rate} Hz... Press Ctrl+C to stop")
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                device=self.device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * self.chunk_duration)
            ):
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[Audio] Error: {e}", file=sys.stderr)
        print("[Audio] Capture stopped")

    def _transcribe_loop(self):
        self._load_model()
        buffer = deque(maxlen=int(self.sample_rate * 10))
        
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            chunk_float = chunk.astype(np.float32) / 32768.0
            buffer.extend(chunk_float.flatten())
            
            if len(buffer) >= int(self.sample_rate * self.chunk_duration):
                audio_np = np.array(buffer, dtype=np.float32)
                
                kwargs = {}
                if self.language:
                    kwargs["language"] = self.language
                
                segments, _ = self._model.transcribe(audio_np, **kwargs)
                text = "".join([s.text for s in segments]).strip()
                
                if text:
                    print(f"\n>>> {text}\n")
                    
                    if self.enable_commands:
                        cmd = match_command(text)
                        if cmd:
                            print(f"[Command] -> {cmd}")
                            write_to_file(f"COMMAND: {cmd} (raw: {text})", self.output_file)
                        else:
                            write_to_file(text, self.output_file)
                    else:
                        write_to_file(text, self.output_file)
                    
                    buffer.clear()
        
        print("[Transcribe] Loop stopped")

    def start(self):
        self._running = True
        capture_t = threading.Thread(target=self._capture_loop, daemon=True)
        transcribe_t = threading.Thread(target=self._transcribe_loop, daemon=True)
        
        capture_t.start()
        transcribe_t.start()
        
        print("\n" + "="*60)
        print("  voiceCommandRobot - STREAMING MODE")
        print("  Speak into your microphone!")
        print("  Press Ctrl+C to STOP and exit.")
        print("="*60 + "\n")
        
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[VoiceRobot] Stopping...")
            self._running = False
            capture_t.join(timeout=2)
            transcribe_t.join(timeout=2)
            print("[VoiceRobot] Stopped. Transcripts saved to:", self.output_file)


def parse_args():
    p = argparse.ArgumentParser(description="voiceCommandRobot - Streaming Voice Transcription")
    p.add_argument("--model", default="tiny",
                   choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                           "medium", "medium.en", "large-v1", "large-v2", "large-v3"],
                   help="Faster-Whisper model size")
    p.add_argument("--model-dir", default=None,
                   help="Directory to cache/load models")
    p.add_argument("--output", default="robot_transcripts.txt",
                   help="Output text file")
    p.add_argument("--device", type=int, default=None,
                   help="Microphone device index")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio devices")
    p.add_argument("--language", default="zh",
                   help="Language hint (zh, en, etc.)")
    p.add_argument("--chunk-duration", type=float, default=2.0,
                   help="Audio chunk duration in seconds")
    p.add_argument("--enable-commands", action="store_true",
                   help="Enable action command matching")
    
    return p.parse_args()


def list_devices():
    import sounddevice as sd
    print("Audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) > 0:
            print(f"  [{i}] {dev.get('name', 'Unknown')}")


def main():
    args = parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    if args.model_dir is None:
        args.model_dir = os.path.expanduser("~/.cache/faster-whisper")
    
    robot = VoiceRobot(
        model_name=args.model,
        model_dir=args.model_dir,
        output_file=args.output,
        device=args.device,
        language=args.language,
        chunk_duration=args.chunk_duration,
        enable_commands=args.enable_commands,
    )
    
    robot.start()


if __name__ == "__main__":
    main()
