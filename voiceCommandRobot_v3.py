#!/usr/bin/env python3
"""
voiceCommandRobot - Real-time Voice Transcription
---------------------------------------------------
Two-thread architecture:
  Thread 1 (Recorder): Captures audio from microphone -> writes to raw PCM file
  Thread 2 (Transcriber): Reads the growing PCM file with 2s delay -> SenseVoice ASR -> text output

Usage:
  python3 voiceCommandRobot.py --device 2 --quiet
  python3 voiceCommandRobot.py --device 2 --enable-commands
  Press Ctrl+C to stop.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["MODELSCOPE_OFFLINE"] = "1"
os.environ["MODELSCOPE_CACHE"] = os.path.expanduser("~/.cache/modelscope")
os.environ["MKL_DNN"] = "0"
os.environ["ONEDNN"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ORT_CUDA"] = "0"
os.environ["ONNXRUNTIME_CUDA"] = "0"
os.environ["ORT_GPU_DEVICE_ID"] = "-1"

import datetime
import argparse
import numpy as np
import threading
import time
import subprocess


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
    "抓取": ["抓取", "抓", "拿", "pick", "grab"],
    "释放": ["释放", "放", "松开", "release", "drop"],
}


def match_command(text: str) -> str:
    text_lower = text.lower().strip()
    for cmd, keywords in ACTION_COMMANDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return cmd
    return ""


# Raw PCM format constants
SAMPLE_WIDTH = 2  # bytes per sample (S16_LE = 16-bit signed little-endian)


class VoiceRobot:
    def __init__(self, args):
        self.sample_rate = args.sample_rate
        self.device = args.device
        self.output_file = args.output
        self.audio_threshold = args.threshold
        self.audio_gain = args.gain
        self.enable_commands = args.enable_commands
        self.model_dir = args.model_dir
        self.quiet = args.quiet
        self.delay_seconds = args.delay

        # Raw PCM file for inter-thread communication
        self._pcm_file = "/tmp/voicerobot_audio.raw"
        self._running = False
        self._model = None
        self._recorder_proc = None

    def _log(self, msg, file=None, flush=True):
        """Print log message only if not in quiet mode."""
        if not self.quiet:
            print(msg, file=file, flush=flush)

    # ------------------------------------------------------------------ #
    #  Model loading (same as before)
    # ------------------------------------------------------------------ #

    def _install_sensevoice(self):
        install_methods = [
            ("pip funasr (Tsinghua)", [
                sys.executable, "-m", "pip", "install", "funasr",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
            ]),
            ("pip funasr (Aliyun)", [
                sys.executable, "-m", "pip", "install", "funasr",
                "-i", "https://mirrors.aliyun.com/pypi/simple/",
                "--trusted-host", "mirrors.aliyun.com"
            ]),
        ]
        for name, cmd in install_methods:
            self._log(f"[Install] Trying {name}...")
            try:
                subprocess.check_call(cmd, timeout=180)
                self._log(f"[Install] Success via {name}!")
                return True
            except Exception as e:
                self._log(f"[Install] Failed via {name}: {e}")
                continue
        self._log("[Install] All methods failed.", file=sys.stderr)
        return False

    def _find_local_model(self):
        if self.model_dir and os.path.exists(self.model_dir):
            if len(os.listdir(self.model_dir)) > 0:
                return self.model_dir
            self._log(f"[Model] Model dir exists but is empty: {self.model_dir}", file=sys.stderr)
            return None

        ms_cache_base = os.path.expanduser("~/.cache/modelscope/hub/models/iic")
        if os.path.exists(ms_cache_base):
            for name in os.listdir(ms_cache_base):
                if name.lower().startswith("sensevoice"):
                    full_path = os.path.join(ms_cache_base, name)
                    if len(os.listdir(full_path)) > 0:
                        self._log(f"[Model] Found local ModelScope model: {full_path}")
                        return full_path
                    self._log(f"[Model] Model dir exists but is empty: {full_path}", file=sys.stderr)

        cache_dir = os.path.expanduser("~/.cache/sensevoice")
        if os.path.exists(cache_dir):
            for name in os.listdir(cache_dir):
                p = os.path.join(cache_dir, name)
                if os.path.isdir(p) and len(os.listdir(p)) > 0:
                    return p

        return None

    def _load_model(self):
        if self._model is not None:
            return

        self._log("[Model] Loading SenseVoice Small INT8...")

        try:
            import torch
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = False
            if hasattr(torch.backends, 'onednn'):
                torch.backends.onednn.enabled = False
            torch.set_num_threads(4)
            from funasr import AutoModel
        except ImportError:
            self._log("[Model] funasr not installed, installing...")
            if self._install_sensevoice():
                import torch
                if hasattr(torch.backends, 'mkldnn'):
                    torch.backends.mkldnn.enabled = False
                if hasattr(torch.backends, 'onednn'):
                    torch.backends.onednn.enabled = False
                torch.set_num_threads(4)
                from funasr import AutoModel
            else:
                raise RuntimeError("funasr install failed. Run: pip install funasr")

        local_path = self._find_local_model()

        if local_path:
            self._log(f"[Model] Using local model: {local_path}")
            try:
                self._model = AutoModel(
                    model=local_path,
                    model_type="asr",
                    use_onnx=True,
                    disable_pbar=True,
                    disable_update=True,
                    trust_remote_code=True,
                )
                self._log("[Model] Loaded from local cache!")
                self._warmup_model()
                return
            except Exception as e:
                self._log(f"[Model] Local load error: {e}", file=sys.stderr)

        raise RuntimeError(
            "No local model found!\n"
            f"Checked: ~/.cache/modelscope/hub/models/iic/SenseVoiceSmall\n"
            f"         ~/.cache/sensevoice/\n"
            "Please download the model first, or use --model-dir to specify the path.\n"
            "Download command: python -c \"from funasr import AutoModel; AutoModel(model='iic/SenseVoiceSmall', use_onnx=True)\""
        )

    def _warmup_model(self):
        self._log("[Model] Warming up...")
        try:
            warmup_audio = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            self._model.generate(warmup_audio)
            self._log("[Model] Warm-up complete!")
        except Exception as e:
            self._log(f"[Model] Warm-up note: {e}")

    # ------------------------------------------------------------------ #
    #  Text cleaning / dedup
    # ------------------------------------------------------------------ #

    def _clean_asr_output(self, text):
        if not text:
            return ""
        tokens = ["<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
                  "<|NEUTRAL|>", "<|Happy|>", "<|Sad|>", "<|Angry|>",
                  "<|Speech|>", "<|woitn|>", "<|EMO_UNKNOWN|>",
                  "<|withitn|>", "<|noitn|>", "<|itn|>", "<|nospeech|>"]
        for token in tokens:
            text = text.replace(token, "")
        return text.strip()

    def _is_hallucination(self, text):
        t = text.strip()
        if not t:
            return True
        if len(t) >= 5:
            counts = {}
            for c in t:
                counts[c] = counts.get(c, 0) + 1
            if max(counts.values()) >= len(t) * 0.8:
                return True
        for pattern in ["字幕", "制作人", "by", "索兰娅", "zither", "harp"]:
            if pattern in text.lower():
                return True
        if len(t) <= 1 and t not in "0123456789":
            return True
        return False

    def _remove_internal_repeats(self, text):
        text = text.strip()
        if not text or len(text) < 2:
            return text
        half = len(text) // 2
        for seg_len in range(1, half + 1):
            segment = text[:seg_len]
            repeated = segment * (len(text) // seg_len)
            if repeated == text or text.startswith(repeated):
                return segment
        return text

    def _is_substring_duplicate(self, new_text, old_text):
        new_c = new_text.strip()
        old_c = old_text.strip()
        if not new_c or not old_c:
            return False
        if new_c == old_c:
            return True
        if new_c in old_c:
            return True
        if old_c in new_c:
            return True
        if len(new_c) >= 2 and len(old_c) >= 2:
            shorter = new_c if len(new_c) <= len(old_c) else old_c
            longer = old_c if len(new_c) <= len(old_c) else new_c
            match_len = 0
            for i in range(len(shorter)):
                for j in range(i + 1, len(shorter) + 1):
                    if shorter[i:j] in longer:
                        match_len = max(match_len, j - i)
            if match_len >= len(shorter) * 0.7:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Thread 1: Recorder — capture mic audio to raw PCM file
    # ------------------------------------------------------------------ #

    def _recorder_thread(self):
        """Record from microphone using arecord, write raw PCM to file."""
        # Remove old file
        if os.path.exists(self._pcm_file):
            os.remove(self._pcm_file)

        device_str = f"plughw:{self.device},0" if self.device is not None else "plughw:0,0"

        cmd = [
            "arecord",
            "-r", str(self.sample_rate),
            "-f", "S16_LE",     # 16-bit signed little-endian
            "-c", "1",          # mono
            "-t", "raw",        # raw PCM (no header)
            "-D", device_str,
        ]

        self._log(f"[Recorder] Starting arecord: {' '.join(cmd)}")

        try:
            self._recorder_proc = subprocess.Popen(
                cmd,
                stdout=open(self._pcm_file, "wb"),
                stderr=subprocess.PIPE,
                bufsize=4096,
            )

            # Quick check if arecord started
            import select
            ready, _, _ = select.select([self._recorder_proc.stderr], [], [], 1.0)
            if ready:
                err = self._recorder_proc.stderr.read1(1024).decode('utf-8', errors='ignore')
                if err and ("error" in err.lower() or "fail" in err.lower()):
                    self._log(f"[Recorder] arecord error: {err.strip()}", file=sys.stderr)
                    self._recorder_proc.terminate()
                    self._recorder_proc.wait(timeout=2)
                    # Try hw: prefix as fallback
                    device_str_fallback = f"hw:{self.device},0" if self.device is not None else "hw:0,0"
                    cmd_fallback = [
                        "arecord",
                        "-r", str(self.sample_rate),
                        "-f", "S16_LE",
                        "-c", "1",
                        "-t", "raw",
                        "-D", device_str_fallback,
                    ]
                    self._log(f"[Recorder] Retrying with: {' '.join(cmd_fallback)}")
                    self._recorder_proc = subprocess.Popen(
                        cmd_fallback,
                        stdout=open(self._pcm_file, "wb"),
                        stderr=subprocess.PIPE,
                        bufsize=4096,
                    )

            self._log(f"[Recorder] Recording to {self._pcm_file}")

            while self._running and self._recorder_proc.poll() is None:
                time.sleep(0.1)

            if self._recorder_proc.poll() is None:
                self._recorder_proc.terminate()
                self._recorder_proc.wait(timeout=2)

            self._log("[Recorder] Stopped")

        except Exception as e:
            self._log(f"[Recorder] Error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Thread 2: Transcriber — read PCM file with delay, run ASR
    # ------------------------------------------------------------------ #

    def _transcriber_thread(self):
        """Read growing PCM file with delay, transcribe complete speech segments as sentences."""
        self._load_model()

        bytes_per_sec = self.sample_rate * SAMPLE_WIDTH
        delay_bytes = int(bytes_per_sec * self.delay_seconds)

        self._log(f"[Transcriber] Delay={self.delay_seconds}s")

        while self._running:
            if os.path.exists(self._pcm_file):
                file_size = os.path.getsize(self._pcm_file)
                if file_size >= delay_bytes:
                    break
            time.sleep(0.1)

        if not self._running:
            return

        self._log("[Transcriber] Starting transcription...")

        fd = open(self._pcm_file, "rb")
        read_pos = 0
        output_history = []

        speech_segment = []
        vad_frames = 0
        silence_frames = 0
        frame_size = 512
        frame_bytes = frame_size * SAMPLE_WIDTH

        while self._running:
            try:
                file_size = os.path.getsize(self._pcm_file)
            except OSError:
                time.sleep(0.05)
                continue

            available = file_size - read_pos - delay_bytes

            if available < frame_bytes:
                time.sleep(0.05)
                continue

            fd.seek(read_pos)
            raw_data = fd.read(frame_bytes)
            if len(raw_data) < frame_bytes:
                time.sleep(0.05)
                continue

            read_pos += frame_bytes

            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            if self.audio_gain != 1.0:
                audio_np = np.clip(audio_np * self.audio_gain, -1.0, 1.0)

            rms = np.sqrt(np.mean(audio_np ** 2))

            if rms > self.audio_threshold:
                vad_frames += 1
                silence_frames = 0
                if vad_frames >= 3:
                    speech_segment.append(audio_np)
            else:
                silence_frames += 1
                if vad_frames >= 3:
                    speech_segment.append(audio_np)

            max_frames = int(self.sample_rate * 10 / frame_size)
            if (silence_frames >= 20 or len(speech_segment) >= max_frames) and len(speech_segment) > 0:
                audio_full = np.concatenate(speech_segment)
                min_samples = int(self.sample_rate * 0.3)
                if len(audio_full) >= min_samples:
                    rms_full = np.sqrt(np.mean(audio_full ** 2))
                    if rms_full >= self.audio_threshold * 0.5:
                        try:
                            result = self._model.generate(audio_full)
                            text = result[0].get('text', '').strip() if result and len(result) > 0 else ""
                        except Exception as e:
                            self._log(f"[Transcriber] Error: {e}", file=sys.stderr, flush=True)
                            text = ""

                        if text and not self._is_hallucination(text):
                            text = self._clean_asr_output(text)
                            text = self._remove_internal_repeats(text)
                            text = text.strip()

                            if text:
                                is_dup = False
                                for old in output_history:
                                    if self._is_substring_duplicate(text, old):
                                        is_dup = True
                                        break
                                if not is_dup:
                                    output_history.append(text)
                                    if len(output_history) > 10:
                                        output_history.pop(0)

                                    print(f">>> {text}", flush=True)

                                    if self.enable_commands:
                                        cmd = match_command(text)
                                        if cmd:
                                            write_to_file(f"COMMAND: {cmd} (raw: {text})", self.output_file)
                                        else:
                                            write_to_file(text, self.output_file)
                                    else:
                                        write_to_file(text, self.output_file)

                speech_segment = []
                vad_frames = 0
                silence_frames = 0

        fd.close()

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #

    def run(self):
        self._log("=" * 60)
        self._log("[VoiceRobot] Speak now... Press Ctrl+C to stop")
        self._log(f"[VoiceRobot] Device={self.device}, Delay={self.delay_seconds}s, Rate={self.sample_rate}Hz")
        self._log("=" * 60)

        self._running = True

        # Thread 1: Recorder
        t_recorder = threading.Thread(target=self._recorder_thread, daemon=True)
        t_recorder.start()

        # Thread 2: Transcriber
        t_transcriber = threading.Thread(target=self._transcriber_thread, daemon=True)
        t_transcriber.start()

        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._log("\n[VoiceRobot] Stopping...")
            self._running = False
            time.sleep(1)
            # Cleanup
            if os.path.exists(self._pcm_file):
                try:
                    os.remove(self._pcm_file)
                except Exception:
                    pass
            self._log("[VoiceRobot] Done!")


def list_alsa_devices():
    print("ALSA Capture Hardware Devices:")
    print("-" * 60)
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"arecord error: {result.stderr}")
    except Exception as e:
        print(f"Failed to run arecord -l: {e}")
    print("\nUsage: --device <card_number>")
    print("Example: --device 2  (uses plughw:2,0)")


def main():
    parser = argparse.ArgumentParser(description="Voice Command Robot - 2-thread real-time ASR")
    parser.add_argument("--device", type=int, default=None,
                        help="ALSA card number (e.g. 2 for plughw:2,0)")
    parser.add_argument("--output", default="voice_output.txt",
                        help="Output text file path")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Audio RMS threshold for VAD")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate (default: 16000)")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Fixed audio gain multiplier")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in seconds between recorder and transcriber (default: 1.0)")
    parser.add_argument("--enable-commands", action="store_true",
                        help="Enable action command matching")
    parser.add_argument("--model-dir", default=None,
                        help="Path to local SenseVoice model directory")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress logs, only show RMS and ASR text")
    parser.add_argument("--list-alsa-devices", action="store_true",
                        help="List ALSA capture hardware devices")

    args = parser.parse_args()

    if args.list_alsa_devices:
        list_alsa_devices()
        return

    robot = VoiceRobot(args)
    robot.run()


if __name__ == "__main__":
    main()
