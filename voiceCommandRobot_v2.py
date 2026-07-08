#!/usr/bin/env python3
"""
voiceCommandRobot - STREAMING Voice Transcription
---------------------------------------------------
A robot voice command tool that:
  - Captures audio from microphone continuously
  - Uses THREE audio backends: sounddevice, pyaudio, alsa (arecord)
  - Transcribes speech to text using SenseVoice Small INT8 (offline streaming)
  - Appends transcriptions to a text file
  - Stops ONLY when you press Ctrl+C
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Disable Intel MKL-DNN on ARM devices (RK3588, Raspberry Pi, etc.)
os.environ["MKL_DNN"] = "0"
os.environ["ONEDNN"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import datetime
import argparse
import numpy as np
import queue
import threading
import time
import subprocess
import shutil
import wave
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


class VoiceRobot:
    def __init__(self, args):
        self.model_name = args.model
        self.language = args.language
        self.device = args.device
        self.output_file = args.output
        self.audio_threshold = args.threshold
        self.min_audio_duration = args.min_duration
        self.sample_rate = args.sample_rate
        self.save_audio = args.save_audio
        self.auto_gain = args.auto_gain
        self.audio_gain = args.gain
        self.enable_commands = args.enable_commands
        self.model_dir = args.model_dir
        
        self._model = None
        self._model_type = "sensevoice"
        self._audio_queue = queue.Queue(maxsize=100)
        self._running = False
        self._audio_ready = threading.Event()
        self._current_gain = self.audio_gain
        self._peak_rms_history = []
        self._actual_sample_rate = self.sample_rate
        
        self._setup_onnxruntime()

    def _setup_onnxruntime(self):
        os.environ["ORT_CUDA"] = "0"
        os.environ["ONNXRUNTIME_CUDA"] = "0"
        os.environ["ORT_GPU_DEVICE_ID"] = "-1"

    def _install_sensevoice(self):
        """Install FunASR which includes SenseVoice model."""
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
            ("pip funasr modelscope (Tsinghua)", [
                sys.executable, "-m", "pip", "install", "funasr", "modelscope",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
            ]),
        ]

        for name, cmd in install_methods:
            print(f"[VoiceRobot] Installing via {name}...", flush=True)
            try:
                subprocess.check_call(cmd, timeout=180)
                print(f"[VoiceRobot] Installed via {name}!", flush=True)
                return True
            except Exception as e:
                print(f"[VoiceRobot] Failed via {name}: {e}", flush=True)
                continue

        print("[VoiceRobot] All install methods failed.", file=sys.stderr)
        return False

    def _download_sensevoice_china(self):
        cache_dir = os.path.expanduser("~/.cache/sensevoice")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_names = {
            "small": "sv_s",
            "medium": "sv_m",
            "large": "sv_l",
        }
        
        model_key = model_names.get(self.model_name.lower(), "sv_s")
        model_dir = os.path.join(cache_dir, model_key)
        
        if os.path.exists(model_dir):
            print(f"[VoiceRobot] Found cached SenseVoice model: {model_dir}", flush=True)
            return model_dir

        download_sources = [
            ("ModelScope",
             f"https://www.modelscope.cn/api/v1/models/FunAudioLLM/SenseVoice_{model_key}/repo",
             f"SenseVoice_{model_key}.zip"),
            ("HF Mirror",
             f"https://hf-mirror.com/FunAudioLLM/SenseVoice_{model_key}",
             f"SenseVoice_{model_key}.zip"),
            ("GitHub Release",
             f"https://github.com/FunAudioLLM/SenseVoice/releases/download/v1.0/SenseVoice_{model_key}.zip",
             f"SenseVoice_{model_key}.zip"),
        ]

        for name, url, filename in download_sources:
            print(f"[VoiceRobot] Downloading SenseVoice {self.model_name} from {name}...", flush=True)
            try:
                import urllib.request
                zip_path = os.path.join(cache_dir, filename)
                
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    pct = downloaded * 100 / total_size if total_size > 0 else 0
                    print(f"\r[VoiceRobot] Download: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({pct:.1f}%)",
                          end="", flush=True)
                
                urllib.request.urlretrieve(url + "/archive/refs/heads/main.zip", zip_path, reporthook=report_progress)
                print("\n[VoiceRobot] Download complete! Extracting...", flush=True)
                
                subprocess.run(["unzip", "-o", zip_path, "-d", cache_dir], check=True)
                print("[VoiceRobot] Extraction complete!", flush=True)
                
                extracted_dir = os.path.join(cache_dir, f"SenseVoice_{model_key}-main")
                if os.path.exists(extracted_dir):
                    shutil.move(extracted_dir, model_dir)
                    return model_dir
                    
            except Exception as e:
                print(f"\n[VoiceRobot] Download failed from {name}: {e}", flush=True)
                continue

        print("[VoiceRobot] All download sources failed.", file=sys.stderr)
        return None

    def _find_local_model(self):
        if self.model_dir and os.path.exists(self.model_dir):
            return self.model_dir

        cache_dir = os.path.expanduser("~/.cache/sensevoice")
        model_names = ["sv_s", "sv_m", "sv_l", "SenseVoice_sv_s", "SenseVoice_sv_m"]
        
        for name in model_names:
            model_path = os.path.join(cache_dir, name)
            if os.path.exists(model_path):
                return model_path

        return None

    def _load_model(self):
        if self._model is not None:
            return

        print(f"[VoiceRobot] Loading SenseVoice {self.model_name} INT8 ASR model...")

        # Try to import FunASR (contains SenseVoice)
        try:
            import torch
            # Disable Intel MKL-DNN on ARM - fixes iJIT_NotifyEvent error
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = False
            if hasattr(torch.backends, 'onednn'):
                torch.backends.onednn.enabled = False
            torch.set_num_threads(4)
            from funasr import AutoModel
        except ImportError as e:
            err_msg = str(e)
            if "iJIT_NotifyEvent" in err_msg or "libtorch_cpu" in err_msg:
                print("[VoiceRobot] PyTorch Intel MKL error on ARM device.", file=sys.stderr)
                print("[VoiceRobot] Try: export MKL_DNN=0; export ONEDNN=0", file=sys.stderr)
                print("[VoiceRobot] Or reinstall PyTorch for ARM: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
            print("[VoiceRobot] funasr not installed. Trying to install...", flush=True)
            if self._install_sensevoice():
                import torch
                if hasattr(torch.backends, 'mkldnn'):
                    torch.backends.mkldnn.enabled = False
                if hasattr(torch.backends, 'onednn'):
                    torch.backends.onednn.enabled = False
                torch.set_num_threads(4)
                from funasr import AutoModel
            else:
                raise RuntimeError(
                    "funasr installation failed. "
                    "Install manually: pip install funasr -i https://pypi.tuna.tsinghua.edu.cn/simple"
                )

        local_model_path = self._find_local_model()
        
        # Model IDs for SenseVoice
        model_ids = {
            "small": "iic/SenseVoiceSmall",
            "medium": "iic/SenseVoiceSmall",  # Only small is available
            "large": "iic/SenseVoiceSmall",
        }
        model_id = model_ids.get(self.model_name.lower(), "iic/SenseVoiceSmall")
        
        if local_model_path:
            print(f"[VoiceRobot] Using LOCAL SenseVoice model: {local_model_path}", flush=True)
            try:
                self._model = AutoModel(
                    model=local_model_path,
                    model_type="asr",
                    use_onnx=True,  # ONNX for INT8
                    disable_pbar=True,
                )
                print("[VoiceRobot] SenseVoice model loaded from local cache!")
                self._warmup_model()
                return
            except Exception as e:
                print(f"[VoiceRobot] Error loading local model: {e}", file=sys.stderr)

        try:
            print(f"[VoiceRobot] Downloading SenseVoice model from ModelScope...", flush=True)
            self._model = AutoModel(
                model=model_id,
                model_type="asr",
                use_onnx=True,
                disable_pbar=True,
                hub="ms",  # ModelScope (no VPN needed in China)
            )
            print("[VoiceRobot] SenseVoice model loaded successfully!")
            self._warmup_model()
        except Exception as e:
            print(f"[VoiceRobot] Error loading SenseVoice: {e}", file=sys.stderr)
            print("[VoiceRobot] Trying to download model from mirror sites...", flush=True)
            model_path = self._download_sensevoice_china()
            if model_path:
                try:
                    self._model = AutoModel(
                        model=model_path,
                        model_type="asr",
                        use_onnx=True,
                        disable_pbar=True,
                    )
                    print("[VoiceRobot] SenseVoice model loaded from downloaded file!")
                    self._warmup_model()
                    return
                except Exception as e2:
                    print(f"[VoiceRobot] Error loading downloaded model: {e2}", file=sys.stderr)
            
            raise RuntimeError(
                "Failed to load SenseVoice model. "
                "Download manually from https://www.modelscope.cn/models/iic/SenseVoiceSmall "
                "and place in ~/.cache/sensevoice/"
            )

    def _warmup_model(self):
        print("[VoiceRobot] Warming up SenseVoice model...", flush=True)
        try:
            warmup_audio = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            result = self._model.generate(warmup_audio)
            print("[VoiceRobot] Warm-up complete!", flush=True)
        except Exception as e:
            print(f"[VoiceRobot] Warm-up note: {e}", flush=True)

    def _is_hallucination(self, text):
        text_stripped = text.strip()
        if not text_stripped:
            return True
        
        if len(text_stripped) >= 5:
            char_counts = {}
            for char in text_stripped:
                char_counts[char] = char_counts.get(char, 0) + 1
            max_count = max(char_counts.values())
            if max_count >= len(text_stripped) * 0.8:
                return True
        
        patterns = ["字幕", "制作人", "by", "索兰娅", "zither", "harp"]
        for pattern in patterns:
            if pattern in text.lower():
                return True
        
        if len(text_stripped) <= 1 and text_stripped not in "0123456789":
            return True
        
        return False

    def _sounddevice_callback(self, indata, frames, time_info, status):
        if status:
            pass
        self._audio_queue.put(indata.copy())

    def _capture_sounddevice(self):
        import sounddevice as sd
        print(f"[Audio] Trying to capture at {self.sample_rate} Hz...")
        
        target_device = self.device
        if target_device is not None:
            try:
                sd.default.device[target_device]
            except Exception:
                print(f"[Audio] Device {target_device} not found, using default", flush=True)
                target_device = None

        supported_rates = [self.sample_rate, 48000, 44100, 32000, 24000, 16000]
        
        if target_device is not None:
            try:
                device_info = sd.query_devices(target_device)
                default_rate = device_info.get('default_samplerate', 48000)
                supported_rates = [default_rate] + supported_rates
            except Exception:
                pass

        for sr in supported_rates:
            if not self._running:
                return
            
            print(f"[Audio] Trying sample rate: {sr} Hz...", flush=True)
            try:
                stream = sd.InputStream(
                    samplerate=sr,
                    blocksize=512,
                    device=target_device,
                    channels=1,
                    dtype='float32',
                    callback=self._sounddevice_callback
                )
                stream.start()
                self._actual_sample_rate = sr
                self._audio_ready.set()
                print(f"[Audio] Started sounddevice stream (device={target_device}, rate={sr} Hz)", flush=True)
                
                while self._running:
                    time.sleep(0.1)
                
                stream.stop()
                stream.close()
                print("[Audio] Capture stopped", flush=True)
                return
            except Exception as e:
                print(f"[Audio] Failed at {sr} Hz: {e}", file=sys.stderr)
                continue
        
        print("[Audio] No supported sample rate found!", file=sys.stderr)
        self._try_next_backend()

    def _capture_pyaudio(self):
        try:
            import pyaudio
        except ImportError:
            print("[Audio] pyaudio not installed", file=sys.stderr)
            self._try_next_backend()
            return

        p = pyaudio.PyAudio()
        
        target_device = self.device
        if target_device is not None:
            try:
                info = p.get_device_info_by_index(target_device)
                print(f"[Audio] Using device: {info['name']}", flush=True)
            except Exception:
                print(f"[Audio] Device {target_device} not found, using default", flush=True)
                target_device = None

        supported_rates = [self.sample_rate, 48000, 44100, 32000, 24000, 16000]

        for sr in supported_rates:
            if not self._running:
                p.terminate()
                return
            
            print(f"[Audio] Trying pyaudio sample rate: {sr} Hz...", flush=True)
            try:
                stream = p.open(
                    rate=sr,
                    channels=1,
                    format=pyaudio.paFloat32,
                    input=True,
                    input_device_index=target_device,
                    frames_per_buffer=512,
                )
                self._actual_sample_rate = sr
                self._audio_ready.set()
                print(f"[Audio] Started pyaudio stream (device={target_device}, rate={sr} Hz)", flush=True)
                
                while self._running:
                    data = stream.read(512, exception_on_overflow=False)
                    audio_np = np.frombuffer(data, dtype=np.float32)
                    self._audio_queue.put(audio_np)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                print("[Audio] Capture stopped", flush=True)
                return
            except Exception as e:
                print(f"[Audio] pyaudio failed at {sr} Hz: {e}", file=sys.stderr)
                continue
        
        p.terminate()
        print("[Audio] No supported sample rate found for pyaudio!", file=sys.stderr)
        self._try_next_backend()

    def _capture_alsa(self):
        # Check available ALSA devices if specific device requested
        if self.device is not None:
            try:
                result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
                if f"card {self.device}:" not in result.stdout:
                    print(f"[Audio] WARNING: Card {self.device} not found in ALSA devices!", file=sys.stderr)
                    print(f"[Audio] Available devices:\n{result.stdout}", file=sys.stderr)
                    print(f"[Audio] Falling back to default device...", file=sys.stderr)
                    self.device = None
            except Exception as e:
                print(f"[Audio] Could not check ALSA devices: {e}", file=sys.stderr)

        supported_rates = [self.sample_rate, 48000, 44100, 32000, 24000, 16000]
        
        # Try plughw first (software resampling), then hw
        device_prefixes = ["plughw", "hw"]
        
        for prefix in device_prefixes:
            if not self._running:
                return
            
            for sr in supported_rates:
                if not self._running:
                    return
                
                device_str = f"{prefix}:{self.device},0" if self.device is not None else f"{prefix}:0,0"
                print(f"[Audio] Trying ALSA {device_str} at {sr} Hz...")
                
                cmd = ["arecord", "-r", str(sr), "-f", "FLOAT_LE", "-c", "1", "-t", "raw", "-D", device_str]
                
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=4096
                    )
                    
                    # Quick test if arecord started successfully
                    import select
                    ready, _, _ = select.select([proc.stderr], [], [], 0.5)
                    if ready:
                        err = proc.stderr.read1(1024).decode('utf-8', errors='ignore')
                        if err and ("error" in err.lower() or "fail" in err.lower() or "busy" in err.lower() or "no such" in err.lower()):
                            print(f"[Audio] ALSA {device_str} failed at {sr} Hz: {err.strip()}", file=sys.stderr)
                            proc.terminate()
                            proc.wait(timeout=1)
                            continue
                    
                    self._actual_sample_rate = sr
                    self._audio_ready.set()
                    print(f"[Audio] Started ALSA capture ({device_str}, rate={sr} Hz)", flush=True)
                    
                    while self._running and proc.poll() is None:
                        data = proc.stdout.read(1024)
                        if not data:
                            time.sleep(0.01)
                            continue
                        audio_np = np.frombuffer(data, dtype=np.float32)
                        self._audio_queue.put(audio_np)
                    
                    proc.terminate()
                    proc.wait(timeout=2)
                    print("[Audio] Capture stopped", flush=True)
                    return
                except Exception as e:
                    print(f"[Audio] ALSA {device_str} failed at {sr} Hz: {e}", file=sys.stderr)
                    continue
        
        print("[Audio] No supported sample rate found for ALSA!", file=sys.stderr)
        self._try_next_backend()

    def _try_next_backend(self):
        if self._running:
            print("[Audio] Trying next backend...", flush=True)
            self._audio_ready.clear()
            self._start_capture()

    def _start_capture(self):
        backends = ['alsa', 'sounddevice']
        
        for backend in backends:
            if self._running:
                print(f"[Audio] Trying {backend}...", flush=True)
                if backend == 'sounddevice':
                    self._capture_sounddevice()
                elif backend == 'alsa':
                    self._capture_alsa()

    def _fast_resample(self, audio, old_sr, new_sr):
        import math
        if old_sr == new_sr:
            return audio
        ratio = float(new_sr) / old_sr
        new_len = int(math.ceil(len(audio) * ratio))
        indices = np.arange(new_len) / ratio
        return np.interp(indices, np.arange(len(audio)), audio)

    def _save_debug_audio(self, audio, sample_rate):
        with wave.open("debug_audio.wav", "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

    def _transcribe_loop(self):
        self._load_model()
        
        buffer_size_seconds = 2.0
        buffer = deque(maxlen=int(self.sample_rate * buffer_size_seconds))
        transcribe_interval = int(self.sample_rate * 0.5)
        new_samples_since_last = 0
        
        last_text = ""
        repeat_count = 0
        
        vad_active = False
        vad_frames = 0
        silence_frames = 0
        frame_size = 512
        
        print(f"[Transcribe] Buffer: {buffer_size_seconds}s, Interval: {transcribe_interval/self.sample_rate:.1f}s", flush=True)
        print(f"[Transcribe] Waiting for audio...", flush=True)
        
        loop_count = 0
        audio_chunk_count = 0
        last_status_time = time.time()
        saved_audio = []
        saved_audio_max = int(self.sample_rate * 10)

        while self._running:
            loop_count += 1
            
            try:
                chunks = []
                try:
                    while True:
                        chunk = self._audio_queue.get_nowait()
                        chunks.append(chunk)
                except queue.Empty:
                    pass
                
                if not chunks:
                    time.sleep(0.005)
                    
                    if time.time() - last_status_time > 5.0:
                        rms_display = 0.0
                        if len(buffer) > 0:
                            recent = np.array(list(buffer)[-512:])
                            rms_display = np.sqrt(np.mean(recent ** 2))
                        gain_str = f", Gain: {self._current_gain:.1f}x" if (self.auto_gain or self._current_gain != 1.0) else ""
                        print(f"[Status] Loop #{loop_count}, Chunks: {audio_chunk_count}, Buf: {len(buffer)}, RMS: {rms_display:.4f}, VAD: {vad_active}{gain_str}", flush=True)
                        last_status_time = time.time()
                    continue
            except Exception:
                continue

            audio_chunk_count += len(chunks)
            total_new_samples = 0
            chunk_rms_max = 0.0
            for chunk in chunks:
                if chunk.dtype == np.int16:
                    chunk_float = chunk.astype(np.float32) / 32768.0
                else:
                    chunk_float = chunk.astype(np.float32)
                
                actual_sr = getattr(self, '_actual_sample_rate', self.sample_rate)
                if actual_sr != self.sample_rate:
                    chunk_float = self._fast_resample(chunk_float.flatten(), actual_sr, self.sample_rate)
                else:
                    chunk_float = chunk_float.flatten()
                
                chunk_rms = np.sqrt(np.mean(chunk_float ** 2))
                if chunk_rms > chunk_rms_max:
                    chunk_rms_max = chunk_rms
                
                if self.auto_gain or self._current_gain != 1.0:
                    chunk_float = np.clip(chunk_float * self._current_gain, -1.0, 1.0)
                
                new_len = len(chunk_float)
                total_new_samples += new_len
                buffer.extend(chunk_float)
                
                if self.save_audio and len(saved_audio) < saved_audio_max:
                    saved_audio.append(chunk_float.copy())
                    if len(saved_audio) * len(chunk_float) >= saved_audio_max and not hasattr(self, '_audio_saved_notified'):
                        self._audio_saved_notified = True
                        all_audio = np.concatenate(saved_audio)[:saved_audio_max]
                        self._save_debug_audio(all_audio, self.sample_rate)
                        print(f"[Debug] Saved 10s debug audio to debug_audio.wav", flush=True)
                
                for i in range(0, new_len - frame_size, frame_size):
                    frame = chunk_float[i:i + frame_size]
                    if len(frame) < frame_size:
                        break
                    rms = np.sqrt(np.mean(frame ** 2))
                    if rms > self.audio_threshold:
                        vad_frames += 1
                        silence_frames = 0
                        if vad_frames >= 3:
                            vad_active = True
                    else:
                        silence_frames += 1
                        if silence_frames >= 30:
                            vad_active = False
                            vad_frames = 0
            
            if self.auto_gain and chunk_rms_max > 0:
                self._peak_rms_history.append(chunk_rms_max)
                if len(self._peak_rms_history) > 20:
                    self._peak_rms_history.pop(0)
                if len(self._peak_rms_history) >= 5:
                    avg_peak = sum(self._peak_rms_history) / len(self._peak_rms_history)
                    target_rms = 0.05
                    if avg_peak > 0:
                        desired_gain = target_rms / avg_peak
                        desired_gain = max(1.0, min(desired_gain, 50.0))
                        new_gain = self._current_gain * 0.8 + desired_gain * 0.2
                        if abs(new_gain - self._current_gain) > 0.5:
                            self._current_gain = new_gain
                            print(f"[AutoGain] Adjusting gain to {self._current_gain:.1f}x (avg peak RMS={avg_peak:.4f})", flush=True)
            
            new_samples_since_last += total_new_samples
            
            if new_samples_since_last < transcribe_interval:
                continue
            
            if not vad_active and len(buffer) < int(self.sample_rate * 0.5):
                new_samples_since_last = 0
                continue
            
            min_samples = int(self.sample_rate * self.min_audio_duration)
            if len(buffer) < min_samples:
                continue
            
            new_samples_since_last = 0
            audio_np = np.array(buffer, dtype=np.float32)
            
            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms < self.audio_threshold * 0.7:
                continue

            print(f"[Transcribe] Processing {len(audio_np)/self.sample_rate:.2f}s, RMS={rms:.4f}, VAD={vad_active}", flush=True)

            try:
                result = self._model.generate(audio_np)
                # FunASR returns list of dicts with 'text' key
                if result and len(result) > 0:
                    text = result[0].get('text', '').strip()
                else:
                    text = ""
            except Exception as e:
                print(f"[Transcribe] Error: {e}", file=sys.stderr, flush=True)
                continue

            if not text:
                print(f"[Transcribe] (empty result - audio too quiet?)", flush=True)
                continue

            if self._is_hallucination(text):
                print(f"[Transcribe] (filtered: '{text[:30]}...')", flush=True)
                continue
            
            if text == last_text:
                repeat_count += 1
                if repeat_count >= 2:
                    continue
            else:
                repeat_count = 0
                last_text = text

            if text:
                print(f"\n>>> {text}\n", flush=True)

                if self.enable_commands:
                    cmd = match_command(text)
                    if cmd:
                        print(f"[Command] -> {cmd}", flush=True)
                        write_to_file(f"COMMAND: {cmd} (raw: {text})", self.output_file)
                    else:
                        write_to_file(text, self.output_file)
                else:
                    write_to_file(text, self.output_file)

    def run(self):
        print("=" * 60)
        print("[Listening] Speak now... Press Ctrl+C to stop")
        print("=" * 60)
        
        self._running = True
        
        capture_thread = threading.Thread(target=self._start_capture)
        capture_thread.daemon = True
        capture_thread.start()
        
        transcribe_thread = threading.Thread(target=self._transcribe_loop)
        transcribe_thread.daemon = True
        transcribe_thread.start()
        
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[VoiceRobot] Stopping...", flush=True)
            self._running = False
            time.sleep(1)
            print("[VoiceRobot] Done!", flush=True)


def list_devices():
    print("Available audio devices:")
    print("-" * 60)
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            print(f"Device {i}: {dev['name']}")
            print(f"           Input: {dev['max_input_channels']}ch, Output: {dev['max_output_channels']}ch")
            print(f"           Default SR: {dev['default_samplerate']} Hz")
            print()
    except Exception as e:
        print(f"sounddevice error: {e}")
        print()


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
    
    print("\nUsage: Use --device <card_number> to select a card")
    print("Example: --device 1  (uses hw:1,0 or plughw:1,0)")


def main():
    parser = argparse.ArgumentParser(description="Voice Command Robot")
    parser.add_argument("--model", default="small", choices=["small", "medium", "large"],
                        help="SenseVoice model size (default: small)")
    parser.add_argument("--language", default="zh", choices=["zh", "en", "ja", "ko", "yue"],
                        help="Language for ASR (default: zh)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio device index")
    parser.add_argument("--output", default="voice_output.txt",
                        help="Output file path")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Audio RMS threshold for VAD")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum audio duration before transcribing")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate")
    parser.add_argument("--save-audio", action="store_true",
                        help="Save first 10s of audio for debugging")
    parser.add_argument("--auto-gain", action="store_true",
                        help="Auto-adjust audio gain")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Fixed audio gain multiplier")
    parser.add_argument("--enable-commands", action="store_true",
                        help="Enable action command matching")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices")
    parser.add_argument("--list-alsa-devices", action="store_true",
                        help="List ALSA capture hardware devices")
    parser.add_argument("--model-dir", default=None,
                        help="Path to local SenseVoice model directory")
    
    args = parser.parse_args()
    
    if args.list_alsa_devices:
        list_alsa_devices()
        return
    
    if args.list_devices:
        list_devices()
        return
    
    robot = VoiceRobot(args)
    robot.run()


if __name__ == "__main__":
    main()
