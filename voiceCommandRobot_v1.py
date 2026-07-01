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
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np


# ------------------------------------------------------------------
# Part 1 — Get voice from microphone
# ------------------------------------------------------------------

def detect_backend(preferred: str = None) -> str:
    """Detect which audio backend is available.

    Returns one of: "sounddevice", "pyaudio", "alsa" (arecord), or "none".
    Checks the preferred backend first; if not available, tries the others.
    """
    order = []
    if preferred:
        order.append(preferred)
    order += [b for b in ("sounddevice", "pyaudio", "alsa") if b not in order]

    for backend in order:
        if backend == "sounddevice":
            try:
                import sounddevice as sd  # noqa: F401
                sd.query_devices()
                return "sounddevice"
            except Exception:
                continue
        elif backend == "pyaudio":
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                count = p.get_device_count()
                p.terminate()
                if count > 0:
                    return "pyaudio"
            except Exception:
                continue
        elif backend == "alsa":
            if shutil.which("arecord"):
                return "alsa"
    return "none"


def list_pyaudio_devices() -> list:
    """List pyaudio capture devices.

    Returns a list of dicts: [{index, name, sample_rate, channels}, ...]
    """
    devices = []
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get('max_input_channels', 0)) > 0:
                devices.append({
                    "index": i,
                    "name": info.get('name', 'unknown'),
                    "sample_rate": int(info.get('defaultSampleRate', 16000)),
                    "channels": int(info.get('max_input_channels', 1)),
                })
        p.terminate()
    except Exception:
        pass
    return devices


def record_with_pyaudio(duration_seconds: float = 5.0,
                       sample_rate: int = 16000,
                       channels: int = 1,
                       device: int = None):
    """Record audio using pyaudio, returns NumPy array instead of WAV file.

    Args:
        duration_seconds: Recording duration in seconds.
        sample_rate:      Target sample rate.
        channels:         Number of channels (default 1).
        device:           pyaudio device index. None = default.

    Returns: (audio_array, actual_sample_rate, info_dict)
    """
    import pyaudio

    FORMAT = pyaudio.paInt16
    CHUNK = 512

    p = pyaudio.PyAudio()

    dev_index = device
    dev_name = "default"
    actual_sr = sample_rate
    if dev_index is not None:
        try:
            info = p.get_device_info_by_index(dev_index)
            dev_name = info.get('name', 'unknown')
        except Exception:
            pass

    stream = None
    # Try to open stream at target sample rate first
    try:
        print(f"[Part 1] Recording with pyaudio from device {dev_index or 'default'} "
              f"'{dev_name}' for {duration_seconds}s "
              f"(sr={actual_sr}, ch={channels}) ...")
        stream = p.open(format=FORMAT,
                        channels=channels,
                        rate=actual_sr,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=CHUNK)
    except Exception:
        # If that fails, try using the device's default sample rate
        print(f"[Part 1] Device {dev_index or 'default'} does not support {sample_rate} Hz. "
              f"Trying default samplerate...")
        if dev_index is None:
            # Get default input device info
            info = p.get_device_info_by_index(p.get_default_input_device_info()['index'])
        else:
            info = p.get_device_info_by_index(dev_index)
        actual_sr = int(info.get('defaultSampleRate', 44100))
        print(f"[Part 1] Recording with pyaudio from device {dev_index or 'default'} "
              f"'{dev_name}' for {duration_seconds}s "
              f"(sr={actual_sr}, target={sample_rate}, ch={channels}) ...")
        stream = p.open(format=FORMAT,
                        channels=channels,
                        rate=actual_sr,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=CHUNK)

    frames = []
    try:
        for _ in range(0, int(actual_sr / CHUNK * duration_seconds)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except Exception as exc:
        stream.stop_stream()
        stream.close()
        p.terminate()
        raise RuntimeError(f"pyaudio recording failed: {exc}") from exc

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert raw frames to numpy array
    raw = b''.join(frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]

    # Resample if needed
    if actual_sr != sample_rate:
        print(f"[Part 1] Resampling from {actual_sr} Hz to {sample_rate} Hz ...")
        audio = audio.astype("float32") / 32768.0
        target_samples = int(len(audio) * sample_rate / actual_sr)
        from scipy.signal import resample
        audio = resample(audio, target_samples)
        audio = (audio * 32768.0).astype("int16")

    # Compute RMS
    peak_rms = int(np.sqrt(np.mean(audio.astype("float64") ** 2)) or 0)

    info = {
        "device_id": dev_index,
        "device_name": dev_name,
        "backend": "pyaudio",
        "sample_rate": sample_rate,
        "actual_sample_rate": actual_sr,
        "resampled": actual_sr != sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_rms": peak_rms,
    }
    print(f"[Part 1] Recorded successfully (peak_rms={peak_rms})")
    return audio, sample_rate, info


def list_alsa_devices() -> list:
    """List ALSA capture devices using arecord.

    Returns a list of dicts: [{index, name, hw_addr}, ...]
    """
    devices = []
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("card ") and ": " in line and ", device " in line:
                # Parse: "card 0: name [longname], device 0: devname [...]"
                card_part = line.strip().split(", device ")[0]  # "card 0: name [longname]"
                rest = line.strip().split(", device ", 1)[1]   # "0: devname [...]"

                # Card number
                card_num = card_part.split()[1].rstrip(":")

                # Device number and name
                dev_parts = rest.split(":", 1)
                dev_num = dev_parts[0].strip()
                name = dev_parts[1].strip() if len(dev_parts) > 1 else rest

                hw_addr = f"hw:{card_num},{dev_num}"
                devices.append({
                    "index": len(devices),
                    "name": name,
                    "hw_addr": hw_addr,
                    "card": card_num,
                    "device": dev_num,
                })
    except Exception as exc:
        print(f"[Part 1] Warning: could not list ALSA devices: {exc}")
    return devices


def record_with_arecord(duration_seconds: float = 5.0,
                       sample_rate: int = 16000,
                       channels: int = 1,
                       device: str = None):
    """Record audio using ALSA's arecord command, returns NumPy array.

    Args:
        duration_seconds: Recording duration in seconds.
        sample_rate:      Target sample rate. Uses plughw for auto-resampling.
        channels:         Number of channels (default 1).
        device:           ALSA device string like "hw:1,0" or "plughw:1,0".
                          If None, uses default. Can also be a numeric index
                          into the list returned by list_alsa_devices().

    Returns: (audio_array, actual_sample_rate, info_dict)
    """
    if device is None:
        dev_str = "default"
    elif isinstance(device, int):
        alsa_devs = list_alsa_devices()
        if device >= len(alsa_devs):
            raise ValueError(
                f"ALSA device index {device} out of range. "
                f"Found {len(alsa_devs)} devices. "
                f"Run --list-devices to see available ones."
            )
        dev_str = "plughw:" + alsa_devs[device]["hw_addr"].split(":")[1]
        dev_name = alsa_devs[device]["name"]
    else:
        dev_str = device
        dev_name = device

    # Use plughw if raw hw given, so ALSA auto-handles sample rate / format
    if dev_str.startswith("hw:"):
        dev_str = "plug" + dev_str

    # Use temp WAV file as intermediate
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    cmd = [
        "arecord",
        "-D", dev_str,
        "-d", str(int(duration_seconds)),
        "-r", str(sample_rate),
        "-f", "S16_LE",
        "-c", str(channels),
        "-t", "wav",
        tmp_wav,
    ]

    print(f"[Part 1] Recording with arecord from device '{dev_str}' "
          f"for {duration_seconds}s (sr={sample_rate}, ch={channels}) ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=duration_seconds + 5)
        if result.returncode != 0:
            raise RuntimeError(
                f"arecord failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"arecord timed out: {exc}") from exc

    if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
        raise RuntimeError("arecord produced no output file")

    # Read WAV to NumPy array
    with wave.open(tmp_wav, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch)[:, 0]
    peak_rms = int(np.sqrt(np.mean(audio.astype("float64") ** 2)) or 0)

    os.unlink(tmp_wav)

    info = {
        "device_id": dev_str,
        "device_name": dev_name if 'dev_name' in dir() else dev_str,
        "backend": "alsa",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_rms": peak_rms,
    }
    print(f"[Part 1] Recorded successfully (peak_rms={peak_rms})")
    return audio, sample_rate, info


def record_from_microphone(duration_seconds: float = 5.0,
                          sample_rate: int = 16000,
                          channels: int = 1,
                          device: int = None):
    """Capture audio from microphone, returns NumPy array instead of WAV file.

    Args:
        duration_seconds: Recording duration in seconds.
        sample_rate:      Target PCM sample rate (default 16000 — required by Whisper).
                          If the device doesn't support this, we'll record at
                          the closest supported rate and resample.
        channels:         Number of audio channels (default 1 = mono).
        device:           Sounddevice device index. None = default microphone.
                          Use --list-devices to see available device IDs.

    Returns: (audio_array, actual_sample_rate, info_dict)
    """
    import sounddevice as sd

    dev_info = sd.query_devices(device=device, kind="input")
    dev_name = dev_info.get("name", "unknown") if dev_info else "default"
    dev_id = device if device is not None else "default"

    actual_sr = sample_rate
    try:
        sd.check_input_settings(device=device, samplerate=sample_rate,
                              channels=channels)
    except Exception:
        print(f"[Part 1] Device {dev_id} does not support {sample_rate} Hz. "
              f"Trying default samplerate...")
        actual_sr = int(dev_info.get("defaultSampleRate", 44100))
        try:
            sd.check_input_settings(device=device, samplerate=actual_sr,
                                  channels=channels)
        except Exception as exc:
            raise RuntimeError(
                f"Device {dev_id} does not support {actual_sr} Hz either. "
                f"Available sample rates: {sd.query_devices(device=device).get('sample_rates')}"
            ) from exc

    print(f"[Part 1] Recording from device {dev_id} '{dev_name}' "
          f"for {duration_seconds}s (sr={actual_sr}, target={sample_rate}, ch={channels}) ...")
    num_samples = int(duration_seconds * actual_sr)
    audio = sd.rec(num_samples, samplerate=actual_sr,
                  channels=channels, dtype="int16",
                  device=device)
    sd.wait()
    audio = np.squeeze(audio)

    if actual_sr != sample_rate:
        print(f"[Part 1] Resampling from {actual_sr} Hz to {sample_rate} Hz ...")
        audio = audio.astype("float32") / 32768.0
        target_samples = int(len(audio) * sample_rate / actual_sr)
        from scipy.signal import resample
        audio = resample(audio, target_samples)
        audio = (audio * 32768.0).astype("int16")

    peak_rms = int(np.sqrt(np.mean(audio.astype("float64") ** 2)) or 0)

    info = {
        "device_id": dev_id,
        "device_name": dev_name,
        "sample_rate": sample_rate,
        "actual_sample_rate": actual_sr,
        "resampled": actual_sr != sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_rms": peak_rms,
    }
    print(f"[Part 1] Recorded successfully (peak_rms={peak_rms})")
    return audio, sample_rate, info


# ------------------------------------------------------------------
# Noise Reduction Helper Function
# ------------------------------------------------------------------

def reduce_background_noise(audio_array: np.ndarray,
                           noise_array: np.ndarray,
                           sample_rate: int) -> np.ndarray:
    """Reduce background noise using noisereduce library."""
    import noisereduce as nr
    try:
        # noisereduce expects float32 in [-1, 1]
        audio_float = audio_array.astype(np.float32) / 32768.0
        noise_float = noise_array.astype(np.float32) / 32768.0

        # Reduce noise
        reduced_audio = nr.reduce_noise(
            y=audio_float,
            y_noise=noise_float,
            sr=sample_rate
        )

        # Convert back to int16
        return (reduced_audio * 32768.0).astype(np.int16)
    except Exception as e:
        print(f"Warning: Noise reduction failed - {e}", file=sys.stderr)
        return audio_array


# ------------------------------------------------------------------
# Action Command Dictionary & Similarity Matching
# ------------------------------------------------------------------

ACTION_COMMANDS = {
    "前进": ["前进", "向前", "往前", "走", "前进前进", "向前走", "往前走走", "向前进", "请前进", "up", "move up", "go up", "forward"],
    "后退": ["后退", "向后", "往后", "退", "后退后退", "向后退", "往后退退", "向后走", "请后退", "退后", "down", "move down", "go down", "backward"],
    "左转": ["左转", "向左转", "往左", "左拐", "左转左转", "向左拐", "往左拐", "请左转", "left", "turn left", "go left"],
    "右转": ["右转", "向右转", "往右", "右拐", "右转右转", "向右拐", "往右拐", "请右转", "right", "turn right", "go right"],
    "停止": ["停止", "停", "停下", "站住", "停一下", "停下来", "请停止", "不要动", "别动", "stop", "halt", "pause", "stop moving"],
    "抓取": ["抓取", "抓", "抓住", "拿", "拿起", "捡", "捡起", "取", "请抓取", "抓东西", "抓起来", "grab", "pick up", "take", "catch"],
    "释放": ["释放", "放", "放开", "放下", "松开", "丢掉", "扔", "请释放", "放下来", "松开手", "release", "let go", "drop", "put down"],
}

COMMAND_ENGLISH = {
    "前进": "up",
    "后退": "down",
    "左转": "left",
    "右转": "right",
    "停止": "stop",
    "抓取": "grab",
    "释放": "release",
}


def _char_set_similarity(text: str, phrase: str) -> float:
    if not text or not phrase:
        return 0.0
    t_chars = set(text)
    p_chars = set(phrase)
    if not p_chars:
        return 0.0
    return len(t_chars & p_chars) / len(p_chars)


def _substring_similarity(text: str, phrase: str) -> float:
    if not text or not phrase:
        return 0.0
    if phrase == text:
        return 1.0
    words = text.split()
    phrase_words = phrase.split()
    if len(phrase_words) > 1:
        if phrase in text:
            return 1.0
        return 0.0
    if phrase in words:
        return 1.0
    max_match = 0
    for i in range(len(phrase)):
        for j in range(i + 1, len(phrase) + 1):
            sub = phrase[i:j]
            if sub in text:
                max_match = max(max_match, len(sub))
    return max_match / len(phrase) if len(phrase) > 0 else 0.0


def match_action_command(text: str, threshold: float = 0.5) -> tuple:
    if not text:
        return "", ""
    text_clean = text.strip().replace(" ", "").replace("，", "").replace("。", "").replace("！", "").replace("？", "")
    text_with_space = text.strip().replace("，", "").replace("。", "").replace("！", "").replace("？", "")
    best_cmd = ""
    best_score = 0.0
    best_text_coverage = 0.0
    all_aliases = []
    for cmd, aliases in ACTION_COMMANDS.items():
        for alias in aliases:
            all_aliases.append((len(alias), cmd, alias))
    all_aliases.sort(key=lambda x: (-x[0], x[1]))
    for _, cmd, alias in all_aliases:
        alias_clean = alias.replace(" ", "")
        s1 = _char_set_similarity(text_clean, alias_clean)
        s2 = _substring_similarity(text_with_space, alias)
        score = 0.4 * s1 + 0.6 * s2
        coverage = len(alias_clean) / len(text_clean) if len(text_clean) > 0 else 0.0
        if score > best_score or (score == best_score and coverage > best_text_coverage):
            best_score = score
            best_text_coverage = coverage
            best_cmd = cmd
    if best_score >= threshold:
        return best_cmd, COMMAND_ENGLISH.get(best_cmd, "")
    return "", ""


# ------------------------------------------------------------------
# Part 2 — Offline Whisper ASR + write to text file
# ------------------------------------------------------------------

def transcribe_with_whisper(audio_array,
                           sample_rate,
                           model_name: str = "tiny",
                           language: str = None,
                           optimize: bool = False) -> str:
    """Run off-line Whisper ASR on NumPy array directly.

    Args:
        audio_array: NumPy array of audio samples
        sample_rate: Sample rate of audio (should be 16000 for Whisper)
        model_name: Name of Whisper model to use
        language: Language hint to help Whisper
        optimize: Enable speed optimizations for Whisper

    Returns the transcription text.
    """
    import whisper

    # Convert to float32 in [-1, 1] range (what Whisper expects)
    audio_float = audio_array.astype(np.float32) / 32768.0

    kwargs = {"fp16": False}
    if language:
        kwargs["language"] = language

    if optimize:
        # Apply speed optimizations
        kwargs.update({
            "beam_size": 1,
            "best_of": 1,
            "condition_on_previous_text": False,
            "temperature": 0.0,
            "word_timestamps": False,
            "verbose": False,
        })

    print(f"[Part 2] Transcribing with Whisper (model '{model_name}') ...")
    # Pass NumPy array directly to Whisper's transcribe function via 'audio' parameter
    result = whisper.load_model(model_name).transcribe(audio_float, **kwargs)
    text = (result.get("text") or "").strip()
    print(f"[Part 2] Transcription result: '{text}'")
    return text


def timestamp() -> str:
    """Return current timestamp string: YYYY-MM-DD HH:MM:SS"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_to_text_file(text: str,
                      output_path: str,
                      append: bool = True) -> str:
    """Persist the transcribed text to a plain text file.

    Each entry is timestamped so repeated runs accumulate.
    Returns the output file path for the caller to report.
    """
    mode = "a" if append and os.path.exists(output_path) else "w"
    stamp = timestamp()
    line = f"[{stamp}] {text}\n"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(line)
    return output_path


# ------------------------------------------------------------------
# Skill-level public API
# ------------------------------------------------------------------

class voiceCommandRobot:
    """Tiny skill: mic -> audio array -> Whisper -> text file (Chinese-only by default)."""

    def __init__(self,
                 model_name: str = "tiny",
                 output_file: str = "robot_transcripts.txt",
                 device=None,
                 backend: str = "auto",
                 optimize: bool = False,
                 language: str = "zh",
                 reduce_noise: bool = False,
                 noise_duration: float = 1.0,
                 enable_commands: bool = False,
                 command_threshold: float = 0.5):
        self.model_name = model_name
        self.output_file = os.path.abspath(output_file)
        self.device = device
        self.backend = backend
        self.optimize = optimize
        self.language = language
        self.reduce_noise = reduce_noise
        self.noise_duration = noise_duration
        self.enable_commands = enable_commands
        self.command_threshold = command_threshold
        self._resolved_backend = None
        self._whisper_model = None

    def _resolve_backend(self) -> str:
        if self._resolved_backend is None:
            preferred = None if self.backend == "auto" else self.backend
            self._resolved_backend = detect_backend(preferred=preferred)
            if self._resolved_backend == "none":
                raise RuntimeError(
                    "No audio recording backend available. "
                    "Install sounddevice (pip install sounddevice) or "
                    "install ALSA (sudo apt-get install alsa-utils)."
                )
            print(f"[Part 1] Using audio backend: {self._resolved_backend}")
        return self._resolved_backend

    def _get_whisper_model(self, language: str = None):
        """Load (once) and cache the Whisper model for fast repeated inference."""
        if self._whisper_model is None:
            print(f"[Part 2] Loading Whisper model '{self.model_name}' ...")
            try:
                import whisper
                self._whisper_model = whisper.load_model(self.model_name)
            except Exception as exc:
                if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSL" in str(exc):
                    print("[Part 2] SSL error. Retrying with verification disabled.")
                    import ssl
                    try:
                        ssl._create_default_https_context = ssl._create_unverified_context
                    except AttributeError:
                        pass
                    import whisper
                    self._whisper_model = whisper.load_model(self.model_name)
                else:
                    raise
        return self._whisper_model

    def _record_audio(self, duration: float) -> tuple:
        """Record audio from microphone, optionally with noise reduction.

        Returns (audio_array, sample_rate, info_dict).
        """
        backend = self._resolve_backend()

        if backend == "sounddevice":
            audio, sr, info = record_from_microphone(
                duration_seconds=duration,
                device=self.device
            )
        elif backend == "alsa":
            audio, sr, info = record_with_arecord(
                duration_seconds=duration,
                device=self.device
            )
        elif backend == "pyaudio":
            audio, sr, info = record_with_pyaudio(
                duration_seconds=duration,
                device=self.device
            )
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        return audio, sr, info

    def run_once(self, duration: float = 5.0,
                language: str = None,
                audio_path: str = None) -> dict:
        if language is None:
            language = self.language
        """Execute the two-part pipeline. Returns a JSON-serializable result."""
        result = {
            "skill": "voiceCommandRobot",
            "ok": False,
            "text": "",
            "audio_file": None,
            "device_id": self.device,
            "backend": None,
            "output_file": self.output_file,
            "error": None,
        }

        # Part 1 — get voice
        try:
            if audio_path:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(audio_path)
                print(f"[Part 1] Using pre-recorded audio: {audio_path}")
                with wave.open(audio_path, "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16)
                if ch > 1:
                    audio = audio.reshape(-1, ch)[:, 0]
                info = {"file": audio_path, "mode": "pre-recorded"}
                actual_sr = sr
            else:
                backend = self._resolve_backend()
                result["backend"] = backend

                # Record noise sample first if noise reduction is enabled
                if self.reduce_noise:
                    print(f"[Part 1] Recording noise sample ({self.noise_duration}s) ...")
                    noise_audio, noise_sr, _ = self._record_audio(self.noise_duration)

                # Record main audio
                audio, actual_sr, info = self._record_audio(duration)

                # Apply noise reduction if enabled
                if self.reduce_noise:
                    print("[Part 1] Reducing background noise ...")
                    audio = reduce_background_noise(audio, noise_audio, actual_sr)
            if "device_name" in info:
                result["device_name"] = info["device_name"]
        except Exception as exc:
            result["error"] = f"Part 1 (mic) failed: {exc}"
            print(f"[voiceCommandRobot] {result['error']}",
                  file=sys.stderr)
            return result

        # Part 2 — Whisper ASR (use cached model for speed)
        try:
            model = self._get_whisper_model(language)
            kwargs = {"fp16": False}
            if language:
                kwargs["language"] = language

            if self.optimize:
                kwargs.update({
                    "beam_size": 1,
                    "best_of": 1,
                    "condition_on_previous_text": False,
                    "temperature": 0.0,
                    "word_timestamps": False,
                    "verbose": False,
                })

            # Convert to float32 in [-1,1] for Whisper
            audio_float = audio.astype(np.float32) / 32768.0

            print(f"[Part 2] Transcribing with Whisper ...")
            asr_result = model.transcribe(audio_float, **kwargs)
            text = (asr_result.get("text") or "").strip()
            result["text"] = text

            if self.enable_commands:
                matched_cmd, english_cmd = match_action_command(text, self.command_threshold)
                result["matched_command"] = matched_cmd
                result["matched_command_english"] = english_cmd
                if matched_cmd:
                    print(f"[Command] Matched action: {matched_cmd} ({english_cmd})")
                    cmd_line = f"[{timestamp()}] COMMAND: {matched_cmd} ({english_cmd})  (raw: \"{text}\")"
                    write_to_text_file(cmd_line, self.output_file, append=True)
                else:
                    print(f"[Command] No matching action command found")
                    write_to_text_file(text, self.output_file, append=True)
            else:
                write_to_text_file(text, self.output_file, append=True)

            result["ok"] = True
        except Exception as exc:
            result["error"] = f"Part 2 (ASR) failed: {exc}"
            print(f"[voiceCommandRobot] {result['error']}",
                  file=sys.stderr)

        return result

    def run_loop(self, duration: float = 5.0,
                language: str = None,
                max_iterations: int = None) -> list:
        if language is None:
            language = self.language
        """Run in a continuous loop with pipelined record->transcribe.

        Recording and transcription overlap so there is no idle gap
        between commands: the next recording starts while the previous
        transcription is running.

        Returns a list of all result dicts.
        """
        import threading
        import queue

        results = []
        backend = self._resolve_backend()

        print()
        print("=" * 60)
        print("  voiceCommandRobot — Continuous Listening Mode")
        print("=" * 60)
        print(f"  Backend     : {backend}")
        print(f"  Device      : {self.device}")
        print(f"  Duration    : {duration}s per command")
        print(f"  Language    : {language or 'auto'}")
        print(f"  Optimize    : {self.optimize}")
        print(f"  Reduce Noise: {self.reduce_noise}")
        print(f"  Output file : {self.output_file}")
        print(f"  Model       : {self.model_name}")
        print("=" * 60)
        print("  Press Ctrl+C to stop.")
        print()

        # Pre-load Whisper model ONCE so every inference is instant
        print("[Loop] Pre-loading Whisper model ...")
        model = self._get_whisper_model(language)
        transcribe_kwargs = {"fp16": False}
        if language:
            transcribe_kwargs["language"] = language

        if self.optimize:
            transcribe_kwargs.update({
                "beam_size": 1,
                "best_of": 1,
                "condition_on_previous_text": False,
                "temperature": 0.0,
                "word_timestamps": False,
                "verbose": False,
            })

        # Record single noise sample if noise reduction is enabled
        noise_sample = None
        noise_sample_sr = None
        if self.reduce_noise:
            print(f"[Loop] Recording noise sample ({self.noise_duration}s) ...")
            noise_sample, noise_sample_sr, _ = self._record_audio(self.noise_duration)

        print("[Loop] Model ready. Starting pipeline.")
        print()

        # --- Shared queue between threads ---
        audio_queue = queue.Queue(maxsize=1)
        info_queue = queue.Queue(maxsize=1)
        error_queue = queue.Queue(maxsize=1)

        def record_one():
            """Record one audio segment and put in queue."""
            try:
                if backend == "sounddevice":
                    audio, actual_sr, info = record_from_microphone(
                        duration_seconds=duration,
                        device=self.device,
                    )
                elif backend == "alsa":
                    audio, actual_sr, info = record_with_arecord(
                        duration_seconds=duration,
                        device=self.device,
                    )
                elif backend == "pyaudio":
                    audio, actual_sr, info = record_with_pyaudio(
                        duration_seconds=duration,
                        device=self.device,
                    )
                else:
                    raise RuntimeError(f"Unknown backend: {backend}")
                audio_queue.put((audio, actual_sr))
                info_queue.put(info)
                error_queue.put(None)
            except Exception as e:
                audio_queue.put(None)
                info_queue.put(None)
                error_queue.put(e)

        # --- Kick off the first recording ---
        t = threading.Thread(target=record_one, daemon=True)
        t.start()

        count = 0
        try:
            while True:
                # Wait for the current recording to finish
                t.join()

                # Get recording from queue
                audio_tuple = audio_queue.get()
                info = info_queue.get()
                exc = error_queue.get()

                count += 1
                stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if exc:
                    print(f"[Loop-{count}] Record error: {exc}", file=sys.stderr)
                    results.append({"skill": "voiceCommandRobot", "ok": False,
                                   "iteration": count, "error": str(exc)})
                else:
                    audio, actual_sr = audio_tuple

                    # Apply noise reduction if enabled
                    if self.reduce_noise and noise_sample is not None:
                        print(f"--- [{count}] {stamp} | Reducing background noise ...")
                        audio = reduce_background_noise(audio, noise_sample, actual_sr)

                    print(f"--- [{count}] {stamp} | Transcribing ...")
                    try:
                        # Convert to float32 for Whisper
                        audio_float = audio.astype(np.float32) / 32768.0
                        asr_result = model.transcribe(audio_float, **transcribe_kwargs)
                        text = (asr_result.get("text") or "").strip()
                    except Exception as e:
                        text = ""
                        print(f"[Loop-{count}] ASR error: {e}", file=sys.stderr)

                    print()
                    print(f"  >> '{text}'")

                    matched_cmd = ""
                    english_cmd = ""
                    if self.enable_commands and text:
                        matched_cmd, english_cmd = match_action_command(text, self.command_threshold)
                        if matched_cmd:
                            print(f"  [COMMAND] {matched_cmd} ({english_cmd})")
                            cmd_line = f"[{stamp}] COMMAND: {matched_cmd} ({english_cmd})  (raw: \"{text}\")"
                            write_to_text_file(cmd_line, self.output_file, append=True)
                        else:
                            print(f"  [Command] No matching action command found")
                            write_to_text_file(text, self.output_file, append=True)
                    else:
                        write_to_text_file(text, self.output_file, append=True)
                    print()

                    results.append({
                        "skill": "voiceCommandRobot",
                        "ok": True,
                        "iteration": count,
                        "text": text,
                        "matched_command": matched_cmd if self.enable_commands else None,
                        "matched_command_english": english_cmd if self.enable_commands else None,
                        "device_id": self.device,
                        "backend": backend,
                        "output_file": self.output_file,
                        "error": None,
                    })

                if max_iterations and count >= max_iterations:
                    print(f"[Loop] Max iterations ({max_iterations}) reached.")
                    break

                # Immediately start the NEXT recording while we transcribe
                t = threading.Thread(target=record_one, daemon=True)
                t.start()
                print(f"  (recording next command in background...)")

        except KeyboardInterrupt:
            print(f"\n[Loop] Stopped by user.")

        print()
        print(f"[Loop] Total commands captured: {len(results)}")
        print(f"[Loop] Results saved to: {self.output_file}")
        print()
        return results


# ------------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------------

def run(args: argparse.Namespace = None) -> dict:
    """Agent-friendly entry. Returns a JSON-serializable dict."""
    if args is None:
        args = parse_args()

    if args.list_devices:
        backend = args.backend
        if backend == "auto":
            backend = detect_backend()
        print(f"Backend: {backend}")
        print()
        if backend == "sounddevice":
            try:
                import sounddevice as sd
                print("Available audio input devices (sounddevice):")
                for i, dev in enumerate(sd.query_devices()):
                    if int(dev.get("max_input_channels", 0)) > 0:
                        print(f"  [{i}] {dev.get('name', 'unknown')}  "
                              f"(inputs={dev.get('max_input_channels')}, "
                              f"sr={int(dev.get('default_samplerate', 0))})")
            except Exception as exc:
                print(f"Error listing sounddevice devices: {exc}")
                print("  Install: pip install sounddevice")
        elif backend == "pyaudio":
            try:
                devs = list_pyaudio_devices()
                print("Available audio input devices (pyaudio):")
                if devs:
                    for d in devs:
                        print(f"  [{d['index']}] {d['name']}  "
                              f"(inputs={d['channels']}, "
                              f"sr={d['sample_rate']})")
                else:
                    print("  (no capture devices found)")
            except Exception as exc:
                print(f"Error listing pyaudio devices: {exc}")
                print("  Install: pip install pyaudio")
        elif backend == "alsa":
            try:
                devs = list_alsa_devices()
                print("Available audio input devices (ALSA / arecord):")
                if devs:
                    for d in devs:
                        print(f"  [{d['index']}] {d['name']}  ({d['hw_addr']})")
                else:
                    print("  (no capture devices found)")
            except Exception as exc:
                print(f"Error listing ALSA devices: {exc}")
                print("  Install: sudo apt-get install alsa-utils")
        else:
            print("No audio backend available.")
            print("  Install sounddevice: pip install sounddevice  (recommended, cross-platform)")
            print("  Or install pyaudio: pip install pyaudio")
            print("  Or install ALSA: sudo apt-get install alsa-utils")
        return {"skill": "voiceCommandRobot", "ok": True,
                "devices_listed": True, "backend": backend}

    robot = voiceCommandRobot(model_name=args.model,
                              output_file=args.output,
                              device=args.device,
                              backend=args.backend,
                              optimize=args.optimize,
                              language=args.language,
                              reduce_noise=args.reduce_noise,
                              noise_duration=args.noise_duration,
                              enable_commands=args.enable_commands,
                              command_threshold=args.command_threshold)

    if args.no_ssl_verify:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        print("[Run] SSL verification disabled via --no-ssl-verify")

    if args.text is not None:
        # Dry-run: bypass the mic and Whisper, just write user-supplied text
        matched_cmd = ""
        english_cmd = ""
        if args.enable_commands:
            matched_cmd, english_cmd = match_action_command(args.text, args.command_threshold)
            if matched_cmd:
                print(f"[Command] Matched action: {matched_cmd} ({english_cmd})")
                cmd_line = f"[{timestamp()}] COMMAND: {matched_cmd} ({english_cmd})  (raw: \"{args.text}\")"
                write_to_text_file(cmd_line, robot.output_file, append=True)
            else:
                print(f"[Command] No matching action command found")
                write_to_text_file(args.text, robot.output_file, append=True)
        else:
            write_to_text_file(args.text, robot.output_file, append=True)
        return {
            "skill": "voiceCommandRobot",
            "ok": True,
            "text": args.text,
            "matched_command": matched_cmd if args.enable_commands else None,
            "matched_command_english": english_cmd if args.enable_commands else None,
            "audio_file": None,
            "output_file": robot.output_file,
            "error": None,
        }

    if args.loop:
        return robot.run_loop(duration=args.duration,
                            language=args.language,
                            max_iterations=args.max)

    return robot.run_once(duration=args.duration,
                        language=args.language,
                        audio_path=args.audio)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="voiceCommandRobot — mic -> audio array -> Whisper -> text file."
    )
    p.add_argument("--model", default="tiny",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (default: tiny).")
    p.add_argument("--duration", type=float, default=5.0,
                   help="Seconds to record from the mic (default: 5.0).")
    p.add_argument("--language", default="zh",
                   help="Language hint for Whisper (default: 'zh' for Chinese).")
    p.add_argument("--output", default="robot_transcripts.txt",
                   help="Text file that receives the transcription. "
                        "Appends by default.")
    p.add_argument("--audio", default=None,
                   help="Path to a pre-recorded WAV file; skips the mic.")
    p.add_argument("--text", default=None,
                   help="Dry run: skip mic & Whisper, just write this string.")
    p.add_argument("--device", type=int, default=None,
                   help="Microphone device index. Run --list-devices to see available IDs.")
    p.add_argument("--list-devices", action="store_true",
                   help="List all available audio input devices and exit.")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "sounddevice", "pyaudio", "alsa"],
                   help="Audio recording backend (default: auto — "
                        "tries sounddevice, pyaudio, then ALSA). "
                        "Use 'alsa' on RK3588 without PortAudio. "
                        "Use 'sounddevice' for cross-platform support (recommended).")
    p.add_argument("--no-ssl-verify", action="store_true",
                   help="Disable SSL certificate verification when "
                        "downloading Whisper model (use on RK3588 if "
                        "CERTIFICATE_VERIFY_FAILED occurs).")
    p.add_argument("--loop", action="store_true",
                   help="Run in continuous loop: listen -> transcribe -> save -> repeat. "
                        "Press Ctrl+C to stop.")
    p.add_argument("--max", type=int, default=None,
                   help="Maximum number of loop iterations (with --loop). "
                        "Default: unlimited.")
    p.add_argument("--optimize", action="store_true",
                   help="Optimize for speed (faster transcription, possibly slightly less accurate).")
    p.add_argument("--reduce-noise", action="store_true",
                   help="Enable background noise reduction (first records a short noise sample).")
    p.add_argument("--noise-duration", type=float, default=1.0,
                   help="Duration of noise sample (in seconds) for noise reduction (default: 1.0).")
    p.add_argument("--enable-commands", action="store_true",
                   help="Enable action command matching (前进/后退/左转/右转/停止/抓取/释放). "
                        "Matches transcribed text to the closest action command.")
    p.add_argument("--command-threshold", type=float, default=0.5,
                   help="Similarity threshold for command matching (default: 0.5). "
                        "Higher = stricter matching.")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()

    if _args.no_ssl_verify:
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass

    result = run(_args)

    if not _args.list_devices:
        print()
        print(json.dumps(result, ensure_ascii=False, indent=2))
