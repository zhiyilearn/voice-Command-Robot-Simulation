#!/usr/bin/env python3
"""
voiceCommandRobot
-------------------
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
  python3 voiceCommandRobot.py --chat           # voice chat mode (STT -> LLM -> TTS -> speak)
  python3 voiceCommandRobot.py --model tiny      # choose Whisper model size
  python3 voiceCommandRobot.py --duration 5      # record for N seconds
  python3 voiceCommandRobot.py --language zh     # hint language to Whisper
  python3 voiceCommandRobot.py --backend alsa   # force ALSA backend (RK3588)
  python3 voiceCommandRobot.py --list-devices   # show available microphones

  # Voice chat (RK3588 — all offline):
  python3 voiceCommandRobot.py --chat --backend alsa --device 1 --language zh
  python3 voiceCommandRobot.py --chat --llm qwen2-0.5b           # auto-download Qwen2 0.5B (zh+en)
  python3 voiceCommandRobot.py --chat --llm qwen2-1.5b           # Qwen2 1.5B (better quality)
  python3 voiceCommandRobot.py --chat --chat-model ./tinyllama.gguf
  python3 voiceCommandRobot.py --chat --piper /usr/bin/piper --piper-voice en_US-lessac-medium.onnx
  python3 voiceCommandRobot.py --list-llm                         # list all LLM presets

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

    Returns one of: "pyaudio", "sounddevice", "alsa" (arecord), or "none".
    Checks the preferred backend first; if not available, tries the others.
    """
    order = []
    if preferred:
        order.append(preferred)
    order += [b for b in ("pyaudio", "sounddevice", "alsa") if b not in order]

    for backend in order:
        if backend == "pyaudio":
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                count = p.get_device_count()
                p.terminate()
                if count > 0:
                    return "pyaudio"
            except Exception:
                continue
        elif backend == "sounddevice":
            try:
                import sounddevice as sd  # noqa: F401
                sd.query_devices()
                return "sounddevice"
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
            if int(info.get('maxInputChannels', 0)) > 0:
                devices.append({
                    "index": i,
                    "name": info.get('name', 'unknown'),
                    "sample_rate": int(info.get('defaultSampleRate', 16000)),
                    "channels": int(info.get('maxInputChannels', 1)),
                })
        p.terminate()
    except Exception:
        pass
    return devices


def record_with_pyaudio(wav_path: str,
                        duration_seconds: float = 5.0,
                        sample_rate: int = 16000,
                        channels: int = 1,
                        device: int = None) -> dict:
    """Record audio using pyaudio (cross-platform, from reference code).

    This is the backend used in the Version 10 reference code.
    Works on Linux, Windows, and macOS.

    Args:
        wav_path:         Output WAV file path.
        duration_seconds: Recording duration in seconds.
        sample_rate:      Target sample rate.
        channels:         Number of channels (default 1).
        device:           pyaudio device index. None = default.

    Returns a dict with recording info.
    """
    import pyaudio

    FORMAT = pyaudio.paInt16
    CHUNK = 512

    p = pyaudio.PyAudio()

    dev_index = device
    dev_name = "default"
    if dev_index is not None:
        try:
            info = p.get_device_info_by_index(dev_index)
            dev_name = info.get('name', 'unknown')
        except Exception:
            pass

    print(f"[Part 1] Recording with pyaudio from device {dev_index or 'default'} "
          f"\"{dev_name}\" for {duration_seconds}s "
          f"(sr={sample_rate}, ch={channels}) ...")

    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    input_device_index=dev_index,
                    frames_per_buffer=CHUNK)

    frames = []
    try:
        for _ in range(0, int(sample_rate / CHUNK * duration_seconds)):
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

    # Save to WAV
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    # Compute RMS
    raw = b''.join(frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    peak_rms = int(np.sqrt(np.mean(audio.astype("float64") ** 2)) or 0)

    info = {
        "file": wav_path,
        "device_id": dev_index,
        "device_name": dev_name,
        "backend": "pyaudio",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_rms": peak_rms,
        "size_bytes": os.path.getsize(wav_path),
    }
    print(f"[Part 1] Saved -> {wav_path}  ({info['size_bytes']} bytes, "
          f"peak_rms={peak_rms})")
    return info


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


def record_with_arecord(wav_path: str,
                        duration_seconds: float = 5.0,
                        sample_rate: int = 16000,
                        channels: int = 1,
                        device: str = None) -> dict:
    """Record audio using ALSA's arecord command (no PortAudio needed).

    This is the fallback backend for embedded systems like RK3588 where
    sounddevice / PortAudio is not available.

    Args:
        wav_path:         Output WAV file path.
        duration_seconds: Recording duration in seconds.
        sample_rate:      Target sample rate. Uses plughw for auto-resampling.
        channels:         Number of channels (default 1).
        device:           ALSA device string like "hw:1,0" or "plughw:1,0".
                          If None, uses default. Can also be a numeric index
                          into the list returned by list_alsa_devices().

    Returns a dict with recording info.
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

    cmd = [
        "arecord",
        "-D", dev_str,
        "-d", str(int(duration_seconds)),
        "-r", str(sample_rate),
        "-f", "S16_LE",
        "-c", str(channels),
        "-t", "wav",
        wav_path,
    ]

    print(f"[Part 1] Recording with arecord from device \"{dev_str}\" "
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

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("arecord produced no output file")

    # Read WAV to compute peak RMS
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch)[:, 0]
    peak_rms = int(np.sqrt(np.mean(audio.astype("float64") ** 2)) or 0)

    info = {
        "file": wav_path,
        "device_id": dev_str,
        "device_name": dev_name if 'dev_name' in dir() else dev_str,
        "backend": "alsa",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_rms": peak_rms,
        "size_bytes": os.path.getsize(wav_path),
    }
    print(f"[Part 1] Saved -> {wav_path}  ({info['size_bytes']} bytes, "
          f"peak_rms={peak_rms})")
    return info


def record_from_microphone(wav_path: str,
                           duration_seconds: float = 5.0,
                           sample_rate: int = 16000,
                           channels: int = 1,
                           device: int = None) -> dict:
    """Capture audio from the specified microphone and write it as WAV.

    Args:
        wav_path:         Output WAV file path.
        duration_seconds: Recording duration in seconds.
        sample_rate:      Target PCM sample rate (default 16000 — required by Whisper).
                          If the device doesn't support this, we'll record at
                          the closest supported rate and resample.
        channels:         Number of audio channels (default 1 = mono).
        device:           Sounddevice device index. None = default microphone.
                          Use --list-devices to see available device IDs.

    Returns a dict describing the recording (file path, duration, RMS, etc.).
    This is Part 1 of the pipeline.
    """
    import sounddevice as sd
    import soundfile as sf

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
        actual_sr = int(dev_info.get("default_samplerate", 44100))
        try:
            sd.check_input_settings(device=device, samplerate=actual_sr,
                                    channels=channels)
        except Exception as exc:
            raise RuntimeError(
                f"Device {dev_id} does not support {actual_sr} Hz either. "
                f"Available sample rates: {sd.query_devices(device=device).get('sample_rates')}"
            ) from exc

    print(f"[Part 1] Recording from device {dev_id} \"{dev_name}\" "
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

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    info = {
        "file": wav_path,
        "device_id": dev_id,
        "device_name": dev_name,
        "sample_rate": sample_rate,
        "actual_sample_rate": actual_sr,
        "resampled": actual_sr != sample_rate,
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_rms": peak_rms,
        "size_bytes": os.path.getsize(wav_path),
    }
    print(f"[Part 1] Saved -> {wav_path}  ({info['size_bytes']} bytes, "
          f"peak_rms={peak_rms})")
    return info


# ------------------------------------------------------------------
# Part 2 — Offline Whisper ASR + write to text file
# ------------------------------------------------------------------

def transcribe_with_whisper(wav_path: str,
                            model_name: str = "tiny",
                            language: str = None) -> str:
    """Run off-line Whisper ASR on a WAV file. Returns the transcription text.

    This is Part 2 of the pipeline.
    """
    import whisper

    print(f"[Part 2] Loading Whisper model '{model_name}' ...")
    try:
        model = whisper.load_model(model_name)
    except Exception as exc:
        # Some embedded systems have stale OpenSSL CA bundles. Retry once
        # with SSL verification disabled — safe for trusted model downloads.
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSL" in str(exc):
            print("[Part 2] SSL error detected. Retrying with "
                  "SSL verification disabled (use --no-ssl-verify to skip).")
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            model = whisper.load_model(model_name)
        else:
            raise

    kwargs = {"fp16": False}
    if language:
        kwargs["language"] = language

    print(f"[Part 2] Transcribing '{wav_path}' with Whisper ...")
    result = model.transcribe(wav_path, **kwargs)
    text = (result.get("text") or "").strip()
    print(f"[Part 2] Transcription result: \"{text}\"")
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
    """Tiny skill: mic -> WAV -> Whisper -> text file."""

    def __init__(self,
                 model_name: str = "tiny",
                 output_file: str = "robot_transcripts.txt",
                 device=None,
                 backend: str = "auto"):
        self.model_name = model_name
        self.output_file = os.path.abspath(output_file)
        self.device = device
        self.backend = backend
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
                    "install ALSA (apt-get install alsa-utils)."
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

    def run_once(self, duration: float = 5.0,
                 language: str = None,
                 audio_path: str = None) -> dict:
        """Execute the two-part pipeline. Returns a JSON-friendly result."""
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
                wav_path = audio_path
                info = {"file": audio_path, "mode": "pre-recorded"}
                print(f"[Part 1] Using pre-recorded audio: {audio_path}")
            else:
                backend = self._resolve_backend()
                result["backend"] = backend
                wav_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ).name
                if backend == "sounddevice":
                    info = record_from_microphone(
                        wav_path,
                        duration_seconds=duration,
                        device=self.device,
                    )
                elif backend == "alsa":
                    info = record_with_arecord(
                        wav_path,
                        duration_seconds=duration,
                        device=self.device,
                    )
                elif backend == "pyaudio":
                    info = record_with_pyaudio(
                        wav_path,
                        duration_seconds=duration,
                        device=self.device,
                    )
                else:
                    raise RuntimeError(f"Unknown backend: {backend}")
            result["audio_file"] = info["file"]
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
            print(f"[Part 2] Transcribing '{wav_path}' with Whisper ...")
            asr_result = model.transcribe(wav_path, **kwargs)
            text = (asr_result.get("text") or "").strip()
            result["text"] = text
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
        """Run in a continuous loop with pipelined record->transcribe.

        Recording and transcription overlap so there is no idle gap
        between commands: the next recording starts while the previous
        transcription is running.

        Returns a list of all result dicts.
        """
        import threading
        results = []
        backend = self._resolve_backend()

        print()
        print("=" * 60)
        print("  voiceCommandRobot — Continuous Listening Mode")
        print("=" * 60)
        print(f"  Backend   : {backend}")
        print(f"  Device    : {self.device}")
        print(f"  Duration  : {duration}s per command")
        print(f"  Language  : {language or 'auto'}")
        print(f"  Output    : {self.output_file}")
        print(f"  Model     : {self.model_name}")
        print("=" * 60)
        print("  Press Ctrl+C to stop.")
        print()

        # Pre-load Whisper model ONCE so every inference is instant
        print("[Loop] Pre-loading Whisper model ...")
        model = self._get_whisper_model(language)
        transcribe_kwargs = {"fp16": False}
        if language:
            transcribe_kwargs["language"] = language
        print("[Loop] Model ready. Starting pipeline.")
        print()

        # --- Shared mutable state between threads ---
        wav_path_ref = [None]      # path of the most recent recording
        record_done_ref = [False]  # True when recording thread has finished
        record_error_ref = [None]  # exception from recording, or None
        record_lock = threading.Lock()

        def record_one():
            """Record one audio segment into a temp file."""
            path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            exc = None
            try:
                if backend == "sounddevice":
                    record_from_microphone(path, duration_seconds=duration,
                                          device=self.device)
                elif backend == "alsa":
                    record_with_arecord(path, duration_seconds=duration,
                                        device=self.device)
                elif backend == "pyaudio":
                    record_with_pyaudio(path, duration_seconds=duration,
                                       device=self.device)
            except Exception as e:
                exc = e
            with record_lock:
                wav_path_ref[0] = path
                record_error_ref[0] = exc
                record_done_ref[0] = True

        # --- Kick off the first recording ---
        t = threading.Thread(target=record_one, daemon=True)
        t.start()

        count = 0
        pending_path = None    # path captured before we start transcribing
        pending_exc = None    # record error from that same path

        try:
            while True:
                # Wait for the current recording to finish
                t.join()
                count += 1
                stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Grab the finished recording
                with record_lock:
                    pending_path = wav_path_ref[0]
                    pending_exc = record_error_ref[0]

                if pending_exc:
                    print(f"[Loop-{count}] Record error: {pending_exc}", file=sys.stderr)
                    results.append({"skill": "voiceCommandRobot", "ok": False,
                                   "iteration": count, "error": str(pending_exc)})
                else:
                    print(f"--- [{count}] {stamp} | Transcribing ...")
                    try:
                        asr_result = model.transcribe(pending_path, **transcribe_kwargs)
                        text = (asr_result.get("text") or "").strip()
                    except Exception as exc:
                        text = ""
                        print(f"[Loop-{count}] ASR error: {exc}", file=sys.stderr)

                    print()
                    print(f"  >> \"{text}\"")
                    print()
                    write_to_text_file(text, self.output_file, append=True)

                    results.append({
                        "skill": "voiceCommandRobot",
                        "ok": True,
                        "iteration": count,
                        "text": text,
                        "audio_file": pending_path,
                        "device_id": self.device,
                        "backend": backend,
                        "output_file": self.output_file,
                        "error": None,
                    })

                if max_iterations and count >= max_iterations:
                    print(f"[Loop] Max iterations ({max_iterations}) reached.")
                    break

                # Immediately start the NEXT recording while we transcribe
                with record_lock:
                    wav_path_ref[0] = None
                    record_done_ref[0] = False
                    record_error_ref[0] = None
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
# Part 2b — Voice Activity Detection (webrtcvad)
# ------------------------------------------------------------------

CHUNK_DURATION_MS = 30  # 30ms chunks (required for VAD)
VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = int(VAD_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


def init_vad(aggressiveness: int = 3):
    """Initialize webrtcvad voice activity detector.

    Args:
        aggressiveness: 0-3 (3 = most strict, filters more noise).

    Returns a Vad object, or None if webrtcvad not available.
    """
    try:
        import webrtcvad
        return webrtcvad.Vad(aggressiveness)
    except ImportError:
        return None


def is_human_voice(audio_bytes: bytes, sample_rate: int = 16000,
                   vad=None) -> bool:
    """Check if raw 16-bit PCM audio contains human voice.

    Args:
        audio_bytes: Raw 16-bit mono PCM audio data.
        sample_rate: Must be 8000, 16000, 32000, or 48000 (webrtcvad requirement).
        vad: Pre-initialized webrtcvad.Vad object. Created if None.

    Returns True if voice detected, False otherwise.
    """
    if vad is None:
        vad = init_vad()
    if vad is None:
        return False  # VAD not available

    if sample_rate not in (8000, 16000, 32000, 48000):
        return False

    # webrtcvad requires exactly 10, 20, or 30ms chunks
    chunk_size = int(sample_rate * CHUNK_DURATION_MS / 1000)
    bytes_per_sample = 2  # 16-bit
    chunk_bytes = chunk_size * bytes_per_sample

    if len(audio_bytes) >= chunk_bytes:
        chunk = audio_bytes[:chunk_bytes]
    else:
        # Pad with silence if too short
        chunk = audio_bytes + b'\x00' * (chunk_bytes - len(audio_bytes))

    try:
        return vad.is_speech(chunk, sample_rate)
    except Exception:
        return False


# ------------------------------------------------------------------
# Part 3 — Offline Chat LLM (llama.cpp / llama-cpp-python)
# Part 4 — Offline TTS (Piper / espeak)
# Part 5 — Speaker output (aplay)
# ------------------------------------------------------------------

# Recommended open-source offline GGUF models for RK3588 edge devices.
# Sorted by size (smallest first). All are bilingual (zh/en) or English-only.
# Q4_K_M quant is the sweet spot for RK3588 (quality vs speed vs RAM).
LLM_MODEL_PRESETS = {
    "qwen2-0.5b": {
        "name": "Qwen2-0.5B-Instruct",
        "repo": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "file": "qwen2-0_5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size_mb": 380,
        "ram_mb": 600,
        "langs": "zh+en",
        "description": "Qwen2 0.5B — smallest, fastest, good Chinese/English. ~0.6GB RAM.",
    },
    "qwen2-1.5b": {
        "name": "Qwen2-1.5B-Instruct",
        "repo": "Qwen/Qwen2-1.5B-Instruct-GGUF",
        "file": "qwen2-1_5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size_mb": 970,
        "ram_mb": 1400,
        "langs": "zh+en",
        "description": "Qwen2 1.5B — great quality/speed balance for RK3588. ~1.4GB RAM.",
    },
    "tinyllama": {
        "name": "TinyLlama-1.1B-Chat",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "model_type": "chatml",
        "size_mb": 700,
        "ram_mb": 1000,
        "langs": "en",
        "description": "TinyLlama 1.1B — fast, English-only. ~1GB RAM.",
    },
    "qwen2-7b": {
        "name": "Qwen2-7B-Instruct",
        "repo": "Qwen/Qwen2-7B-Instruct-GGUF",
        "file": "qwen2-7b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size_mb": 4700,
        "ram_mb": 6000,
        "langs": "zh+en",
        "description": "Qwen2 7B — best quality, needs 8GB+ RAM. ~4.7GB file.",
    },
}


def get_llm_preset(preset_name: str) -> dict:
    """Get model preset info by name. Case-insensitive. Returns None if not found."""
    return LLM_MODEL_PRESETS.get(preset_name.lower())


def list_llm_presets() -> str:
    """Return a formatted string listing all available model presets."""
    lines = ["Available LLM presets for RK3588 (all Q4_K_M GGUF):"]
    lines.append("-" * 70)
    for key, info in LLM_MODEL_PRESETS.items():
        lines.append(f"  {key:15s}  {info['size_mb']:>5d}MB  {info['langs']:>6s}  {info['description']}")
    lines.append("-" * 70)
    lines.append("  Use --llm qwen2-0.5b  (auto-downloads to ~/.cache/llm/)")
    lines.append("  Use --list-llm         to show this list")
    return "\n".join(lines)


def download_file(url: str, dest_path: str, show_progress: bool = True) -> bool:
    """Download a file from URL to dest_path. Returns True on success."""
    import urllib.request

    print(f"[Download] From: {url}")
    print(f"[Download] To:   {dest_path}")

    # SSL fallback for RK3588
    try:
        opener = urllib.request.build_opener()
    except Exception:
        pass

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "voiceCommandRobot/1.0"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"\r[Download] {mb:.1f}/{mb_total:.1f} MB ({pct:.1f}%)",
                              end="", flush=True)
        if show_progress and total:
            print()
        print(f"[Download] Done ({os.path.getsize(dest_path)} bytes)")
        return True
    except Exception as exc:
        # Retry with SSL disabled
        if "SSL" in str(exc) or "CERTIFICATE" in str(exc):
            print(f"[Download] SSL error, retrying without verification ...")
            import ssl
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "voiceCommandRobot/1.0"})
                with urllib.request.urlopen(req) as response:
                    total = int(response.headers.get("Content-Length", 0))
                    downloaded = 0
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with open(dest_path, "wb") as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if show_progress and total:
                                pct = downloaded / total * 100
                                mb = downloaded / (1024 * 1024)
                                mb_total = total / (1024 * 1024)
                                print(f"\r[Download] {mb:.1f}/{mb_total:.1f} MB ({pct:.1f}%)",
                                      end="", flush=True)
                if show_progress and total:
                    print()
                print(f"[Download] Done ({os.path.getsize(dest_path)} bytes)")
                return True
            except Exception as exc2:
                print(f"[Download] Failed (even without SSL): {exc2}")
                if os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                return False
        else:
            print(f"[Download] Failed: {exc}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
            return False


def resolve_chat_model(model_path: str = None,
                       llm_preset: str = None,
                       models_dir: str = None) -> tuple:
    """Resolve the chat model path. Returns (model_path, model_type).

    Priority:
      1. Explicit --chat-model path
      2. --llm preset (auto-download if missing)
      3. Auto-find in common directories
      4. Default preset (qwen2-0.5b, auto-download)
    """
    if models_dir is None:
        models_dir = os.path.expanduser("~/.cache/llm")

    # 1. Explicit path
    if model_path and os.path.exists(model_path):
        return model_path, "auto"

    # 2. LLM preset
    if llm_preset:
        preset = get_llm_preset(llm_preset)
        if preset is None:
            raise ValueError(
                f"Unknown LLM preset '{llm_preset}'. "
                f"Available: {', '.join(LLM_MODEL_PRESETS.keys())}. "
                f"Use --list-llm to see details."
            )
        dest = os.path.join(models_dir, preset["file"])
        if not os.path.exists(dest):
            print(f"[Chat] Model '{preset['name']}' not found locally.")
            print(f"[Chat]   Size : ~{preset['size_mb']} MB")
            print(f"[Chat]   RAM  : ~{preset['ram_mb']} MB")
            print(f"[Chat]   Lang : {preset['langs']}")
            print(f"[Chat] Downloading (one-time) ...")
            ok = download_file(preset["url"], dest)
            if not ok:
                raise RuntimeError(f"Failed to download model '{llm_preset}'")
        return dest, preset["model_type"]

    # 3. Auto-find
    if model_path is None:
        search = [
            models_dir,
            os.path.expanduser("~/.cache/llama.cpp/"),
            os.path.expanduser("~/.cache/"),
            os.path.expanduser("~/models/"),
            "./models/",
            "./",
        ]
        for d in search:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".gguf"):
                        found = os.path.join(d, f)
                        print(f"[Chat] Auto-found model: {found}")
                        return found, "auto"

    # 4. Default: download qwen2-0.5b
    print("[Chat] No model specified and none found locally.")
    print("[Chat] Using default preset: qwen2-0.5b (Qwen2 0.5B Instruct, zh+en)")
    preset = get_llm_preset("qwen2-0.5b")
    dest = os.path.join(models_dir, preset["file"])
    if not os.path.exists(dest):
        ok = download_file(preset["url"], dest)
        if not ok:
            raise RuntimeError("Failed to download default model (qwen2-0.5b)")
    return dest, preset["model_type"]


def find_llama_cli() -> str:
    """Find llama.cpp command-line executable (llama-cli / main / llama-server).

    Returns the path or None.
    """
    candidates = [
        "llama-cli",
        "main",
        "llama-main",
        os.path.expanduser("~/llama.cpp/main"),
        os.path.expanduser("~/llama.cpp/llama-cli"),
        "./llama.cpp/main",
        "./llama.cpp/llama-cli",
    ]
    for c in candidates:
        path = shutil.which(c)
        if path:
            return path
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def init_chat_model_cli(model_path: str,
                        model_type: str = "auto",
                        n_ctx: int = 2048,
                        n_threads: int = None) -> dict:
    """Initialize chat using llama.cpp command-line (no Python library needed).

    This is a fallback when llama-cpp-python is not installed.
    Uses `llama-cli` or `main` from llama.cpp.

    Returns {"backend": "cli", "model_path": ..., "model_type": ..., "llama_cli": ...}
    """
    import multiprocessing

    if n_threads is None:
        n_threads = max(1, multiprocessing.cpu_count() - 1)

    llama_cli = find_llama_cli()
    if llama_cli is None:
        raise FileNotFoundError(
            "llama.cpp command-line tool not found. "
            "Install either: \n"
            "  1. pip install llama-cpp-python (Python library)\n"
            "  2. Or build llama.cpp from source and put 'llama-cli' or 'main' in PATH\n"
            "     https://github.com/ggerganov/llama.cpp"
        )

    print(f"[Chat] Using llama-cli: {llama_cli}")
    print(f"[Chat] Model: {model_path}")
    print(f"[Chat] n_ctx={n_ctx}, n_threads={n_threads}")
    print(f"[Chat] LLM backend: llama-cli (command-line)")

    return {
        "backend": "cli",
        "model_path": model_path,
        "model_type": model_type,
        "llama_cli": llama_cli,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
    }


def chat_with_cli(llm_info: dict, user_message: str,
                  system_prompt: str = None,
                  max_tokens: int = 256,
                  temperature: float = 0.7,
                  verbose: bool = False) -> str:
    """Chat using llama.cpp command-line (llama-cli / main).

    Builds the prompt manually and parses the output.
    """
    import tempfile as _tmp

    model_path = llm_info["model_path"]
    llama_cli = llm_info["llama_cli"]
    n_ctx = llm_info.get("n_ctx", 2048)
    n_threads = llm_info.get("n_threads", 4)
    model_type = llm_info.get("model_type", "auto")

    # Build prompt based on model type
    if model_type == "chatml" or model_type == "tinyllama":
        # ChatML format
        prompt = ""
        if system_prompt:
            prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        stop = ["<|im_end|>"]
    elif model_type == "qwen2" or model_type == "qwen":
        # Qwen format
        prompt = ""
        if system_prompt:
            prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        stop = ["<|im_end|>", "<|endoftext|>"]
    else:
        # Generic instruction format
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        prompt += f"User: {user_message}\nAssistant:"
        stop = ["User:", "\nUser:"]

    if verbose:
        print(f"  [Chat] CLI prompt:\n{prompt[:200]}...")

    # Write prompt to temp file (avoids shell escaping issues)
    prompt_file = _tmp.mktemp(suffix="_prompt.txt")
    try:
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        # Build llama-cli command
        cmd = [
            llama_cli,
            "-m", model_path,
            "-c", str(n_ctx),
            "-t", str(n_threads),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-f", prompt_file,
            "--color", "0",
            "--no-penalize-nl",
        ]

        # Add stop tokens
        for s in stop:
            cmd += ["--stop", s]

        if verbose:
            print(f"  [Chat] CLI cmd: {' '.join(cmd[:6])} ...")

        # Run
        result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120, env=os.environ.copy())

        if result.returncode != 0 and verbose:
            print(f"  [Chat] CLI stderr: {result.stderr[:300]}")

        # Parse output — llama-cli outputs the prompt + completion
        output = result.stdout

        # Remove the prompt from the beginning if present
        if output.startswith(prompt):
            output = output[len(prompt):]
        elif prompt[-50:] in output:
            idx = output.find(prompt[-50:])
            if idx >= 0:
                output = output[idx + 50:]

        # Clean up
        reply = output.strip()

        # Truncate at stop tokens
        for s in stop:
            if s in reply:
                reply = reply[:reply.index(s)].strip()

        if verbose:
            print(f"  [Chat] CLI reply: '{reply[:200]}...'")

        return reply
    except Exception as exc:
        if verbose:
            print(f"  [Chat] CLI error: {exc}")
        return ""
    finally:
        try:
            os.remove(prompt_file)
        except Exception:
            pass


def init_chat_model(model_path: str = None,
                    model_type: str = "auto",
                    llm_preset: str = None,
                    n_ctx: int = 2048,
                    n_threads: int = None,
                    n_gpu_layers: int = 0) -> dict:
    """Load an offline chat LLM.

    Tries backends in order:
      1. llama-cpp-python (fast, Python library)
      2. llama.cpp command-line (no Python dependency needed)

    Supports any GGUF model (TinyLlama, Qwen, Phi, etc.).
    Use llm_preset for easy model selection on RK3588.

    Args:
        model_path: Path to a GGUF model file. If None, tries auto-find.
        model_type: Model architecture hint — "llama", "qwen2", "phi2", "auto", "chatml".
        llm_preset: Named preset (e.g. "qwen2-0.5b", "qwen2-1.5b", "tinyllama").
                    Auto-downloads if not cached.
        n_ctx:      Context window size.
        n_threads:  CPU threads. None = auto-detect.
        n_gpu_layers: Layers offloaded to GPU (RK3588 NPU not supported
                      via this path; set 0 for CPU-only).

    Returns {"model": ..., "model_path": str, "model_type": str, "backend": str}.
    Raises ImportError if no LLM backend is available.
    """
    import multiprocessing

    if n_threads is None:
        n_threads = max(1, multiprocessing.cpu_count() - 1)

    # Resolve model path (may trigger download)
    resolved_path, resolved_type = resolve_chat_model(
        model_path=model_path,
        llm_preset=llm_preset,
    )
    if model_type == "auto":
        model_type = resolved_type

    print(f"[Chat] Loading LLM from: {resolved_path}")
    print(f"[Chat] Model type: {model_type}")
    print(f"[Chat] n_ctx={n_ctx}, n_threads={n_threads}, n_gpu_layers={n_gpu_layers}")

    # Try llama-cpp-python first
    try:
        from llama_cpp import Llama
        llm_kwargs = {
            "model_path": resolved_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
        }
        if model_type != "auto":
            llm_kwargs["chat_format"] = model_type

        llm = Llama(**llm_kwargs)
        print(f"[Chat] LLM loaded successfully (llama-cpp-python backend).")
        return {"model": llm, "model_path": resolved_path,
                "model_type": model_type, "backend": "python"}
    except ImportError:
        print(f"[Chat] llama-cpp-python not installed.")
    except Exception as exc:
        print(f"[Chat] llama-cpp-python failed: {exc}")

    # Fallback: llama.cpp command-line
    try:
        return init_chat_model_cli(resolved_path, model_type,
                                   n_ctx=n_ctx, n_threads=n_threads)
    except FileNotFoundError as exc:
        print(f"[Chat] llama-cli not found either.")
        raise ImportError(
            "No LLM backend available. Install one of:\n"
            "\n"
            "  Option 1 — Python library (recommended):\n"
            "    pip install llama-cpp-python\n"
            "\n"
            "    On RK3588 ARM64, build from source:\n"
            "    pip install llama-cpp-python --no-cache-dir \\\n"
            "      --extra-cmake-args=\"-DLLAMA_BLAS=off -DLLAMA_CUBLAS=off\"\n"
            "\n"
            "  Option 2 — llama.cpp command-line:\n"
            "    git clone https://github.com/ggerganov/llama.cpp\n"
            "    cd llama.cpp && make\n"
            "    # Then put llama-cli/main in PATH\n"
        ) from exc


def chat_with_model(llm_info, user_message: str,
                    system_prompt: str = None,
                    max_tokens: int = 256,
                    temperature: float = 0.7,
                    verbose: bool = False) -> str:
    """Send a message to the LLM and return the assistant's reply.

    Supports multiple backends:
      - "python": llama-cpp-python library (fast)
      - "cli": llama.cpp command-line (no Python dependency)

    Args:
        llm_info:       Dict from init_chat_model() with "model", "backend", etc.
        user_message:   The user's input text.
        system_prompt:  Optional system prompt.
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature.
        verbose:        Print debug info (response structure, errors, etc.).

    Returns the assistant's reply string (empty string on failure).
    """
    # Handle both old-style (just the model object) and new-style (dict)
    if isinstance(llm_info, dict):
        backend = llm_info.get("backend", "python")
        model = llm_info.get("model")
        if backend == "cli":
            return chat_with_cli(llm_info, user_message,
                                 system_prompt=system_prompt,
                                 max_tokens=max_tokens,
                                 temperature=temperature,
                                 verbose=verbose)
    else:
        model = llm_info
        backend = "python"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    reply = ""

    # Try chat completion first
    try:
        if verbose:
            print(f"  [Chat] Calling create_chat_completion ...")
            print(f"  [Chat] Messages: {len(messages)} messages")
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>", "</s>"],
        )
        if verbose:
            import json
            print(f"  [Chat] Response keys: {list(response.keys())}")
            print(f"  [Chat] Full response: {json.dumps(response, ensure_ascii=False, indent=2)[:500]}")

        # Try multiple possible response structures
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = (choice["message"]["content"] or "").strip()
            elif "text" in choice:
                reply = (choice["text"] or "").strip()
            elif "delta" in choice and "content" in choice["delta"]:
                reply = (choice["delta"]["content"] or "").strip()
        if not reply and verbose:
            print(f"  [Chat] Empty reply from chat completion")

    except Exception as exc:
        if verbose:
            print(f"  [Chat] Chat completion failed: {exc}")
            import traceback
            traceback.print_exc()

    # Fallback: raw completion
    if not reply:
        if verbose:
            print(f"  [Chat] Trying raw completion ...")
        try:
            prompt = user_message
            if system_prompt:
                prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>", "</s>", "User:"],
            )
            if verbose:
                import json
                print(f"  [Chat] Raw response: {json.dumps(response, ensure_ascii=False, indent=2)[:500]}")
            if "choices" in response and len(response["choices"]) > 0:
                reply = (response["choices"][0].get("text", "") or "").strip()
        except Exception as exc:
            if verbose:
                print(f"  [Chat] Raw completion failed: {exc}")
                import traceback
                traceback.print_exc()

    if verbose:
        print(f"  [Chat] Final reply: '{reply}'")
        if not reply:
            print(f"  [Chat] WARNING: Empty reply!")

    return reply


def find_piper_tts() -> str:
    """Find or suggest a Piper TTS executable on the system."""
    candidates = [
        "/usr/local/bin/piper",
        "/usr/bin/piper",
        os.path.expanduser("~/piper/piper"),
        "./piper",
        "piper",
    ]
    for p in candidates:
        if shutil.which(p) or os.path.exists(p):
            return p
    return None


def find_espeak() -> str:
    """Find espeak-ng TTS executable."""
    for name in ["espeak-ng", "espeak"]:
        path = shutil.which(name)
        if path:
            return name
    return None


def list_alsa_playback_devices() -> list:
    """List ALSA playback devices using aplay.

    Returns a list of dicts: [{index, name, hw_addr, card, device}, ...]
    """
    devices = []
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("card ") and ": " in line and ", device " in line:
                card_part = line.strip().split(", device ")[0]
                rest = line.strip().split(", device ", 1)[1]
                card_num = card_part.split()[1].rstrip(":")
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
        print(f"[ALSA] Warning: could not list playback devices: {exc}")
    return devices


def get_default_alsa_device(device_type: str = "playback") -> str:
    """Auto-detect the working ALSA device.

    Tries common device names in order of likelihood for RK3588.
    Returns the device string (e.g. "plughw:0,0") or "default".
    """
    # Check environment variable first
    env_dev = os.environ.get("ALSA_PCM_NAME") or os.environ.get("AUDIO_DEVICE")
    if env_dev:
        return env_dev

    # Try to detect from aplay -l
    devs = []
    if device_type == "playback":
        devs = list_alsa_playback_devices()
    else:
        devs = list_alsa_devices()

    if devs:
        # Use plughw for auto-resampling
        first = devs[0]["hw_addr"].replace("hw:", "")
        return f"plughw:{first}"

    # Common RK3588 device names to try
    candidates = [
        "plughw:0,0",
        "plughw:1,0",
        "plughw:2,0",
        "hw:0,0",
        "default",
    ]
    for dev in candidates:
        try:
            result = subprocess.run(
                ["aplay", "-D", dev, "--dump-pcm", "-q", "/dev/zero"],
                capture_output=True, timeout=2
            )
            # If it doesn't fail immediately, it might work
            if result.returncode == 0 or "No such file" not in result.stderr:
                return dev
        except Exception:
            continue
    return "default"


def setup_alsa_env(playback_device: str = None,
                   capture_device: str = None) -> dict:
    """Set up ALSA environment variables to avoid config errors.

    On RK3588 and other embedded systems, the default ALSA config
    references non-existent devices (dmix, dsnoop, etc.) causing
    lots of warning messages. This function sets environment variables
    to point to the actual hardware device.

    Returns dict of environment variables set.
    """
    env = {}

    if playback_device is None:
        playback_device = get_default_alsa_device("playback")
    if capture_device is None:
        capture_device = get_default_alsa_device("capture")

    # Suppress ALSA warnings by pointing to real devices
    env["ALSA_CARD"] = "0"
    env["AUDIODEV"] = playback_device

    # Try to create a minimal ~/.asoundrc if it doesn't exist
    asoundrc = os.path.expanduser("~/.asoundrc")
    if not os.path.exists(asoundrc):
        try:
            pcm_dev = playback_device or "default"
            with open(asoundrc, "w") as f:
                f.write(f"""# Auto-generated by voiceCommandRobot
pcm.!default {{
    type plug
    slave {{
        pcm "{pcm_dev}"
        rate 44100
    }}
}}

ctl.!default {{
    type hw
    card 0
}}
""")
            print(f"[ALSA] Created minimal ~/.asoundrc with device: {pcm_dev}")
        except Exception as exc:
            print(f"[ALSA] Could not write ~/.asoundrc: {exc}")

    return env


def speak_espeak_direct(text: str, lang: str = "en",
                        device: str = None) -> bool:
    """Speak text directly using espeak-ng with proper ALSA device.

    Args:
        text:   Text to speak.
        lang:   Language code (en, zh, etc.).
        device: ALSA device for output. Auto-detected if None.

    Returns True if successful.
    """
    espeak = find_espeak()
    if not espeak:
        return False

    if device is None:
        device = get_default_alsa_device("playback")

    # Set up clean ALSA environment
    env = os.environ.copy()
    alsa_env = setup_alsa_env(playback_device=device)
    env.update(alsa_env)

    # Redirect stderr to suppress ALSA warnings
    try:
        cmd = [espeak, "-v", lang, "-a", "200", text]
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def synthesize_with_piper(text: str, output_wav: str,
                          piper_path: str = None,
                          voice_model: str = None) -> bool:
    """Convert text to speech using Piper (ONNX-based, fast, high quality).

    Piper is the best choice for RK3588 — pre-built ARM64 binaries available.
    Download: https://github.com/rhasspy/piper/releases
    Example voice models (English):
      en_US-lessac-medium.onnx + en_US-lessac-medium.onnx.json
    Example voice models (Chinese — try):
      zh_CN-huayan-medium.onnx + zh_CN-huayan-medium.onnx.json
    """
    if piper_path is None:
        piper_path = find_piper_tts()

    if piper_path is None or not shutil.which(piper_path):
        return False

    cmd = [piper_path]
    if voice_model:
        cmd += ["-m", voice_model]
    cmd += ["-o", output_wav]

    try:
        proc = subprocess.run(
            cmd, input=text.encode("utf-8"),
            capture_output=True, timeout=30,
        )
        return proc.returncode == 0 and os.path.exists(output_wav)
    except Exception:
        return False


def synthesize_with_espeak(text: str, output_wav: str,
                           voice: str = "en") -> bool:
    """Convert text to speech using espeak-ng (always available on Linux).

    Fallback if Piper is not installed.
    """
    espeak = find_espeak()
    if espeak is None:
        return False

    # espeak-ng writes WAV to stdout, pipe to file
    cmd = [espeak, "-w", output_wav, text]
    try:
        result = subprocess.run(cmd, capture_output=True,
                              text=True, timeout=15)
        return result.returncode == 0 and os.path.exists(output_wav)
    except Exception:
        return False


def synthesize_with_pyttsx3(text: str, output_wav: str = None,
                            lang: str = "en", rate: int = 200,
                            volume: float = 1.0,
                            play_direct: bool = False) -> bool:
    """Convert text to speech using pyttsx3 (cross-platform Python TTS).

    pyttsx3 uses system TTS engines:
      - Linux: espeak-ng (via pyttsx3 driver)
      - Windows: SAPI5
      - macOS: NSSpeechSynthesizer

    Args:
        text:        Text to synthesize.
        output_wav:  Path to save WAV file. If None and play_direct=False,
                     a temp file is used.
        lang:        Language hint: 'en', 'zh', etc.
        rate:        Speech rate (words per minute). Default 200.
        volume:      Volume 0.0 - 1.0. Default 1.0.
        play_direct: If True, speak directly (don't save to WAV).
                     Uses pyttsx3's built-in audio output.

    Returns True on success.
    """
    try:
        import pyttsx3
    except ImportError:
        return False

    try:
        engine = pyttsx3.init(driverName='espeak')
    except Exception:
        try:
            engine = pyttsx3.init()
        except Exception:
            return False

    try:
        # Try to set Chinese voice if lang is zh
        if lang and lang.startswith("zh"):
            voices = engine.getProperty('voices')
            for v in voices:
                v_langs = v.languages if hasattr(v, 'languages') else []
                v_id = v.id if hasattr(v, 'id') else ''
                # Look for Chinese voice
                if any('zh' in str(l).lower() or 'cmn' in str(l).lower()
                       for l in v_langs) or 'zh' in v_id.lower() or 'cmn' in v_id.lower():
                    engine.setProperty('voice', v_id)
                    break

        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)

        if play_direct:
            # Speak directly through speakers
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True
        else:
            # Save to WAV file
            engine.save_to_file(text, output_wav)
            engine.runAndWait()
            engine.stop()
            return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
    except Exception:
        return False
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def generate_test_tone(output_wav: str,
                       frequency: int = 440,
                       duration: float = 2.0,
                       sample_rate: int = 16000) -> bool:
    """Generate a simple sine wave test tone WAV file.

    Useful for testing speaker output without needing TTS tools.
    """
    try:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Convert to 16-bit PCM
        pcm = (tone * 32767).astype(np.int16)
        with wave.open(output_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
    except Exception:
        return False


def debug_speaker(speaker_device: str = None,
                  tts_lang: str = "en") -> dict:
    """Comprehensive speaker diagnostic. Tests every playback path.

    Returns a dict with test results for each method.
    """
    results = {
        "system": {},
        "playback_tools": {},
        "tts_tools": {},
        "playback_tests": {},
        "tts_tests": {},
    }

    print("=" * 60)
    print("  voiceCommandRobot — Speaker Diagnostic")
    print("=" * 60)
    print()

    # --- System info ---
    print("[1/5] System audio info...")
    results["system"]["os"] = sys.platform
    results["system"]["sound_cards"] = os.path.exists("/proc/asound/cards")
    results["system"]["snd_dev"] = os.path.exists("/dev/snd")
    if os.path.exists("/proc/asound/cards"):
        try:
            with open("/proc/asound/cards") as f:
                results["system"]["cards_text"] = f.read().strip()
        except Exception:
            pass
    print(f"  /dev/snd exists     : {results['system']['snd_dev']}")
    print(f"  /proc/asound/cards  : {results['system']['sound_cards']}")
    print()

    # --- Playback tools ---
    print("[2/5] Checking playback tools...")
    for tool in ["aplay", "ffplay", "paplay", "play", "mplayer", "mpv"]:
        path = shutil.which(tool)
        results["playback_tools"][tool] = path is not None
        status = "✓ found" if path else "✗ not found"
        print(f"  {tool:15s}: {status}")
    print()

    # --- TTS tools ---
    print("[3/5] Checking TTS tools...")
    results["tts_tools"]["espeak"] = find_espeak() is not None
    results["tts_tools"]["piper"] = find_piper_tts() is not None
    try:
        import pyttsx3
        results["tts_tools"]["pyttsx3"] = True
    except ImportError:
        results["tts_tools"]["pyttsx3"] = False
    for tool, ok in results["tts_tools"].items():
        status = "✓ found" if ok else "✗ not found"
        print(f"  {tool:15s}: {status}")
    print()

    # --- Playback tests ---
    print("[4/5] Testing WAV playback methods...")
    tmp_wav = tempfile.mktemp(suffix="_test.wav")
    tone_ok = generate_test_tone(tmp_wav)
    print(f"  Test tone generated : {'✓' if tone_ok else '✗'}")

    if tone_ok:
        # ffplay
        print("  Testing ffplay ... ", end="", flush=True)
        try:
            ok = play_audio(tmp_wav, backend="ffplay", alsa_device=speaker_device)
            results["playback_tests"]["ffplay"] = ok
            print("✓" if ok else "✗")
        except Exception as e:
            results["playback_tests"]["ffplay"] = False
            print(f"✗ ({e})")

        # aplay
        print("  Testing aplay ... ", end="", flush=True)
        try:
            ok = play_audio(tmp_wav, backend="alsa", alsa_device=speaker_device)
            results["playback_tests"]["aplay"] = ok
            print("✓" if ok else "✗")
        except Exception as e:
            results["playback_tests"]["aplay"] = False
            print(f"✗ ({e})")

        # sounddevice
        print("  Testing sounddevice ... ", end="", flush=True)
        try:
            ok = play_audio(tmp_wav, backend="sounddevice",
                          alsa_device=speaker_device)
            results["playback_tests"]["sounddevice"] = ok
            print("✓" if ok else "✗")
        except Exception as e:
            results["playback_tests"]["sounddevice"] = False
            print(f"✗ ({e})")

    # Clean up
    try:
        os.remove(tmp_wav)
    except Exception:
        pass
    print()

    # --- TTS tests ---
    print("[5/5] Testing TTS synthesis + playback...")
    sample = "Hello" if tts_lang == "en" else "你好"
    tts_wav = tempfile.mktemp(suffix="_tts.wav")

    # espeak-ng synthesize
    print("  espeak-ng synthesize ... ", end="", flush=True)
    ok = synthesize_with_espeak(sample, tts_wav, voice=tts_lang)
    results["tts_tests"]["espeak_synth"] = ok
    print("✓" if ok else "✗")

    # pyttsx3 synthesize
    print("  pyttsx3 synthesize ... ", end="", flush=True)
    ok = synthesize_with_pyttsx3(sample, tts_wav, lang=tts_lang)
    results["tts_tests"]["pyttsx3_synth"] = ok
    print("✓" if ok else "✗")

    # speak_direct (full pipeline)
    print("  speak_direct() ... ", end="", flush=True)
    ok = speak_direct(sample, lang=tts_lang, speaker_device=speaker_device)
    results["tts_tests"]["speak_direct"] = ok
    print("✓" if ok else "✗")

    # Clean up
    try:
        os.remove(tts_wav)
    except Exception:
        pass
    print()

    # --- Summary ---
    print("=" * 60)
    print("  Summary")
    print("=" * 60)

    # Find a working playback method
    working_playback = [k for k, v in results["playback_tests"].items() if v]
    working_tts = [k for k, v in results["tts_tests"].items() if v]

    if working_playback:
        print(f"  ✓ Working playback: {', '.join(working_playback)}")
    else:
        print("  ✗ NO WORKING PLAYBACK METHOD FOUND")
        print()
        print("  Possible causes:")
        print("    1. No sound card / speaker hardware")
        print("    2. ALSA not configured")
        print("    3. No audio tools installed")

    if working_tts:
        print(f"  ✓ Working TTS: {', '.join(working_tts)}")
    else:
        print("  ✗ NO WORKING TTS METHOD FOUND")
        print()
        print("  To install TTS:")
        print("    sudo apt-get install espeak-ng")
        print("    pip install pyttsx3")

    print("=" * 60)
    return results


def speak_direct(text: str, lang: str = "en",
                 rate: int = 200, volume: float = 1.0,
                 speaker_device: str = None) -> bool:
    """Speak text directly through speakers using the best available method.

    Tries (in order):
      1. pyttsx3 direct playback (most reliable, cross-platform)
      2. espeak direct command-line playback (with proper ALSA config)
      3. ffplay with synthesized WAV (fallback)
    """
    # Set up ALSA environment first to suppress config errors
    try:
        setup_alsa_env(playback_device=speaker_device)
    except Exception:
        pass

    # 1. pyttsx3 direct
    ok = synthesize_with_pyttsx3(text, lang=lang, rate=rate,
                                 volume=volume, play_direct=True)
    if ok:
        return True

    # 2. espeak direct (with proper ALSA device)
    if speak_espeak_direct(text, lang=lang, device=speaker_device):
        return True

    # 3. synthesize + ffplay fallback
    tmp_wav = tempfile.mktemp(suffix=".wav")
    if synthesize_with_pyttsx3(text, output_wav=tmp_wav, lang=lang, rate=rate):
        ok = play_audio(tmp_wav, backend="auto")
        try:
            os.remove(tmp_wav)
        except Exception:
            pass
        return ok

    # 4. Last resort: espeak synthesize WAV + ffplay
    if synthesize_with_espeak(text, tmp_wav, voice=lang):
        ok = play_audio(tmp_wav, backend="auto")
        try:
            os.remove(tmp_wav)
        except Exception:
            pass
        return ok

    return False


def synthesize_say(text: str, output_wav: str,
                  lang: str = "en") -> bool:
    """Try Piper first, fall back to pyttsx3, then espeak-ng.

    Returns True if WAV file was successfully created.
    """
    if synthesize_with_piper(text, output_wav):
        return True
    if synthesize_with_pyttsx3(text, output_wav, lang=lang):
        return True
    if synthesize_with_espeak(text, output_wav, voice=lang):
        return True
    return False


def play_audio(wav_path: str,
               backend: str = "alsa",
               device: int = None,
               alsa_device: str = None) -> bool:
    """Play a WAV file through the speaker.

    Args:
        backend: "alsa" (aplay), "ffplay" (ffmpeg), or "auto" (try available).
        device:  sounddevice device index (for sounddevice backend).
        alsa_device: ALSA device string like "plughw:0,0" (for ALSA backend).
    """
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return False

    def try_backend(name: str, cmd_func) -> bool:
        """Try a backend; return True if it works."""
        try:
            result = subprocess.run(cmd_func(), capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception as exc:
            print(f"[Play] {name} failed: {exc}", file=sys.stderr)
            return False

    if backend == "auto":
        # Try each backend in order of preference
        for b in ("alsa", "ffplay", "sounddevice"):
            if b == "alsa" and shutil.which("aplay"):
                dev = alsa_device or "default"
                if try_backend("aplay", lambda d=dev: ["aplay", "-D", d, "-q", wav_path]):
                    return True
            elif b == "ffplay" and shutil.which("ffplay"):
                # ffplay uses SDL for audio output
                # Set SDL_AUDIODRIVER and AUDIODEV env vars for ALSA device
                env = os.environ.copy()
                if alsa_device:
                    env["SDL_AUDIODRIVER"] = "alsa"
                    env["AUDIODEV"] = alsa_device
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=30, env=env)
                    if result.returncode == 0:
                        return True
                except Exception as exc:
                    print(f"[Play] ffplay failed: {exc}", file=sys.stderr)
            elif b == "sounddevice":
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    data, sr = sf.read(wav_path, dtype="float32")
                    if data.ndim > 1:
                        data = data[:, 0]
                    sd.play(data, sr, device=device)
                    sd.wait()
                    return True
                except Exception as exc:
                    print(f"[Play] sounddevice failed: {exc}", file=sys.stderr)
        return False

    try:
        if backend == "alsa":
            dev = alsa_device or "default"
            cmd = ["aplay", "-D", dev, "-q", wav_path]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
        elif backend == "ffplay":
            # ffplay uses SDL for audio output
            env = os.environ.copy()
            if alsa_device:
                env["SDL_AUDIODRIVER"] = "alsa"
                env["AUDIODEV"] = alsa_device
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
            result = subprocess.run(cmd, capture_output=True, timeout=30, env=env)
            return result.returncode == 0
        elif backend == "sounddevice":
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(wav_path, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            sd.play(data, sr, device=device)
            sd.wait()
            return True
    except Exception as exc:
        print(f"[Play] Error: {exc}", file=sys.stderr)
    return False


def run_chat_loop(robot,
                  duration: float = 5.0,
                  language: str = None,
                  max_iterations: int = None,
                  chat_model_path: str = None,
                  chat_model_type: str = "auto",
                  llm_preset: str = None,
                  chat_system_prompt: str = None,
                  piper_path: str = None,
                  piper_voice: str = None,
                  speaker_device: str = None,
                  speaker_backend: str = "auto",
                  tts_lang: str = "en",
                  verbose: bool = False) -> list:
    """Full voice chat loop: listen -> STT -> LLM chat -> TTS -> speak -> repeat.

    All models run offline on RK3588.

    Args:
        robot:           voiceCommandRobot instance (must have backend resolved).
        duration:        Seconds to listen per turn.
        language:        Whisper language hint.
        max_iterations:  Max turns. None = unlimited (Ctrl+C to stop).
        chat_model_path: Path to GGUF LLM model. None = auto-find.
        chat_model_type: "llama", "qwen2", "phi2", "auto", "chatml".
        llm_preset:      Named model preset (qwen2-0.5b, qwen2-1.5b, tinyllama, qwen2-7b).
                         Auto-downloads to ~/.cache/llm/ if not present.
        chat_system_prompt: System prompt for the LLM.
        piper_path:      Path to Piper executable. None = auto-detect.
        piper_voice:     Piper voice model file. None = default.
        speaker_device:  ALSA device for aplay, e.g. "plughw:0,0".
        speaker_backend: Playback backend: auto, alsa, ffplay, sounddevice.
        tts_lang:        espeak language code: "en", "zh", etc.
    """
    import threading

    results = []
    backend = robot._resolve_backend()
    aplay_dev = speaker_device

    # --- 1. Load Whisper (cached) ---
    print()
    print("=" * 60)
    print("  voiceCommandRobot — Offline Voice Chat Mode")
    print("=" * 60)
    print(f"  ASR backend : {backend}")
    print(f"  ASR lang    : {language or 'auto'}")
    print(f"  TTS         : Piper → espeak-ng (offline)")
    print(f"  TTS lang    : {tts_lang}")
    print(f"  Speaker     : {aplay_dev or 'default'}")
    if llm_preset:
        print(f"  LLM preset  : {llm_preset}")
    print(f"  Output file : {robot.output_file}")
    print("=" * 60)
    print()
    print("  Conversation will be saved to text file.")
    print("  Press Ctrl+C to stop.")
    print()

    print("[Chat] Loading Whisper model ...")
    model = robot._get_whisper_model(language)
    transcribe_kwargs = {"fp16": False}
    if language:
        transcribe_kwargs["language"] = language
    print()

    # --- 2. Load Chat LLM ---
    llm_info = None
    if chat_model_path is not None or llm_preset is not None or True:  # always try
        try:
            llm_info = init_chat_model(
                model_path=chat_model_path,
                model_type=chat_model_type,
                llm_preset=llm_preset,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[Chat] Warning: {exc}")
            print("[Chat] Chat LLM not available. Running in STT-only mode.")
        except ImportError:
            print("[Chat] Warning: llama-cpp-python not installed.")
            print("[Chat]   pip install llama-cpp-python")
            print("[Chat] Chat LLM not available. Running in STT-only mode.")
    print()

    # --- 3. TTS check ---
    tts_available = False
    try:
        import pyttsx3
        tts_available = True
        print(f"[Chat] TTS ({tts_lang}) is available (pyttsx3 / espeak).")
    except ImportError:
        tts_available = synthesize_say("test", "/tmp/tts_test.wav", lang=tts_lang)
        if tts_available:
            print(f"[Chat] TTS ({tts_lang}) is available (Piper/espeak-ng).")
    if not tts_available:
        print(f"[Chat] Warning: No TTS found. Install one of:")
        print(f"[Chat]   pyttsx3:   pip install pyttsx3   (recommended, cross-platform)")
        print(f"[Chat]   espeak-ng: sudo apt-get install espeak-ng")
        print(f"[Chat]   Piper:     https://github.com/rhasspy/piper/releases")
        print(f"[Chat] Continuing in text-only chat mode.")
    print()
    print("[Chat] Ready. Speak now!")
    print()

    # --- Shared state for record thread ---
    wav_path_ref = [None]
    record_error_ref = [None]
    record_lock = threading.Lock()

    def record_one():
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        exc = None
        try:
            if backend == "sounddevice":
                record_from_microphone(path, duration_seconds=duration,
                                     device=robot.device)
            elif backend == "alsa":
                record_with_arecord(path, duration_seconds=duration,
                                   device=robot.device)
            elif backend == "pyaudio":
                record_with_pyaudio(path, duration_seconds=duration,
                                  device=robot.device)
        except Exception as e:
            exc = e
        with record_lock:
            wav_path_ref[0] = path
            record_error_ref[0] = exc

    # Start first recording
    t = threading.Thread(target=record_one, daemon=True)
    t.start()

    count = 0
    try:
        while True:
            t.join()
            count += 1
            stamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Grab recording
            with record_lock:
                pending_path = wav_path_ref[0]
                pending_exc = record_error_ref[0]

            print(f"\n[{count}] {stamp}")
            result_entry = {
                "skill": "voiceCommandRobot",
                "ok": False,
                "iteration": count,
                "text": "",
                "response": "",
                "error": None,
            }

            if pending_exc:
                msg = f"Record error: {pending_exc}"
                print(f"  ! {msg}")
                result_entry["error"] = msg
            else:
                # --- STT ---
                print(f"  Listening... (transcribing)")
                try:
                    asr_result = model.transcribe(pending_path, **transcribe_kwargs)
                    text = (asr_result.get("text") or "").strip()
                except Exception as exc:
                    text = ""
                    print(f"  ! ASR error: {exc}")

                print()
                print("=" * 60)
                print(f"  [YOU]  {text}")
                print("=" * 60)
                result_entry["text"] = text
                write_to_text_file(f"[{timestamp()}] [YOU] {text}", robot.output_file, append=True)

                if text:
                    # --- Chat LLM ---
                    reply = ""
                    if llm_info is not None:
                        print(f"  Thinking...")
                        try:
                            reply = chat_with_model(
                                llm_info, text,
                                system_prompt=chat_system_prompt,
                                verbose=verbose,
                            )
                        except Exception as exc:
                            reply = ""
                            print(f"  ! LLM error: {exc}")

                    result_entry["response"] = reply
                    result_entry["ok"] = True

                    if reply:
                        # Print to command line
                        print(f"  [BOT]  {reply}")
                        print("=" * 60)
                        print()
                        # Write to text file
                        write_to_text_file(f"[{timestamp()}] [BOT] {reply}", robot.output_file, append=True)
                        write_to_text_file("-" * 60, robot.output_file, append=True)

                        # --- TTS + Play ---
                        if tts_available:
                            print(f"  Speaking...")
                            # Try direct playback first (pyttsx3, espeak)
                            # which is more reliable than WAV + separate player
                            spoke = speak_direct(reply, lang=tts_lang,
                                                speaker_device=aplay_dev)
                            if not spoke:
                                # Fallback: synthesize to WAV + play
                                out_wav = tempfile.mktemp(suffix=".wav")
                                ok = synthesize_say(reply, out_wav, lang=tts_lang)
                                if ok:
                                    play_audio(out_wav, backend=speaker_backend,
                                             alsa_device=aplay_dev)
                                    try:
                                        os.remove(out_wav)
                                    except Exception:
                                        pass
                                else:
                                    print(f"  ! TTS synthesis failed")
                        else:
                            print(f"  (TTS not available — text-only response)")
                else:
                    print(f"  (no speech detected)")
                    result_entry["ok"] = True

            results.append(result_entry)

            if max_iterations and count >= max_iterations:
                print(f"\n[Chat] Max iterations ({max_iterations}) reached.")
                break

            # Start next recording immediately
            with record_lock:
                wav_path_ref[0] = None
                record_error_ref[0] = None
            t = threading.Thread(target=record_one, daemon=True)
            t.start()

    except KeyboardInterrupt:
        print(f"\n[Chat] Stopped by user.")

    print()
    print(f"[Chat] Total turns: {len(results)}")
    print(f"[Chat] Transcript: {robot.output_file}")
    return results


# ------------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------------

def run(args: argparse.Namespace = None) -> dict:
    """Agent-friendly entry. Returns a JSON-serializable dict."""
    if args is None:
        args = parse_args()

    if args.list_llm:
        print(list_llm_presets())
        return {"skill": "voiceCommandRobot", "ok": True, "llm_listed": True}

    if args.fix_audio:
        print("=== voiceCommandRobot — Audio Fix ===")
        print()
        env = setup_alsa_env()
        print()
        print("ALSA environment variables set:")
        for k, v in env.items():
            print(f"  {k}={v}")
        print()
        print("Done. Try running with --test-speaker to verify.")
        return {"skill": "voiceCommandRobot", "ok": True, "audio_fixed": True}

    if args.test_speaker:
        print("=== voiceCommandRobot — Speaker Test ===")
        print()
        setup_alsa_env(playback_device=args.speaker)
        lang = args.tts_lang or "en"
        sample_texts = {
            "zh": "你好，这是一个语音测试。",
            "en": "Hello, this is a voice test.",
            "cmn": "你好，这是一个语音测试。",
        }
        text = sample_texts.get(lang, sample_texts["en"])
        print(f"Language : {lang}")
        print(f"Text     : {text}")
        print(f"Speaker  : {args.speaker or 'auto'}")
        print()
        print("Speaking...")
        ok = speak_direct(text, lang=lang, speaker_device=args.speaker)
        print()
        if ok:
            print("✅ Speaker test passed!")
        else:
            print("❌ Speaker test failed.")
            print()
            print("Try:")
            print("  1. Run --fix-audio first")
            print("  2. Install espeak-ng: sudo apt-get install espeak-ng")
            print("  3. Install pyttsx3: pip install pyttsx3")
            print("  4. Check with: aplay -l")
        return {"skill": "voiceCommandRobot", "ok": ok, "speaker_test": ok}

    if args.debug_speaker:
        results = debug_speaker(speaker_device=args.speaker,
                                tts_lang=args.tts_lang or "en")
        return {"skill": "voiceCommandRobot", "ok": True, "debug": results}

    if args.test_chat:
        print("=" * 60)
        print("  voiceCommandRobot — LLM Chat Test")
        print("=" * 60)
        print()

        verbose = args.debug_chat

        # Load the model
        print("[1/3] Loading LLM model ...")
        try:
            llm_info = init_chat_model(
                model_path=args.chat_model,
                model_type=args.chat_model_type,
                llm_preset=args.llm,
            )
            print(f"  Model: {llm_info['model_path']}")
            print(f"  Type : {llm_info['model_type']}")
            print("  ✓ Model loaded successfully")
        except Exception as exc:
            print(f"  ✗ Failed to load model: {exc}")
            print()
            print("  Try:")
            print("    python3 voiceCommandRobot.py --test-chat --llm qwen2-0.5b")
            print("    python3 voiceCommandRobot.py --list-llm")
            return {"skill": "voiceCommandRobot", "ok": False, "error": str(exc)}
        print()

        # Test message
        test_message = "你好，请介绍一下你自己。" if args.language and args.language.startswith("zh") else "Hello, introduce yourself."
        print(f"[2/3] Sending test message ...")
        print(f"  User : {test_message}")
        print()

        # Get reply
        print("[3/3] Waiting for reply ...")
        reply = chat_with_model(
            llm_info,
            test_message,
            system_prompt=args.chat_system_prompt,
            verbose=verbose,
        )
        print()

        if reply:
            print("=" * 60)
            print(f"  [BOT] {reply}")
            print("=" * 60)
            print()
            print("  ✓ Chat test passed!")
        else:
            print("  ✗ Empty reply — model returned nothing")
            print()
            print("  Possible causes:")
            print("    1. Wrong model type (try --chat-model-type llama or chatml)")
            print("    2. Model file corrupted (re-download)")
            print("    3. Not enough RAM (try a smaller model like qwen2-0.5b)")
            print()
            print("  Run with --debug-chat for more info:")
            print("    python3 voiceCommandRobot.py --test-chat --debug-chat --llm qwen2-0.5b")

        return {"skill": "voiceCommandRobot", "ok": bool(reply), "reply": reply,
                "model": llm_info.get("model_path")}

    if args.list_devices:
        backend = args.backend
        if backend == "auto":
            backend = detect_backend()
        print(f"Backend: {backend}")
        print()
        if backend == "pyaudio":
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
        elif backend == "sounddevice":
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
            print("  Install pyaudio: pip install pyaudio  (recommended, cross-platform)")
            print("  Or install sounddevice: pip install sounddevice")
            print("  Or install ALSA: sudo apt-get install alsa-utils")
        return {"skill": "voiceCommandRobot", "ok": True,
                "devices_listed": True, "backend": backend}

    robot = voiceCommandRobot(model_name=args.model,
                              output_file=args.output,
                              device=args.device,
                              backend=args.backend)

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
        write_to_text_file(args.text, robot.output_file, append=True)
        return {
            "skill": "voiceCommandRobot",
            "ok": True,
            "text": args.text,
            "audio_file": None,
            "output_file": robot.output_file,
            "error": None,
        }

    return robot.run_once(duration=args.duration,
                          language=args.language,
                          audio_path=args.audio)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="voiceCommandRobot — mic -> WAV -> Whisper -> text file."
    )
    p.add_argument("--model", default="tiny",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (default: tiny).")
    p.add_argument("--duration", type=float, default=5.0,
                   help="Seconds to record from the mic (default: 5.0).")
    p.add_argument("--language", default=None,
                   help="Optional language hint for Whisper, e.g. 'en' or 'zh'.")
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
                   choices=["auto", "pyaudio", "sounddevice", "alsa"],
                   help="Audio recording backend (default: auto — "
                        "tries pyaudio, sounddevice, then ALSA). "
                        "Use 'alsa' on RK3588 without PortAudio. "
                        "Use 'pyaudio' for cross-platform support.")
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
    p.add_argument("--chat", action="store_true",
                   help="Enable offline voice chat: listen -> STT -> LLM -> TTS -> speak. "
                        "Requires llama-cpp-python and a GGUF model.")
    p.add_argument("--chat-model", default=None,
                   help="Path to a GGUF chat model file (e.g. tinyllama.gguf). "
                        "Auto-finds in ~/.cache/llama.cpp/ if not given.")
    p.add_argument("--chat-model-type", default="auto",
                   choices=["llama", "qwen2", "phi2", "auto", "chatml"],
                   help="GGUF model architecture (default: auto).")
    p.add_argument("--llm", default=None,
                   choices=list(LLM_MODEL_PRESETS.keys()),
                   help="Named LLM preset for RK3588 — auto-downloads if missing. "
                        "E.g. qwen2-0.5b (default, zh+en), qwen2-1.5b, tinyllama, qwen2-7b. "
                        "Use --list-llm to see all.")
    p.add_argument("--list-llm", action="store_true",
                   help="List all available LLM model presets for RK3588 and exit.")
    p.add_argument("--fix-audio", action="store_true",
                   help="Fix ALSA audio configuration on RK3588 (creates ~/.asoundrc) and exit.")
    p.add_argument("--test-speaker", action="store_true",
                   help="Test speaker output with a sample voice and exit. "
                        "Use with --speaker and --tts-lang.")
    p.add_argument("--debug-speaker", action="store_true",
                   help="Run comprehensive speaker diagnostic (tests all playback/TTS methods) and exit.")
    p.add_argument("--test-chat", action="store_true",
                   help="Test LLM chat without microphone (type a message, get a reply). "
                        "Use with --llm or --chat-model.")
    p.add_argument("--debug-chat", action="store_true",
                   help="Enable verbose debug output for chat LLM (prints full response JSON).")
    p.add_argument("--chat-system-prompt", default=None,
                   help="System prompt for the chat LLM. "
                        "Example: 'You are a helpful robot assistant.'")
    p.add_argument("--piper", default=None,
                   help="Path to Piper TTS executable. "
                        "Download: https://github.com/rhasspy/piper/releases")
    p.add_argument("--piper-voice", default=None,
                   help="Path to Piper .onnx voice model file. "
                        "Example: en_US-lessac-medium.onnx")
    p.add_argument("--speaker", default=None,
                   help="ALSA device for speaker output, e.g. 'plughw:0,0'. "
                        "Use aplay -l to check device numbers.")
    p.add_argument("--speaker-backend", default="auto",
                   choices=["auto", "alsa", "ffplay", "sounddevice"],
                   help="Audio playback backend (default: auto — tries aplay, "
                        "ffplay, then sounddevice).")
    p.add_argument("--tts-lang", default="en",
                   help="TTS language code for espeak-ng fallback: 'en', 'zh', etc.")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()

    if _args.no_ssl_verify:
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass

    if _args.chat:
        robot = voiceCommandRobot(model_name=_args.model,
                                output_file=_args.output,
                                device=_args.device,
                                backend=_args.backend)
        chat_results = run_chat_loop(
            robot,
            duration=_args.duration,
            language=_args.language,
            max_iterations=_args.max,
            chat_model_path=_args.chat_model,
            chat_model_type=_args.chat_model_type,
            llm_preset=_args.llm,
            chat_system_prompt=_args.chat_system_prompt,
            piper_path=_args.piper,
            piper_voice=_args.piper_voice,
            speaker_device=_args.speaker,
            speaker_backend=_args.speaker_backend,
            tts_lang=_args.tts_lang,
            verbose=_args.debug_chat,
        )
        print(json.dumps({
            "skill": "voiceCommandRobot",
            "ok": True,
            "mode": "chat",
            "total_turns": len(chat_results),
            "results": chat_results,
        }, ensure_ascii=False, indent=2))
        sys.exit(0)

    _result = run(_args)

    if _args.loop:
        robot = voiceCommandRobot(model_name=_args.model,
                                output_file=_args.output,
                                device=_args.device,
                                backend=_args.backend)
        loop_results = robot.run_loop(duration=_args.duration,
                                      language=_args.language,
                                      max_iterations=_args.max)
        print(json.dumps({
            "skill": "voiceCommandRobot",
            "ok": True,
            "mode": "loop",
            "total_iterations": len(loop_results),
            "results": loop_results,
        }, ensure_ascii=False, indent=2))
        sys.exit(0)
    else:
        print()
        print(json.dumps(_result, ensure_ascii=False, indent=2))
        sys.exit(0 if _result.get("ok") else 1)
