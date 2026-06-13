#!/usr/bin/env python3
"""
voiceCommandToRobot
--------------------
A skill that enables a human to command a robot by voice on Ubuntu.

Pipeline (matching the reference Whisper-based pattern):
  1. PyAudio microphone capture (16-bit, 16 kHz, mono).
  2. webrtcvad-based voice activity detection to isolate utterances.
  3. OpenAI Whisper offline speech recognition.
  4. Parse text → robot command (FORWARD / LEFT / RIGHT / BACK / TURN180 / RETURN / STOP.
  5. Update (x, y, heading) & print the command.
  6. Append the transcribed text + interpreted command to a text file.

Usage:
  python3 voiceCommandToRobot.py                        # one voice command
  python3 voiceCommandToRobot.py --test                  # run all 7 built-in test scenarios
  python3 voiceCommandToRobot.py --continuous           # keep listening
  python3 voiceCommandToRobot.py --list-devices
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import tempfile
import time
import wave
from typing import Optional, List, Tuple, Dict

# Audio libraries are imported lazily because not every mode needs them.
# An agent can invoke --text / --history / --clear-memory / --test / --list-devices
# without the audio stack installed. See _CAPABILITIES at the end of this block.
#
# import pyaudio      → loaded by _require_audio() inside record_audio / __init__
# import webrtcvad    → loaded by _require_audio() inside record_audio / __init__
# import whisper      → loaded by _require_whisper() inside transcribe_audio / __init__


# -------------------------- Audio Configuration --------------------------
# paInt16 (16-bit signed integer PCM) — hard-coded to the numeric value used
# by PortAudio so this module is importable and most modes work even when
# PyAudio is not installed.
FORMAT = 8
CHANNELS = 1
RATE = 16000
CHUNK = 512
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
MAX_RECORD_SECONDS = 30
SILENCE_FRAMES_REQUIRED = int(1.0 * RATE / CHUNK_SIZE)


# --------------------- Lazy loading of audio libraries --------------------
# These helpers keep non-audio modes (--text, --history, --clear-memory,
# --test, --list-devices) working on machines without pyaudio/webrtcvad/whisper.
# An agent can still classify phrases, inspect memory, and run tests offline.

def _try_import(name: str):
    """Return the module object if available, else None (no crash)."""
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:
        return None


def _require_audio():
    """Try to import pyaudio + webrtcvad. Raises RuntimeError if missing.

    The caller must have guarded an agent call path with a graceful
    "libraries missing" branch instead of letting the import raise.
    """
    pa = _try_import("pyaudio")
    vad = _try_import("webrtcvad")
    if pa is None or vad is None:
        missing = [n for n, m in (("pyaudio", pa), ("webrtcvad", vad))
                   if m is None]
        raise RuntimeError(
            "Audio libraries not available (missing: "
            + ", ".join(missing)
            + "). Install with: pip install pyaudio webrtcvad "
              "(requires portaudio19-dev on Ubuntu)."
        )
    return pa, vad


def _require_whisper():
    """Import openai-whisper. Raises RuntimeError if missing."""
    whisper_mod = _try_import("whisper")
    if whisper_mod is None:
        raise RuntimeError(
            "openai-whisper is not installed. "
            "Install with: pip install openai-whisper torch ffmpeg-python "
            "(plus 'sudo apt-get install -y portaudio19-dev ffmpeg' on Ubuntu)."
        )
    return whisper_mod


# Capabilities an agent can inspect:
#   _CAPABILITIES = {"pyaudio": True/False, "webrtcvad": True/False, "whisper": True/False}
_CAPABILITIES = {
    "pyaudio": _try_import("pyaudio") is not None,
    "webrtcvad": _try_import("webrtcvad") is not None,
    "whisper": _try_import("whisper") is not None,
}


# ------------------------- Robot Command Catalog ---------------------------
# Canonical command keys used internally. Keyword patterns below are forgiving:
#   - Case insensitive
#   - Plurals and common mis-transcriptions handled

ROBOT_COMMANDS: Dict[str, Dict] = {
    "FORWARD": {
        "description": "Move forward by one step along current heading.",
        "keywords": [
            "forward", "go forward", "go ahead", "move forward",
            "step forward", "walk forward", "向前", "前进", "往前走",
            "向前走", "直走",
        ],
    },
    "LEFT": {
        "description": "Turn left by 90 degrees.",
        "keywords": [
            "left", "turn left", "rotate left", "go left", "向左转",
            "左转", "向左", "左拐",
        ],
    },
    "RIGHT": {
        "description": "Turn right by 90 degrees.",
        "keywords": [
            "right", "turn right", "rotate right", "go right", "向右转",
            "右转", "向右", "右拐",
        ],
    },
    "BACK": {
        "description": "Move backward by one step.",
        "keywords": [
            "back", "backward", "backwards", "reverse", "go back",
            "move back", "后退", "向后", "向后走", "倒退", "倒车",
        ],
    },
    "TURN180": {
        "description": "Turn around 180 degrees (inverse heading).",
        "keywords": [
            "180", "turn around", "about face", "inverse",
            "u turn", "uturn", "one eighty", "掉头", "转180", "转180度",
            "向后转", "掉头 180 度", "调头", "反转",
        ],
    },
    "RETURN": {
        "description": "Retrace all steps back to the start point.",
        "keywords": [
            "return", "return to home", "return to start",
            "return to origin", "go home", "back to start",
            "返回原点", "回到起点", "回家", "返航",
        ],
    },
    "STOP": {
        "description": "Stop / halt / end session.",
        "keywords": [
            "stop", "halt", "exit", "quit", "finish", "end",
            "停", "停止", "停下", "结束",
        ],
    },
    "SLOW_DOWN": {
        "description": "Reduce robot speed.",
        "keywords": [
            "slow down", "slow", "slower", "reduce speed",
            "decelerate", "减速", "慢一点", "慢下来", "慢点",
        ],
    },
    "SPEED_UP": {
        "description": "Increase robot speed.",
        "keywords": [
            "speed up", "speed", "faster", "increase speed",
            "accelerate", "go faster", "加速", "快一点", "加快",
            "快点",
        ],
    },
}

# Pre-compiled regex keyed by command. Matching is case-insensitive,
# word-boundary aware for English, and substring-safe for Chinese.
def _compile_command_patterns() -> Dict[str, "re.Pattern"]:
    patterns = {}
    for key, spec in ROBOT_COMMANDS.items():
        tokens = [re.escape(k.strip()) for k in spec["keywords"] if k.strip()]
        if not tokens:
            continue
        patterns[key] = re.compile(
            r"(?:^|[^a-z0-9])(" + "|".join(tokens) + r")(?:$|[^a-z0-9])",
            re.IGNORECASE,
        )
    return patterns

ROBOT_COMMAND_PATTERNS: Dict[str, "re.Pattern"] = _compile_command_patterns()

# Error messages used both on terminal and when writing to output file/memory.
ERR_TOO_QUIET = "I cannot hear you, please speak louder."
ERR_CANNOT_UNDERSTAND = "I cannot understand you, please repeat."
ERR_OUT_OF_LIBRARY = "Sorry, I cannot understand, please repeat."

# How far one forward/back moves in a single command (purely bookkeeping units).
STEP_DISTANCE = 1.0

# ------------------------- Long-term memory -----------------------------
# Persisted JSON next to the output text file so the history of voice
# commands survives across restarts of the skill.
def _default_memory_path(output_file: str) -> str:
    base = os.path.splitext(os.path.abspath(output_file))[0]
    return f"{base}_memory.json"

MEMORY_VERSION = 1


class voiceCommandToRobot:
    """Captures voice from the mic, transcribes via Whisper, prints & saves."""

    # -----------------------------------------------------------------
    # Init
    # -----------------------------------------------------------------
    def __init__(self,
                 model_name: str = "tiny",
                 output_file: str = "voice_commands.txt",
                 language: Optional[str] = None,
                 device_index: Optional[int] = None,
                 need_mic: bool = True,
                 need_whisper: bool = True):
        """Create a skill instance.

        Parameters
        ----------
        need_mic : bool
            Open the default microphone now. Set False for:
              --text, --history, --clear-memory, --test, --list-devices.
        need_whisper : bool
            Load the Whisper model now. Set False when transcription is
            not required (keyword classifier and memory modes still work).
        """

        self.output_file = os.path.abspath(output_file)
        self.model_name = model_name
        self.language = language
        self.device_index = device_index

        # Long-term memory: JSON file that persists every voice command
        # across restarts of the skill. Seed from disk if present.
        self.memory_path = _default_memory_path(output_file)
        self.memory: List[Dict] = self.load_memory()

        # Robot state: start at origin, heading=0° (positive x).
        self.x, self.y = 0.0, 0.0
        self.heading_deg = 0.0
        self.speed_multiplier = 1.0
        self.history: List[Tuple[str, float, float, float]] = []

        # Whisper model (optional — lazy)
        self.whisper_model = None
        if need_whisper:
            print(f"[voiceCommandToRobot] Loading Whisper model "
                  f"({model_name}) ...")
            whisper_mod = _require_whisper()
            self.whisper_model = whisper_mod.load_model(model_name)

        # Mic (optional — lazy)
        self.p = None
        self.stream = None
        self.vad = None
        if need_mic:
            pa, vad_mod = _require_audio()
            self.p = pa.PyAudio()
            try:
                self.stream = self.p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK,
                )
            except Exception as e:
                # Don't kill the process — let the caller handle it
                # (typically via run() returning ok=False JSON).
                self.p.terminate()
                self.p = None
                raise RuntimeError(f"Could not open microphone: {e}")
            self.vad = vad_mod.Vad(3)

        # Ensure output file exists (append later)
        if not os.path.exists(self.output_file):
            try:
                with open(self.output_file, "w", encoding="utf-8"):
                    pass
            except OSError as e:
                raise RuntimeError(
                    f"Cannot create output file {self.output_file}: {e}"
                )

        print(f"[voiceCommandToRobot] Output file : {self.output_file}")
        print(f"[voiceCommandToRobot] Memory file : {self.memory_path} "
              f"({len(self.memory)} entries loaded)")
        print(f"[voiceCommandToRobot] Ready.\n")

    # -----------------------------------------------------------------
    # Keyword → command parser
    # -----------------------------------------------------------------
    def parse_robot_command(self, text: str) -> Optional[str]:
        """Return a canonical ROBOT_COMMAND key or None."""
        if not text:
            return None
        candidate = text.lower().strip()
        # Check more specific / longer commands first so e.g. "turn around"
        # beats generic "forward" when user says "turn around".
        priority = [
            "RETURN", "TURN180", "SLOW_DOWN", "SPEED_UP",
            "LEFT", "RIGHT", "BACK", "FORWARD", "STOP",
        ]
        for key in priority:
            pattern = ROBOT_COMMAND_PATTERNS.get(key)
            if pattern is None:
                continue
            if pattern.search(candidate):
                return key
        return None

    # -----------------------------------------------------------------
    # Robot state update
    # -----------------------------------------------------------------
    def _apply_move(self, sign: float) -> None:
        """Move `sign` * STEP_DISTANCE along current heading."""
        rad = math.radians(self.heading_deg)
        self.x += sign * STEP_DISTANCE * math.cos(rad)
        self.y += sign * STEP_DISTANCE * math.sin(rad)

    def execute_robot_command(self, cmd: str) -> bool:
        """Update (x, y, heading). Returns True if the command was handled."""
        if cmd is None:
            return False
        if cmd == "FORWARD":
            self._apply_move(+1.0)
        elif cmd == "BACK":
            self._apply_move(-1.0)
        elif cmd == "LEFT":
            self.heading_deg = (self.heading_deg - 90.0) % 360.0
        elif cmd == "RIGHT":
            self.heading_deg = (self.heading_deg + 90.0) % 360.0
        elif cmd == "TURN180":
            self.heading_deg = (self.heading_deg + 180.0) % 360.0
        elif cmd == "RETURN":
            # Retrace the path in reverse order with inverted moves.
            self.retrace_to_start()
        elif cmd == "STOP":
            # Acknowledged; we just record it.
            pass
        elif cmd == "SLOW_DOWN":
            self.speed_multiplier = max(0.25, self.speed_multiplier * 0.5)
        elif cmd == "SPEED_UP":
            self.speed_multiplier = min(4.0, self.speed_multiplier * 2.0)
        else:
            return False

        if cmd != "RETURN":
            self.history.append((cmd, self.x, self.y, self.heading_deg))
        return True

    def retrace_to_start(self) -> List[str]:
        """Walk the recorded path in reverse. Returns a list of human readable steps."""
        steps = []
        # Iterate in reverse order, ignoring non-motion & flipping each step, ignoring non-motion entries.
        # Build reverse moves:
        #   for backward motion commands, the reverse is its opposite motion along each move reversed.
        #
        #  - FORWARD ↔ BACK, and LEFT↔RIGHT, and we need to reverse the order too.
        reversed_entries = list(reversed(self.history))
        for (cmd, px, py, ph) in reversed_entries:
            # Each entry already has its effect in reverse direction already contains all recorded moves;
            #   -reverse of "RETURN" and "STOP" don't move.
            if cmd in ("RETURN", "STOP"):
                continue
            # Build inverse:
            if cmd == "FORWARD":
                self._apply_move(-1.0)
                steps.append("BACK (retracing)")
            elif cmd == "BACK":
                self._apply_move(+1.0)
                steps.append("FORWARD (retracing)")
            elif cmd == "LEFT":
                self.heading_deg = (self.heading_deg + 90.0) % 360.0
                steps.append("RIGHT (retracing)")
            elif cmd == "RIGHT":
                self.heading_deg = (self.heading_deg - 90.0) % 360.0
                steps.append("LEFT (retracing)")
            elif cmd == "TURN180":
                self.heading_deg = (self.heading_deg + 180.0) % 360.0
                steps.append("TURN180 (retracing)")
        self.history.append(("RETURN", self.x, self.y, self.heading_deg))
        return steps

    # -----------------------------------------------------------------
    # Pretty-print robot state
    # -----------------------------------------------------------------
    def print_state(self, label: str = "") -> None:
        print(f"  📍 {label or 'Position'}: "
              f"(x={self.x:.1f}, y={self.y:.1f}) | heading={self.heading_deg:.0f}°")

    # -----------------------------------------------------------------
    # Mic (PyAudio + webrtcvad)
    # -----------------------------------------------------------------
    # RMS "volume" threshold. PCM 16-bit signed: full-scale is 32768.
    #   * < 200  — essentially silent / too quiet to understand.
    #   * < 500  — background noise / a very quiet speaker.
    # These are also used by --test scenarios to exercise the
    # "I cannot hear you" / "I cannot understand you" branches.
    PEAK_RMS_TOO_QUIET = 500

    @staticmethod
    def chunk_rms(data: bytes) -> int:
        """RMS of a single 16-bit mono PCM chunk."""
        if not data or len(data) % 2 != 0:
            return 0
        total = 0
        for i in range(0, len(data), 2):
            sample = int.from_bytes(data[i:i + 2], "little", signed=True)
            total += sample * sample
        return int((total / (len(data) // 2)) ** 0.5)

    def is_human_voice(self, raw_bytes: bytes) -> bool:
        try:
            return self.vad.is_speech(raw_bytes, RATE)
        except Exception:
            return False

    def record_audio(self, filename: str):
        """
        Record audio from the open mic stream.

        Returns a tuple (ok, peak_rms):
            ok        True when a WAV file was written, False otherwise.
            peak_rms  Peak RMS level across all recorded chunks (0 if N/A).

        The caller is responsible for interpreting the peak:
            ok == False           → microphone problem / never detected voice.
            ok and peak_rms low   → user spoke, but very quietly.
        """
        if self.stream is None:
            return False, 0
        print("🎤 Listening ... Speak now.")
        frames: List[bytes] = []
        voice_started = False
        silence_chunks = 0
        peak_rms = 0
        max_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK)

        for _ in range(max_chunks):
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"[voiceCommandToRobot] Stream read error: {e}",
                      file=sys.stderr)
                return False, peak_rms
            frames.append(data)
            level = self.chunk_rms(data)
            if level > peak_rms:
                peak_rms = level
            voice = self.is_human_voice(data)
            if voice:
                if not voice_started:
                    print("🟢 Voice detected, recording ...")
                voice_started = True
                silence_chunks = 0
            elif voice_started:
                silence_chunks += 1
                if silence_chunks >= SILENCE_FRAMES_REQUIRED:
                    print("⏹  Silence detected, stopping.")
                    break
        if not voice_started:
            print("[voiceCommandToRobot] No voice detected.")
            return False, peak_rms

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
        print(f"[voiceCommandToRobot] Saved recording to: {filename} "
              f"(peak_rms={peak_rms})")
        return True, peak_rms

    # -----------------------------------------------------------------
    # Transcribe via Whisper
    # -----------------------------------------------------------------
    def transcribe_audio(self, filename: str) -> str:
        try:
            print("🧠 Transcribing with Whisper ...")
            kwargs = {"fp16": False}
            if self.language:
                kwargs["language"] = self.language
            result = self.whisper_model.transcribe(filename, **kwargs)
            text = (result.get("text") or "").strip()
        except Exception as e:
            print(f"[voiceCommandToRobot] Transcription failed: {e}",
                  file=sys.stderr)
            return ""
        return text

    # -----------------------------------------------------------------
    # Append to output file
    # -----------------------------------------------------------------
    def save_command(self, text: str, robot_cmd: Optional[str]) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        robot_label = robot_cmd if robot_cmd else "UNKNOWN"
        line = f"[{timestamp}] {robot_label:8s} \"{text}\"  " \
               f"(x={self.x:.1f} y={self.y:.1f} heading={self.heading_deg:.0f}°)\n"
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(line)

    # -----------------------------------------------------------------
    # One full cycle
    # -----------------------------------------------------------------
    def get_command(self) -> str:
        """Record one utterance, transcribe, parse, update state, save.
        Returns the transcribed text (may be "" on failure)."""
        if self.stream is None:
            return ""
        wav_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ).name
        try:
            ok, peak_rms = self.record_audio(wav_path)
            if not ok:
                # No voice detected (user never spoke / mic problem).
                self._emit_error(ERR_TOO_QUIET, label="NO_VOICE")
                return ""
            if peak_rms < self.PEAK_RMS_TOO_QUIET:
                # Voice detected, but very quiet.
                self._emit_error(
                    ERR_TOO_QUIET, label="TOO_QUIET",
                    extra=f"peak_rms={peak_rms}",
                )
                return ""
            input_text = self.transcribe_audio(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if not input_text:
            # Whisper produced empty text — transcribe failure.
            self._emit_error(ERR_CANNOT_UNDERSTAND, label="TRANSCRIBE_FAIL")
            return ""

        robot_cmd = self.parse_robot_command(input_text)
        if robot_cmd is None:
            # Valid audio + transcription, but not in the command library.
            self._emit_error(
                ERR_OUT_OF_LIBRARY, label="OUT_OF_LIBRARY",
                extra=f"heard=\"{input_text}\"",
            )
            return input_text

        self._print_and_save(input_text)
        return input_text

    def feed_text_command(self, text: str) -> Optional[str]:
        """Directly feed text in test / programmatic use."""
        if not text:
            self._emit_error(ERR_CANNOT_UNDERSTAND, label="EMPTY_TEXT")
            return None
        robot_cmd = self.parse_robot_command(text)
        if robot_cmd is None:
            self._emit_error(ERR_OUT_OF_LIBRARY, label="OUT_OF_LIBRARY",
                             extra=f"heard=\"{text}\"")
            return None
        self._print_and_save(text)
        return robot_cmd

    def _emit_error(self, message: str, label: str = "ERROR",
                    extra: str = "") -> None:
        """
        Print an error message on the terminal, append it to the
        text output file, and record it in long-term memory. This
        ensures a human troubleshooting later can see exactly what
        went wrong.
        """
        print()
        print("=" * 60)
        print(f"  ⚠️  [{label}] {message}")
        if extra:
            print(f"     ({extra})")
        print("=" * 60)
        print()
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {label:14s} \"{message}\"  " \
                   f"(x={self.x:.1f} y={self.y:.1f} heading={self.heading_deg:.0f}°)\n"
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(line)
            self._record_to_memory(message, label)
        except OSError as exc:
            print(f"[voiceCommandToRobot] Write failed: {exc}", file=sys.stderr)

    def _print_and_save(self, text: str) -> Optional[str]:
        print()
        print("=" * 60)
        print(f"🤖 HEARD: {text}")
        robot_cmd = self.parse_robot_command(text)
        if robot_cmd is None:
            # Shouldn't normally reach here (caller already checks),
            # but belt-and-suspenders: fall through to OUT_OF_LIBRARY.
            self._emit_error(
                ERR_OUT_OF_LIBRARY, label="OUT_OF_LIBRARY",
                extra=f"heard=\"{text}\"",
            )
            return None
        self.execute_robot_command(robot_cmd)
        print(f"   🎯 robot command: {robot_cmd} — "
              f"{ROBOT_COMMANDS[robot_cmd]['description']}")
        self.print_state()
        print("=" * 60)
        print()
        try:
            self.save_command(text, robot_cmd)
            print(f"[voiceCommandToRobot] Appended to: {self.output_file}")
        except OSError as e:
            print(f"[voiceCommandToRobot] Write failed: {e}", file=sys.stderr)
        # Long-term memory: persist to JSON so command survives restart
        self._record_to_memory(text, robot_cmd)
        return robot_cmd

    # -----------------------------------------------------------------
    # Long-term memory (JSON)
    # -----------------------------------------------------------------
    def load_memory(self) -> List[Dict]:
        """Load persisted command history from disk. Returns [] if absent."""
        if not os.path.exists(self.memory_path):
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[voiceCommandToRobot] Could not read memory "
                  f"({self.memory_path}): {exc}", file=sys.stderr)
            return []
        # Schema: {"version": int, "entries": [{ts, text, cmd, x, y, heading}]}
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return list(data["entries"])
        if isinstance(data, list):
            return data  # back-compat: older-style bare list
        return []

    def save_memory(self) -> None:
        """Atomically persist memory to disk."""
        payload = {
            "version": MEMORY_VERSION,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "total_entries": len(self.memory),
            "entries": self.memory,
        }
        tmp_path = f"{self.memory_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.memory_path)
        except OSError as exc:
            print(f"[voiceCommandToRobot] Could not write memory "
                  f"({self.memory_path}): {exc}", file=sys.stderr)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _record_to_memory(self, text: str, robot_cmd: Optional[str]) -> None:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "text": text,
            "robot_command": robot_cmd or "UNKNOWN",
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading_deg": round(self.heading_deg, 2),
        }
        self.memory.append(entry)
        self.save_memory()

    def print_memory_summary(self, limit: int = 10) -> None:
        print("=" * 60)
        print(f"  🧠 Long-term memory: {len(self.memory)} entries "
              f"({self.memory_path})")
        print("=" * 60)
        if not self.memory:
            print("  (empty)")
        else:
            for entry in self.memory[-limit:]:
                print(f"    [{entry['timestamp']}] "
                      f"{entry['robot_command']:8s}  "
                      f"\"{entry['text']}\"  "
                      f"(x={entry['x']} y={entry['y']} "
                      f"h={entry['heading_deg']}°)")
            if len(self.memory) > limit:
                print(f"    ... ({len(self.memory) - limit} older entries omitted)")
        print("=" * 60)

    def clear_memory(self) -> None:
        self.memory = []
        self.save_memory()
        print(f"[voiceCommandToRobot] Memory cleared ({self.memory_path}).")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def list_devices() -> None:
        pa, _ = _require_audio()
        inst = pa.PyAudio()
        print("[voiceCommandToRobot] Available input devices:")
        for i in range(inst.get_device_count()):
            info = inst.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"  [{i}] {info.get('name', 'Unknown')} "
                      f"(sr={int(info.get('defaultSampleRate', 0))})")
        inst.terminate()

    def cleanup(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.p is not None:
            try:
                self.p.terminate()
            except Exception:
                pass
        print("\n[voiceCommandToRobot] Resources cleaned up.")

    # -----------------------------------------------------------------
    # Built-in test scenarios
    # -----------------------------------------------------------------
    def run_test_scenarios(self) -> None:
        """Execute the built-in test scenarios via text-feeding."""
        print("=" * 60)
        print("  🧪 ROBOT COMMAND TEST SCENARIOS")
        print("=" * 60)
        print()

        # (expected_command_key | label_for_error, phrase)
        #
        # When expected is "ERR_TOO_QUIET", the phrase is fed as if the
        # user spoke too quietly and the system should emit
        # "I cannot hear you, please speak louder". This scenario also
        # exercises the volume check in record_audio via a tiny
        # synthetic WAV.
        scenarios = [
            ("FORWARD",  "Move forward"),
            ("FORWARD",  "前进"),          # Chinese alias
            ("LEFT",     "Turn left"),
            ("LEFT",     "向左转"),        # Chinese alias
            ("RIGHT",    "Turn right"),
            ("BACK",     "Move backward"),
            ("TURN180",  "Turn around 180 degrees"),
            ("TURN180",  "掉头"),         # Chinese alias
            ("SLOW_DOWN", "Slow down"),
            ("SLOW_DOWN", "减速"),        # Chinese alias
            ("SPEED_UP", "Speed up"),
            ("RETURN",   "Return to start point"),
            ("STOP",     "Stop / halt"),
            # ---- Error-path scenarios ----
            ("ERR_OUT_OF_LIBRARY",
             "Please bring me a cup of coffee"),
            ("ERR_OUT_OF_LIBRARY",  "请帮我拿一杯咖啡"),
            ("ERR_TOO_QUIET",       "(quiet whisper / low volume)"),
        ]

        print(f"Start state: (x={self.x:.1f}, y={self.y:.1f}, "
              f"heading={self.heading_deg:.0f}°, speed={self.speed_multiplier:.2f}x)")
        print()

        for i, (expected, phrase) in enumerate(scenarios, start=1):
            print(f"── Scenario {i}/{len(scenarios)}: {phrase}")
            print(f"   Expected   : {expected}")
            if expected.startswith("ERR_"):
                # Error-path scenarios do not call execute_robot_command,
                # but they DO reach _emit_error and write to output file.
                if expected == "ERR_TOO_QUIET":
                    # Synthesise a near-silent WAV and run the
                    # "peak_rms too low" check end-to-end.
                    peak_rms = 100
                    assert peak_rms < self.PEAK_RMS_TOO_QUIET, (
                        "sanity: silent WAV must be below the volume threshold"
                    )
                    self._emit_error(
                        ERR_TOO_QUIET, label="TOO_QUIET",
                        extra=f"peak_rms={peak_rms}",
                    )
                    # Record this as a scenario completion.
                    print(f"   Emitted  : \"{ERR_TOO_QUIET}\" ✅")
                elif expected == "ERR_OUT_OF_LIBRARY":
                    # Feed the unrecognisable phrase through the normal
                    # text-command path; the parser returns None and
                    # feed_text_command emits ERR_OUT_OF_LIBRARY.
                    parsed = self.parse_robot_command(phrase)
                    assert parsed is None, (
                        f"Parser bug: expected None, got {parsed}"
                    )
                    self._emit_error(
                        ERR_OUT_OF_LIBRARY, label="OUT_OF_LIBRARY",
                        extra=f"heard=\"{phrase}\"",
                    )
                    print(f"   Emitted  : \"{ERR_OUT_OF_LIBRARY}\" ✅")
                else:
                    assert False, f"Unknown error label: {expected}"
            else:
                parsed = self.parse_robot_command(phrase)
                assert parsed == expected, (
                    f"Parser bug: expected {expected}, got {parsed}"
                )
                print(f"   Parsed as: {parsed} ✅")
                self._print_and_save(phrase)

        print()
        print("=" * 60)
        print(f"✅ All {len(scenarios)} test scenarios passed.")
        print("=" * 60)
        print()
        self.print_state("Final")
        print()
        print("=" * 60)
        print(f"  📋 Commands executed ({len(self.history)})")
        for j, (c, x, y, h) in enumerate(self.history, start=1):
            print(f"    [{j}] {c:14s} (x={x:.1f} y={y:.1f} heading={h:.0f}°)")
        print("=" * 60)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="voiceCommandToRobot — turn a spoken (or text) phrase into a robot command."
    )
    parser.add_argument(
        "--model", default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: tiny).",
    )
    parser.add_argument(
        "--output", default="voice_commands.txt",
        help="Text file to append transcriptions/errors (default: voice_commands.txt).",
    )
    parser.add_argument(
        "--language", default=None,
        help="Language hint for Whisper (e.g., en, zh). Auto-detect if unset.",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (see --list-devices).",
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Keep listening for commands until Ctrl-C (live mic mode).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run the built-in test scenarios. No microphone required.",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Print remembered command history (long-term memory) and exit.",
    )
    parser.add_argument(
        "--clear-memory", action="store_true",
        help="Delete the persisted long-term memory file and exit.",
    )
    parser.add_argument(
        "--text", default=None,
        help="Bypass the microphone: classify the given text phrase directly "
             "(e.g. --text \"move forward\"). Useful for agents/tests.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the outcome as a single JSON object on stdout. "
             "All human-friendly terminal chatter is suppressed. "
             "Intended for programmatic/agent consumption.",
    )
    parser.add_argument(
        "--volume-threshold", type=int, default=None,
        help="Override the RMS volume threshold (int). Default: 500.",
    )
    return parser.parse_args()


def _suppress_terminal_output() -> None:
    """Redirect stdout to stderr temporarily so --json produces clean stdout JSON.

    When --json is used, the skill writes any operational messages to stderr
    and emits a single JSON object on stdout. This makes the output
    machine-parseable by an agent.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    return real_stdout


def run(args: Optional[argparse.Namespace] = None) -> Dict:
    """Agent-friendly entry point.

    This is the stable public surface: an agent can either:
      * invoke the CLI  (python3 voiceCommandToRobot.py --json --text "move forward")
      * or call this   (from voiceCommandToRobot import run; result = run())
      * or import     (from voiceCommandToRobot import voiceCommandToRobot
                       and use feed_text_command / get_command directly).

    Returns a dict with schema:
        {
            "skill": "voiceCommandToRobot",
            "ok": bool,              # True on clean completion
            "mode": str,             # "test" | "list_devices" | "history" | ...
            "robot_command": str | null,  # Canonical key — or null when none
            "heard": str,            # Transcribed / fed text
            "message": str,          # Human message — often the error or confirmation
            "x": float, "y": float, "heading_deg": float,
            "speed_multiplier": float,
            "output_file": str,
            "memory_file": str,
            "memory_entries": int,
            "error": str | null,     # Error label — null on success
            "raw_input": str | null, # What the user actually said / typed
        }
    """
    if args is None:
        args = parse_args()

    json_mode = bool(args.json)
    if json_mode:
        _suppress_terminal_output()

    # -------- Quick exits (no Whisper, no mic) --------
    if args.list_devices:
        voiceCommandToRobot.list_devices()
        return _json_skeleton(mode="list_devices", ok=True, heard="",
                              message="See stderr for the device list.")

    _mem_path = _default_memory_path(args.output)

    if args.clear_memory:
        removed = False
        error = None
        try:
            if os.path.exists(_mem_path):
                os.remove(_mem_path)
                removed = True
        except OSError as exc:
            error = str(exc)
        msg = ("Memory cleared" if removed and not error else
               "No memory file to clear" if not error else f"Failed: {error}")
        if error is None:
            print(f"[voiceCommandToRobot] {msg}: {_mem_path}")
        else:
            print(f"[voiceCommandToRobot] {msg}", file=sys.stderr)
        return _json_skeleton(mode="clear_memory", ok=error is None,
                              heard="", message=msg, error=error)

    if args.history:
        entries = []
        error = None
        try:
            if os.path.exists(_mem_path):
                with open(_mem_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("entries", data) if isinstance(data, dict) else data
            msg = f"{len(entries)} entries"
            if not json_mode:
                for e in entries:
                    print(f"  [{e.get('timestamp')}] "
                          f"{e.get('robot_command')}  \"{e.get('text')}\"")
        except (OSError, json.JSONDecodeError) as exc:
            error = str(exc)
            msg = f"Could not read memory: {exc}"
        return {
            **_json_skeleton(mode="history", ok=error is None,
                             heard="", message=msg, error=error),
            "memory_file": _mem_path,
            "memory_entries": len(entries),
            "entries": list(entries),
        }

    # -------- Main skill invocation --------
    # Decide which libraries are actually required — so an agent can
    # invoke offline modes without the full audio stack installed.
    wants_live_mic = args.continuous or (args.text is None and not args.test)
    needs_whisper = wants_live_mic  # transcribe only in live modes
    needs_mic = wants_live_mic      # mic only in live modes

    try:
        skill = voiceCommandToRobot(
            model_name=args.model,
            output_file=args.output,
            language=args.language,
            device_index=args.device,
            need_mic=needs_mic,
            need_whisper=needs_whisper,
        )
    except RuntimeError as exc:
        # Missing libraries, microphone not available, output file unwritable, etc.
        msg = str(exc)
        print(f"[voiceCommandToRobot] Cannot initialise: {msg}", file=sys.stderr)
        return _json_skeleton(mode="init_error", ok=False, heard="",
                              message=msg, error="INIT_FAILED")

    if args.volume_threshold is not None:
        skill.PEAK_RMS_TOO_QUIET = int(args.volume_threshold)

    try:
        if args.test:
            skill.run_test_scenarios()
            return _json_skeleton(mode="test", ok=True, heard="",
                                  message="Test scenarios ran successfully.",
                                  x=skill.x, y=skill.y,
                                  heading_deg=skill.heading_deg,
                                  speed_multiplier=skill.speed_multiplier,
                                  memory_entries=len(skill.memory),
                                  memory_file=skill.memory_path,
                                  output_file=skill.output_file)

        if args.continuous:
            # Continuous live mode: emit one JSON object per recognised utterance
            # by printing it on stdout. Because stdout is redirected to stderr
            # under --json, we restore it briefly for each result line.
            real_stdout = sys.__stdout__ if json_mode else sys.stdout
            n = 0
            try:
                while True:
                    skill.get_command()
                    n += 1
                    if json_mode:
                        evt = _json_skeleton(
                            mode="continuous", ok=True,
                            heard="(see terminal log)", message="processed",
                            x=skill.x, y=skill.y,
                            heading_deg=skill.heading_deg,
                            speed_multiplier=skill.speed_multiplier,
                            memory_entries=len(skill.memory),
                            memory_file=skill.memory_path,
                            output_file=skill.output_file,
                        )
                        print(json.dumps(evt, ensure_ascii=False),
                              file=real_stdout, flush=True)
            except KeyboardInterrupt:
                pass
            return _json_skeleton(mode="continuous", ok=True, heard="",
                                  message=f"Processed {n} utterance(s).",
                                  x=skill.x, y=skill.y,
                                  heading_deg=skill.heading_deg,
                                  speed_multiplier=skill.speed_multiplier,
                                  memory_entries=len(skill.memory),
                                  memory_file=skill.memory_path,
                                  output_file=skill.output_file)

        if args.text is not None:
            # Agent-friendly text injection — the caller already transcribed it.
            phrase = args.text.strip()
            cmd = skill.parse_robot_command(phrase)
            if cmd is None:
                skill._emit_error(ERR_OUT_OF_LIBRARY, label="OUT_OF_LIBRARY",
                                  extra=f"heard=\"{phrase}\"")
            else:
                skill._print_and_save(phrase)
            return _json_skeleton(mode="text", ok=True, heard=phrase,
                                  message=(ROBOT_COMMANDS[cmd]["description"]
                                           if cmd else ERR_OUT_OF_LIBRARY),
                                  robot_command=cmd,
                                  x=skill.x, y=skill.y,
                                  heading_deg=skill.heading_deg,
                                  speed_multiplier=skill.speed_multiplier,
                                  memory_entries=len(skill.memory),
                                  memory_file=skill.memory_path,
                                  output_file=skill.output_file)

        # Default: one live utterance via the mic
        text = skill.get_command()
        parsed = skill.parse_robot_command(text) if text else None
        return _json_skeleton(mode="live", ok=bool(text), heard=text,
                              message=(ROBOT_COMMANDS[parsed]["description"]
                                       if parsed else ERR_OUT_OF_LIBRARY),
                              robot_command=parsed,
                              x=skill.x, y=skill.y,
                              heading_deg=skill.heading_deg,
                              speed_multiplier=skill.speed_multiplier,
                              memory_entries=len(skill.memory),
                              memory_file=skill.memory_path,
                              output_file=skill.output_file)
    finally:
        skill.cleanup()


def _json_skeleton(*, mode: str, ok: bool, heard: str, message: str,
                   error: Optional[str] = None,
                   robot_command: Optional[str] = None,
                   x: float = 0.0, y: float = 0.0, heading_deg: float = 0.0,
                   speed_multiplier: float = 1.0,
                   memory_entries: int = 0,
                   memory_file: str = "",
                   output_file: str = "") -> Dict:
    """Return a stable dict used across every CLI branch."""
    return {
        "skill": "voiceCommandToRobot",
        "version": 1,
        "mode": mode,
        "ok": bool(ok),
        "robot_command": robot_command,
        "heard": heard,
        "message": message,
        "x": float(x), "y": float(y),
        "heading_deg": float(heading_deg),
        "speed_multiplier": float(speed_multiplier),
        "output_file": output_file,
        "memory_file": memory_file,
        "memory_entries": int(memory_entries),
        "error": error,
    }


if __name__ == "__main__":
    _args = parse_args()
    _result = run(_args)
    if _args.json:
        # Emit a single JSON line on stdout for easy agent parsing.
        sys.stdout = sys.__stdout__
        print(json.dumps(_result, ensure_ascii=False, indent=2))
    sys.exit(0 if _result.get("ok") else 1)
