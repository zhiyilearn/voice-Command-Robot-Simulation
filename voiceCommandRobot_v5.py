#!/usr/bin/env python3
"""
Voice Controlled Robot Car
---------------------------
- Listens to microphone via arecord (raw PCM)
- Transcribes with SenseVoice Small (FunASR)
- Matches Chinese voice commands to robot car API
- Sends HTTP commands to car robot via hotspot WiFi

API:
  GET http://{robot_ip}/api/control?action={action}&speed={speed}[&time={seconds}]

Actions: up, down, left, right, stop, grab, release

Usage:
  python3 voiceCommandRobot.py --device 2 --robot-ip 192.168.4.1 --quiet
  python3 voiceCommandRobot.py --list-alsa-devices
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
import urllib.request
import urllib.error

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

SAMPLE_WIDTH = 2  # bytes per sample (S16_LE = 16-bit signed little-endian)
SAMPLE_RATE = 16000
PCM_FILE = "/tmp/voicerobot_audio.raw"

# Voice command semantic understanding
# Robot API actions: up, down, left, right, stop, grab, release

# Direction vocabulary (Chinese + English)
DIRECTION_WORDS = {
    "forward": ["前", "前进", "向前", "往前", "进", "直走", "直行", "前行", "向前走", "往前走",
                "forward", "up", "go", "move", "walk", "ahead"],
    "backward": ["后", "后退", "向后", "往后", "退", "倒", "倒车", "倒退", "向后退", "往后退",
                 "backward", "back", "down", "reverse"],
    "left": ["左", "左转", "向左", "往左", "左拐", "左弯", "向左转", "往左转", "左转弯",
             "left", "turn left"],
    "right": ["右", "右转", "向右", "往右", "右拐", "右弯", "向右转", "往右转", "右转弯",
              "right", "turn right"],
}

# Action vocabulary
ACTION_WORDS = {
    "move": ["走", "行", "动", "移", "前进", "进", "go", "move", "walk", "proceed"],
    "turn": ["转", "拐", "弯", "转动", "turn", "rotate"],
    "stop": ["停", "止", "住", "别动", "停下", "停止", "stop", "halt", "freeze"],
    "grab": ["抓", "拿", "夹", "取", "捉", "拾", "grab", "pick", "take", "grasp", "catch"],
    "release": ["放", "松", "开", "丢", "扔", "release", "drop", "let go", "put down"],
}

# Priority actions (override everything)
PRIORITY_ACTIONS = {
    "stop": ["停", "停止", "停下", "停住", "别动", "站住", "stop", "halt"],
    "grab": ["抓", "拿", "夹", "grab", "pick"],
    "release": ["放", "松开", "放下", "release", "drop"],
}

# Camera commands (separate from movement)
CAMERA_COMMANDS = {
    "photo": ["拍照", "照相", "拍个照", "拍张照", "拍照片", "拍一张", "snapshot", "photo", "take picture"],
    "record_start": ["录像", "开始录像", "录视频", "拍视频", "开始拍视频", "录制", "record", "start recording"],
    "record_stop": ["停止录像", "结束录像", "停录像", "不录了", "stop recording", "stop video"],
    "camera_on": ["打开摄像头", "开摄像头", "显示画面", "显示图像", "打开相机", "开相机", "camera on", "show camera"],
    "camera_off": ["关闭摄像头", "关摄像头", "关闭相机", "关相机", "隐藏画面", "camera off", "hide camera"],
}

# Wake words to activate voice command listening
WAKE_WORDS = [
    "机器人", "小车", "助手", "小助手",
    "wake up", "start", "hello", "hi",
    "你好", "在吗", "开始了",
]

# Sleep words to deactivate voice command listening
SLEEP_WORDS = [
    "睡觉", "休眠", "休息", "暂停", "闭嘴",
    "sleep", "stop listening", "pause", "over",
    "再见", "拜拜", "退下",
]

# Chinese label for each action (for display)
ACTION_LABEL = {
    "up": "前进",
    "down": "后退",
    "left": "左转",
    "right": "右转",
    "stop": "停止",
    "grab": "抓取",
    "release": "释放",
    "camera_on": "打开摄像头",
    "camera_off": "关闭摄像头",
    "photo": "拍照",
    "record_start": "开始录像",
    "record_stop": "停止录像",
}

# SenseVoice special tokens to strip from output
ASR_TOKENS = [
    "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
    "<|NEUTRAL|>", "<|Happy|>", "<|Sad|>", "<|Angry|>",
    "<|Speech|>", "<|woitn|>", "<|EMO_UNKNOWN|>",
    "<|withitn|>", "<|noitn|>", "<|itn|>", "<|nospeech|>",
    "<|Event_UNK|>", "<|UNKNOWN|>", "<|Music|>", "<|Noise|>",
]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def understand_command(text: str) -> str:
    """Semantically understand transcribed text and return robot action.
    
    Logic:
    1. Check for camera commands first
    2. Check for priority actions (stop, grab, release)
    3. Detect direction and action type
    4. Combine: direction (left/right) = turn, direction (forward/backward) = move
    5. Return empty string if unclear/no action detected
    """
    if not text:
        return ""
    
    text = text.lower().strip()
    text_clean = text.replace(" ", "").replace(",", "").replace("。", "").replace("，", "")
    
    # 0. Camera commands (check first, distinct from movement)
    for action, words in CAMERA_COMMANDS.items():
        for w in words:
            if w.lower() in text or w.lower() in text_clean:
                return action
    
    # 1. Priority actions (stop, grab, release) — these override everything
    for action, words in PRIORITY_ACTIONS.items():
        for w in words:
            if w.lower() in text or w.lower() in text_clean:
                return action
    
    # 2. Analyze direction and action type
    detected_direction = None
    detected_action_type = None
    
    # Check for direction words
    direction_scores = {}
    for dir_type, words in DIRECTION_WORDS.items():
        score = 0
        for w in words:
            w_lower = w.lower()
            if w_lower in text:
                score += len(w)
            elif w_lower in text_clean:
                score += len(w)
        if score > 0:
            direction_scores[dir_type] = score
    
    # Check for action words
    action_scores = {}
    for act_type, words in ACTION_WORDS.items():
        score = 0
        for w in words:
            w_lower = w.lower()
            if w_lower in text:
                score += len(w)
            elif w_lower in text_clean:
                score += len(w)
        if score > 0:
            action_scores[act_type] = score
    
    # Determine best direction and action
    if direction_scores:
        detected_direction = max(direction_scores.keys(), 
                                  key=lambda k: direction_scores[k])
    
    if action_scores:
        detected_action_type = max(action_scores.keys(),
                                    key=lambda k: action_scores[k])
    
    # 3. Combine direction + action to determine robot command
    # IMPORTANT: left/right direction always means turn, regardless of action word
    if detected_direction == "left":
        return "left"
    if detected_direction == "right":
        return "right"
    
    # Forward/backward direction means movement
    if detected_direction == "forward":
        return "up"
    if detected_direction == "backward":
        return "down"
    
    # No direction detected — check action type only
    if detected_action_type == "turn":
        # "转" without direction is unclear
        return ""
    if detected_action_type == "move":
        # "走" without direction — default to forward
        return "up"
    
    # No clear action detected
    return ""


# Keep old match_action for backwards compatibility with test mode
def match_action(text: str) -> str:
    """Legacy keyword matching (wrapper for understand_command)."""
    return understand_command(text)


def is_wake_word(text: str) -> bool:
    """Check if text contains a wake word."""
    if not text:
        return False
    text_clean = text.lower().strip().replace(" ", "")
    for word in WAKE_WORDS:
        if word.lower().replace(" ", "") in text_clean:
            return True
    return False


def is_sleep_word(text: str) -> bool:
    """Check if text contains a sleep word."""
    if not text:
        return False
    text_clean = text.lower().strip().replace(" ", "")
    for word in SLEEP_WORDS:
        if word.lower().replace(" ", "") in text_clean:
            return True
    return False


# Chinese numeral mapping
_CHINESE_NUMBERS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
    "千": 1000, "万": 10000,
}


def _parse_chinese_number(text: str) -> float:
    """Parse Chinese number string to float.
    Handles: 一, 二, 三, 十, 二十, 二十五, 一百, 两百, 三点五, etc.
    """
    text = text.strip()
    if not text:
        return 0

    # Try direct float first (Arabic numerals)
    try:
        return float(text)
    except ValueError:
        pass

    # Handle decimal like "三点五"
    if "点" in text or "." in text:
        parts = text.replace("点", ".").split(".")
        if len(parts) == 2:
            int_part = _parse_chinese_integer(parts[0])
            dec_part = _parse_chinese_decimal(parts[1])
            if int_part is not None and dec_part is not None:
                return int_part + dec_part
        return 0

    result = _parse_chinese_integer(text)
    return result if result is not None else 0


def _parse_chinese_integer(text: str):
    """Parse Chinese integer (up to 万)."""
    if not text:
        return 0

    # Pure Arabic numerals
    if text.isdigit():
        return int(text)

    total = 0
    current = 0
    for ch in text:
        if ch in _CHINESE_NUMBERS:
            val = _CHINESE_NUMBERS[ch]
            if val >= 10:
                if current == 0:
                    current = 1
                total += current * val
                current = 0
            else:
                current = val
    total += current
    return total


def _parse_chinese_decimal(text: str):
    """Parse Chinese decimal part like '三五' -> 0.35."""
    if not text:
        return 0
    result = 0.0
    divisor = 10.0
    for ch in text:
        if ch in _CHINESE_NUMBERS and _CHINESE_NUMBERS[ch] < 10:
            result += _CHINESE_NUMBERS[ch] / divisor
            divisor *= 10.0
        elif ch.isdigit():
            result += int(ch) / divisor
            divisor *= 10.0
    return result


def extract_parameters(text: str, base_speed: int = 50) -> dict:
    """Extract command parameters from text.
    Returns dict with: duration (seconds), speed (0-100), distance (meters), angle (degrees)
    """
    params = {
        "duration": None,
        "speed": base_speed,
        "distance": None,
        "angle": None,
    }
    text = text.strip()
    if not text:
        return params

    import re

    # --- Speed modifiers ---
    speed_up_words = ["快点", "快一点", "快些", "加速", "加快", "高速", "快速", "最快", "全速"]
    slow_down_words = ["慢点", "慢一点", "慢些", "减速", "放慢", "低速", "慢速", "慢慢", "缓慢"]
    for w in speed_up_words:
        if w in text:
            params["speed"] = min(100, base_speed + 30)
            break
    for w in slow_down_words:
        if w in text:
            params["speed"] = max(10, base_speed - 30)
            break

    # --- Distance (米/公尺) ---
    dist_patterns = [
        r"([\d\.]+)\s*(米|公尺|m)",
        r"([零一二两三四五六七八九十百千点\.]+)\s*(米|公尺)",
    ]
    for pat in dist_patterns:
        m = re.search(pat, text)
        if m:
            dist = _parse_chinese_number(m.group(1))
            if dist > 0:
                params["distance"] = dist
                break

    # --- Duration (秒/秒钟) ---
    time_patterns = [
        r"([\d\.]+)\s*(秒|秒钟|s)\b",
        r"([零一二两三四五六七八九十百千点\.]+)\s*(秒|秒钟)",
    ]
    for pat in time_patterns:
        m = re.search(pat, text)
        if m:
            secs = _parse_chinese_number(m.group(1))
            if secs > 0:
                params["duration"] = secs
                break

    # --- Angle (度) for turns ---
    angle_patterns = [
        r"([\d\.]+)\s*(度|°)",
        r"([零一二两三四五六七八九十百千点\.]+)\s*(度)",
    ]
    for pat in angle_patterns:
        m = re.search(pat, text)
        if m:
            angle = _parse_chinese_number(m.group(1))
            if angle > 0:
                params["angle"] = angle
                break

    # Special: "左转/右转 九十度/45度"
    turn_patterns = [
        r"(左转|右转|向左转|向右转|往左|往右)\s*([零一二两三四五六七八九十百千点\d\.]+)\s*度",
    ]
    for pat in turn_patterns:
        m = re.search(pat, text)
        if m:
            angle = _parse_chinese_number(m.group(2))
            if angle > 0:
                params["angle"] = angle
                break

    return params


def compute_action_duration(params: dict, action: str, default_duration: float = None,
                            distance_factor: float = 0.5) -> float:
    """Compute the actual duration to send to the API.
    If distance is specified for movement commands, estimate time from distance.
    distance_factor: meters per second at speed=50 (adjust for your robot).
    """
    # If duration explicitly given, use it
    if params["duration"] is not None:
        return params["duration"]

    # If distance given and it's a movement command, estimate time
    movement_actions = {"up", "down"}
    if params["distance"] is not None and action in movement_actions:
        speed_factor = params["speed"] / 50.0
        meters_per_second = distance_factor * speed_factor
        if meters_per_second > 0:
            return params["distance"] / meters_per_second

    return default_duration


def compute_turn_duration(params: dict, action: str, base_speed: int = 50,
                          turn_factor: float = 1.5,
                          default_turn_duration: float = None) -> float:
    """Compute turn duration from angle if specified.
    
    Args:
        params: Parameter dict with 'angle' key
        action: 'left' or 'right'
        base_speed: Speed value (0-100)
        turn_factor: Seconds to turn 90° at speed=50
        default_turn_duration: Default duration if no angle specified (seconds)
    
    Returns:
        Duration in seconds, or None if no angle and no default
    """
    turn_actions = {"left", "right"}
    if action not in turn_actions:
        return None
    
    # If angle specified, compute from angle
    if params["angle"] is not None:
        speed_factor = base_speed / 50.0
        return (params["angle"] / 90.0) * turn_factor / speed_factor
    
    # No angle specified - use default duration for simple turn
    if default_turn_duration is not None:
        return default_turn_duration
    
    return None


def clean_asr_text(text: str) -> str:
    """Strip SenseVoice special tokens from ASR output."""
    if not text:
        return ""
    for token in ASR_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def correct_asr_errors(text: str) -> str:
    """Post-process ASR output to correct common misrecognitions.
    These are domain-specific corrections for robot voice commands.
    """
    if not text:
        return text

    # Common ASR errors in voice command context
    # ONLY correct obvious typos that don't make sense as valid commands
    corrections = {
        # Stop misrecognized (obvious typos only)
        "后推": "停止",
        "后题": "停止",
        "后腿": "停止",
        # Forward misrecognized
        "前径": "前进",
        "前尽": "前进",
        "前近": "前进",
        # Turn misrecognized
        "左赚": "左转",
        "右赚": "右转",
        # Other common errors
        "往左": "左转",
        "往右": "右转",
        "抓去": "抓取",
        "抓起": "抓取",
        "释饭": "释放",
        "释方": "释放",
    }

    # Apply corrections only if the text is short (likely a command, not a sentence)
    if len(text) <= 15:
        for wrong, correct in corrections.items():
            if wrong in text and correct not in text:
                text = text.replace(wrong, correct)

    return text


def is_hallucination(text: str) -> bool:
    """Heuristic check for ASR garbage output."""
    t = text.strip()
    if not t:
        return True
    if len(t) >= 5:
        counts = {}
        for c in t:
            counts[c] = counts.get(c, 0) + 1
        if max(counts.values()) >= len(t) * 0.8:
            return True
    for pattern in ["字幕", "制作人", "索兰娅", "zither", "harp"]:
        if pattern in text.lower():
            return True
    if len(t) <= 1 and t not in "0123456789":
        return True
    return False


def remove_internal_repeats(text: str) -> str:
    """Remove repeated substrings like '前进前进' -> '前进'."""
    text = text.strip()
    if not text or len(text) < 2:
        return text
    half = len(text) // 2
    for n in range(1, half + 1):
        seg = text[:n]
        rep = seg * (len(text) // n)
        if rep == text or text.startswith(rep):
            return seg
    return text


def is_duplicate(new_text: str, old_text: str) -> bool:
    """Check if new text is a duplicate of old text.
    Only returns True for exact matches or very high-overlap cases.
    """
    new_c = new_text.strip()
    old_c = old_text.strip()
    if not new_c or not old_c:
        return False
    if new_c == old_c:
        return True
    return False


# --------------------------------------------------------------------------- #
#  Complex Command Detection and Sequencing
#  L1 Basic Primitives (5) + L2 Geometric Trajectories (5)
# --------------------------------------------------------------------------- #

COMPLEX_ACTION_LABEL = {
    "complex_forward_back_repeat": "前后往返重复",
    "complex_spin_left_right": "左右旋转360度",
    "complex_diagonal": "对角斜线",
    "complex_variable_speed": "变速前进",
    "complex_fullspeed_emergency_stop": "全速紧急刹停",
    "complex_square": "正方形轨迹",
    "complex_circle": "圆形轨迹",
    "complex_triangle": "三角形轨迹",
    "complex_figure8": "数字8轨迹",
    "complex_s_curve": "S型蜿蜒曲线",
}


def detect_complex_command(text: str) -> str:
    """Detect if text matches one of the 10 complex commands.
    Returns the complex action key, or empty string if not matched.
    """
    if not text:
        return ""

    text_clean = text.replace(" ", "").lower()

    # L1.1 向前走2米，再向后走2米，连续重复3次
    if (("向前" in text_clean or "前进" in text_clean) and
        ("向后" in text_clean or "后退" in text_clean) and
        ("重复" in text_clean or "连续" in text_clean)):
        if "米" in text_clean:
            return "complex_forward_back_repeat"

    # L1.2 原地向左转360度，再原地向右转360度
    if (("左转" in text_clean or "向左转" in text_clean) and
        ("右转" in text_clean or "向右转" in text_clean) and
        ("360" in text_clean or "三百六十" in text_clean)):
        return "complex_spin_left_right"

    # L1.3 向前走1米，同时向右平移1米，走一条对角斜线
    if (("对角" in text_clean or "斜线" in text_clean) and
        ("向前" in text_clean or "前进" in text_clean)):
        return "complex_diagonal"

    # L1.4 以每秒0.3米的速度向前走2米，再以每秒1米的速度向前走2米
    if ("速度" in text_clean and "向前" in text_clean and
        ("再" in text_clean or "然后" in text_clean)):
        return "complex_variable_speed"

    # L1.5 全速向前直行3米，到达后执行紧急刹停
    if (("全速" in text_clean or "最快" in text_clean) and
        ("紧急" in text_clean or "刹停" in text_clean or "急停" in text_clean)):
        return "complex_fullspeed_emergency_stop"

    # L2.1 走一个边长1米的正方形，每到拐角原地转90度
    if ("正方形" in text_clean and
        ("拐角" in text_clean or "转角" in text_clean or "90" in text_clean)):
        return "complex_square"
    if ("正方形" in text_clean and "边长" in text_clean):
        return "complex_square"

    # L2.2 走出一个半径0.5米的完整圆形轨迹
    if ("圆形" in text_clean or "圆圈" in text_clean or "圆" in text_clean):
        if ("半径" in text_clean or "完整" in text_clean or "一圈" in text_clean):
            return "complex_circle"

    # L2.3 走一个边长1米的等边三角形，拐角处原地旋转120度
    if ("三角形" in text_clean and
        ("等边" in text_clean or "120" in text_clean or "一百二十" in text_clean)):
        return "complex_triangle"

    # L2.4 沿着数字8的路线行走，左右两个圆圈保持对称
    if ("数字8" in text_clean or "八字" in text_clean or "8字" in text_clean or
        "数字八" in text_clean or "8" in text_clean and "对称" in text_clean):
        return "complex_figure8"

    # L2.5 走出一个S型的蜿蜒曲线，包含连续两段反向圆弧
    if (("s型" in text_clean or "s形" in text_clean or "s曲线" in text_clean or
         "蜿蜒" in text_clean) and
        ("圆弧" in text_clean or "反向" in text_clean or "曲线" in text_clean)):
        return "complex_s_curve"

    return ""


def _extract_number_from_text(text: str, unit: str = "") -> float:
    """Extract a number (Arabic or Chinese) followed by optional unit."""
    import re

    if unit:
        pattern = r"([\d\.]+)\s*" + re.escape(unit)
        m = re.search(pattern, text)
        if m:
            return float(m.group(1))

        pattern_cn = r"([零一二两三四五六七八九十百千点\.]+)\s*" + re.escape(unit)
        m = re.search(pattern_cn, text)
        if m:
            return _parse_chinese_number(m.group(1))

    # Generic: find first number
    m = re.search(r"([\d\.]+)", text)
    if m:
        return float(m.group(1))

    m = re.search(r"([零一二两三四五六七八九十百千点\.]+)", text)
    if m:
        return _parse_chinese_number(m.group(1))

    return 0.0


def generate_complex_sequence(action: str, text: str,
                              base_speed: int = 50,
                              distance_factor: float = 0.5,
                              turn_factor: float = 1.5) -> list:
    """Generate a list of basic command steps for a complex action.

    Each step is a dict: {action, speed, duration, label, delay_after}

    Approximations:
    - Diagonal: forward + right turn + forward (approximates diagonal)
    - Circle: continuous forward + slight turn (approximated by short steps)
    - Figure-8: two circles in opposite directions
    - S-curve: two half-circles in opposite directions
    """
    steps = []
    text_clean = text.replace(" ", "")

    def move_duration(distance_m, speed=base_speed):
        speed_factor = speed / 50.0
        mps = distance_factor * speed_factor
        if mps > 0:
            return distance_m / mps
        return distance_m / distance_factor

    def turn_duration(degrees, speed=base_speed):
        speed_factor = speed / 50.0
        return (degrees / 90.0) * turn_factor / speed_factor

    if action == "complex_forward_back_repeat":
        # Forward 2m, backward 2m, repeat 3 times
        fwd_dist = _extract_number_from_text(text_clean, "米") or 2.0
        if fwd_dist <= 0:
            fwd_dist = 2.0
        repeat_count = 3
        m = __import__('re').search(r'重复.*?(\d+)次', text_clean)
        if m:
            repeat_count = int(m.group(1))
        else:
            m = __import__('re').search(r'(\d+)次', text_clean)
            if m:
                repeat_count = int(m.group(1))

        fwd_dur = move_duration(fwd_dist)
        back_dur = move_duration(fwd_dist)

        for i in range(repeat_count):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"前进 {fwd_dist}m (第{i+1}次)",
                         "delay_after": 0.1})
            steps.append({"action": "stop", "speed": base_speed,
                         "duration": None,
                         "label": "停止",
                         "delay_after": 0.2})
            steps.append({"action": "down", "speed": base_speed,
                         "duration": back_dur,
                         "label": f"后退 {fwd_dist}m (第{i+1}次)",
                         "delay_after": 0.1})
            steps.append({"action": "stop", "speed": base_speed,
                         "duration": None,
                         "label": "停止",
                         "delay_after": 0.2})

    elif action == "complex_spin_left_right":
        # Spin left 360°, then spin right 360°
        left_dur = turn_duration(360)
        right_dur = turn_duration(360)
        steps.append({"action": "left", "speed": base_speed,
                     "duration": left_dur,
                     "label": "左转 360°",
                     "delay_after": 0.1})
        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "停止",
                     "delay_after": 0.2})
        steps.append({"action": "right", "speed": base_speed,
                     "duration": right_dur,
                     "label": "右转 360°",
                     "delay_after": 0.1})
        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "停止",
                     "delay_after": 0.0})

    elif action == "complex_diagonal":
        # Diagonal: approximate with forward + slight right turn + forward
        dist = _extract_number_from_text(text_clean, "米") or 1.0
        if dist <= 0:
            dist = 1.0
        # Approximate diagonal: forward 0.707x distance, turn slightly, forward 0.707x
        half_dist = dist * 0.707
        fwd_dur = move_duration(half_dist)
        small_turn_dur = turn_duration(15)  # slight angle

        steps.append({"action": "up", "speed": base_speed,
                     "duration": fwd_dur,
                     "label": f"前进 {half_dist:.1f}m",
                     "delay_after": 0.1})
        steps.append({"action": "right", "speed": base_speed,
                     "duration": small_turn_dur,
                     "label": "微调向右",
                     "delay_after": 0.1})
        steps.append({"action": "up", "speed": base_speed,
                     "duration": fwd_dur,
                     "label": f"前进 {half_dist:.1f}m",
                     "delay_after": 0.1})
        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "停止",
                     "delay_after": 0.0})

    elif action == "complex_variable_speed":
        # Two segments at different speeds
        # Parse: first speed, second speed
        import re
        speed_matches = re.findall(r'每秒([\d\.]+)米', text)
        dist_matches = re.findall(r'走([\d\.]+)米', text)
        if not dist_matches:
            dist_matches = re.findall(r'([\d\.]+)米', text)

        speeds = [float(s) for s in speed_matches] if speed_matches else [0.3, 1.0]
        dists = [float(d) for d in dist_matches[:2]] if dist_matches else [2.0, 2.0]
        while len(speeds) < 2:
            speeds.append(speeds[-1] if speeds else 0.5)
        while len(dists) < 2:
            dists.append(dists[-1] if dists else 2.0)

        for i in range(2):
            speed_mps = speeds[i]
            distance = dists[i]
            # Convert m/s to speed value (assuming 0.5 m/s at speed=50)
            speed_val = int((speed_mps / distance_factor) * 50.0)
            speed_val = max(10, min(100, speed_val))
            duration = distance / speed_mps if speed_mps > 0 else 4.0

            steps.append({"action": "up", "speed": speed_val,
                         "duration": duration,
                         "label": f"前进 {distance}m @ {speed_mps}m/s (speed={speed_val})",
                         "delay_after": 0.2})

        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "停止",
                     "delay_after": 0.0})

    elif action == "complex_fullspeed_emergency_stop":
        # Full speed forward 3m, then emergency stop
        dist = _extract_number_from_text(text_clean, "米") or 3.0
        if dist <= 0:
            dist = 3.0
        full_speed = 100
        fwd_dur = move_duration(dist, full_speed)

        steps.append({"action": "up", "speed": full_speed,
                     "duration": fwd_dur,
                     "label": f"全速前进 {dist}m (speed=100)",
                     "delay_after": 0.0})
        # Emergency stop - send stop immediately after
        steps.append({"action": "stop", "speed": full_speed,
                     "duration": None,
                     "label": "紧急刹停",
                     "delay_after": 0.0})

    elif action == "complex_square":
        # Square with 1m sides, 90° turns
        side = _extract_number_from_text(text_clean, "米") or 1.0
        if side <= 0:
            side = 1.0
        fwd_dur = move_duration(side)
        turn_dur = turn_duration(90)

        for i in range(4):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"边{i+1}: 前进 {side}m",
                         "delay_after": 0.1})
            steps.append({"action": "stop", "speed": base_speed,
                         "duration": None,
                         "label": "停止",
                         "delay_after": 0.2})
            if i < 3:
                steps.append({"action": "left", "speed": base_speed,
                             "duration": turn_dur,
                             "label": f"左转 90°",
                             "delay_after": 0.1})
                steps.append({"action": "stop", "speed": base_speed,
                             "duration": None,
                             "label": "停止",
                             "delay_after": 0.2})

    elif action == "complex_circle":
        # Circle: approximated by many small forward + turn steps
        radius = _extract_number_from_text(text_clean, "米") or 0.5
        if radius <= 0:
            radius = 0.5
        # Approximate circle with 12 segments
        num_segments = 12
        segment_angle = 360.0 / num_segments
        # Chord length for each segment
        import math
        chord_length = 2 * radius * math.sin(math.radians(segment_angle / 2))
        fwd_dur = move_duration(chord_length, base_speed)
        turn_dur = turn_duration(segment_angle, base_speed)

        for i in range(num_segments):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"段{i+1}/{num_segments}",
                         "delay_after": 0.02})
            steps.append({"action": "left", "speed": base_speed,
                         "duration": turn_dur,
                         "label": f"转 {segment_angle:.0f}°",
                         "delay_after": 0.02})

        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "停止",
                     "delay_after": 0.0})

    elif action == "complex_triangle":
        # Equilateral triangle with 1m sides, 120° turns
        side = _extract_number_from_text(text_clean, "米") or 1.0
        if side <= 0:
            side = 1.0
        fwd_dur = move_duration(side)
        turn_dur = turn_duration(120)

        for i in range(3):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"边{i+1}: 前进 {side}m",
                         "delay_after": 0.1})
            steps.append({"action": "stop", "speed": base_speed,
                         "duration": None,
                         "label": "停止",
                         "delay_after": 0.2})
            if i < 2:
                steps.append({"action": "left", "speed": base_speed,
                             "duration": turn_dur,
                             "label": f"左转 120°",
                             "delay_after": 0.1})
                steps.append({"action": "stop", "speed": base_speed,
                             "duration": None,
                             "label": "停止",
                             "delay_after": 0.2})

    elif action == "complex_figure8":
        # Figure-8: two circles in opposite directions
        radius = _extract_number_from_text(text_clean, "米") or 0.5
        if radius <= 0:
            radius = 0.5
        num_segments = 8  # per circle
        segment_angle = 360.0 / num_segments
        import math
        chord_length = 2 * radius * math.sin(math.radians(segment_angle / 2))
        fwd_dur = move_duration(chord_length, base_speed)
        left_turn_dur = turn_duration(segment_angle, base_speed)
        right_turn_dur = turn_duration(segment_angle, base_speed)

        # Left circle
        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "开始左半圈",
                     "delay_after": 0.2})
        for i in range(num_segments):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"左圈{i+1}",
                         "delay_after": 0.02})
            steps.append({"action": "left", "speed": base_speed,
                         "duration": left_turn_dur,
                         "label": "",
                         "delay_after": 0.02})

        # Right circle
        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "开始右半圈",
                     "delay_after": 0.2})
        for i in range(num_segments):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"右圈{i+1}",
                         "delay_after": 0.02})
            steps.append({"action": "right", "speed": base_speed,
                         "duration": right_turn_dur,
                         "label": "",
                         "delay_after": 0.02})

        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "完成8字",
                     "delay_after": 0.0})

    elif action == "complex_s_curve":
        # S-curve: two half-circles in opposite directions
        radius = _extract_number_from_text(text_clean, "米") or 0.5
        if radius <= 0:
            radius = 0.5
        num_segments = 6  # per half-circle
        segment_angle = 180.0 / num_segments
        import math
        chord_length = 2 * radius * math.sin(math.radians(segment_angle / 2))
        fwd_dur = move_duration(chord_length, base_speed)
        left_turn_dur = turn_duration(segment_angle, base_speed)
        right_turn_dur = turn_duration(segment_angle, base_speed)

        # First half-circle (left curve)
        for i in range(num_segments):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"左弧{i+1}",
                         "delay_after": 0.02})
            steps.append({"action": "left", "speed": base_speed,
                         "duration": left_turn_dur,
                         "label": "",
                         "delay_after": 0.02})

        # Second half-circle (right curve, reverse direction)
        for i in range(num_segments):
            steps.append({"action": "up", "speed": base_speed,
                         "duration": fwd_dur,
                         "label": f"右弧{i+1}",
                         "delay_after": 0.02})
            steps.append({"action": "right", "speed": base_speed,
                         "duration": right_turn_dur,
                         "label": "",
                         "delay_after": 0.02})

        steps.append({"action": "stop", "speed": base_speed,
                     "duration": None,
                     "label": "完成S型",
                     "delay_after": 0.0})

    return steps


def send_robot_command_async(robot_ip: str, action: str, speed: int,
                             duration: float = None, log_func=None,
                             label: str = "", detail: str = "",
                             output_file: str = None, text: str = "",
                             is_stop: bool = False):
    """Reliable async HTTP command with retry logic.
    Spawns a daemon thread that sends the request and logs the result.
    All commands get retry logic for reliability.
    """
    url = f"http://{robot_ip}/api/control?action={action}&speed={speed}"
    if duration is not None:
        time_ms = int(duration * 1000)
        url += f"&time={time_ms}"

    def _worker():
        ok = False
        last_error = None
        retries = 5 if is_stop else 4
        timeout = 3.0 if is_stop else 2.5

        # Debug: log the URL being sent
        if log_func is not None:
            log_func(f"[Robot] Sending: {url}")

        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, method='GET')
                resp = urllib.request.urlopen(req, timeout=timeout)
                resp_content = resp.read()
                ok = True
                if log_func is not None:
                    log_func(f"[Robot] Response: {resp_content.decode('utf-8', errors='ignore')[:100]}")
                break
            except urllib.error.URLError as e:
                last_error = f"URL Error: {e.reason}"
                if log_func is not None:
                    log_func(f"[Robot] Retry {attempt+1}/{retries} for {action}: {last_error}", file=sys.stderr)
                time.sleep(0.3 * (attempt + 1))
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}"
                if e.code == 404:
                    break  # Don't retry on 404
                if log_func is not None:
                    log_func(f"[Robot] Retry {attempt+1}/{retries} for {action}: {last_error}", file=sys.stderr)
                time.sleep(0.3 * (attempt + 1))
            except Exception as e:
                last_error = str(e)
                if log_func is not None:
                    log_func(f"[Robot] Retry {attempt+1}/{retries} for {action}: {last_error}", file=sys.stderr)
                time.sleep(0.3 * (attempt + 1))

        status = "OK" if ok else "FAIL"
        detail_str = f" [{detail}]" if detail else ""
        err_str = f" ({last_error})" if not ok and last_error else ""

        if log_func is not None:
            log_func(f"[Robot] {label} ({action}){detail_str} -> {status}{err_str}")
        if output_file:
            try:
                write_log(f"COMMAND: {action} [{status}] (text: {text})", output_file)
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def write_log(text: str, filepath: str):
    mode = "a" if os.path.exists(filepath) else "w"
    line = f"[{timestamp()}] {text}\n"
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(line)


class CameraStream:
    """Camera stream handler using direct HTTP MJPEG streaming.

    Avoids OpenCV VideoCapture/FFmpeg backend which often fails with
    TCP connection errors on embedded camera servers.
    """

    def __init__(self, robot_ip: str, log_func=None, custom_url=None):
        self.robot_ip = robot_ip
        self._custom_url = custom_url
        self.stream_url = custom_url if custom_url else f"http://{robot_ip}/api/camera/stream"
        self._log = log_func if log_func else lambda msg, **kw: None
        self._running = False
        self._thread = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._recording = False
        self._video_writer = None
        self._photo_dir = os.path.join(os.path.expanduser("~"), "robot_photos")
        self._video_dir = os.path.join(os.path.expanduser("~"), "robot_videos")
        os.makedirs(self._photo_dir, exist_ok=True)
        os.makedirs(self._video_dir, exist_ok=True)

    def start(self):
        if self._running:
            return True
        if not HAS_CV2:
            self._log("[Camera] ERROR: OpenCV not installed. Install with: pip install opencv-python", file=sys.stderr)
            return False

        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        self._log("[Camera] Opening stream...")
        return True

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._recording:
            self.stop_recording()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._log("[Camera] Stream stopped.")

    def _get_stream_urls(self):
        """Return list of URLs to try, in priority order."""
        urls = []
        if self._custom_url:
            urls.append(self._custom_url)
            return urls

        base = f"http://{self.robot_ip}"
        # Primary URLs first
        urls.append(f"{base}/api/camera/stream")
        urls.append(f"{base}/camera/stream")
        urls.append(f"{base}/video/stream")
        urls.append(f"{base}/camera")
        urls.append(f"{base}/video")
        urls.append(f"{base}/mjpeg/stream")
        urls.append(f"{base}/stream")
        # Port 8080 fallbacks
        urls.append(f"{base}:8080/api/camera/stream")
        urls.append(f"{base}:8080/camera/stream")
        urls.append(f"{base}:8080/video")
        return urls

    def _test_url(self, url: str, timeout: float = 3.0) -> tuple:
        """Test if URL returns a valid MJPEG stream. Returns (ok, response_or_error)."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': '*/*',
                }
            )
            response = urllib.request.urlopen(req, timeout=timeout)
            # Read first chunk to verify it's JPEG data
            chunk = response.read(8192)
            if b'\xff\xd8' in chunk:
                # Put the chunk back by creating a new connection
                response.close()
                return (True, None)
            # Could be multipart boundary - check for content-type
            if b'Content-Type' in chunk or b'content-type' in chunk:
                response.close()
                return (True, None)
            response.close()
            return (False, "No JPEG data in response")
        except urllib.error.HTTPError as e:
            return (False, f"HTTP {e.code}")
        except urllib.error.URLError as e:
            return (False, f"Connection failed: {e.reason}")
        except Exception as e:
            return (False, str(e))

    def _stream_loop(self):
        """Main stream loop - tries HTTP MJPEG direct streaming first."""
        stream_urls = self._get_stream_urls()
        working_url = None

        # Test each URL
        for url in stream_urls:
            self._log(f"[Camera] Testing: {url}")
            ok, err = self._test_url(url)
            if ok:
                working_url = url
                self.stream_url = url
                self._log(f"[Camera] Stream URL ready: {url}")
                break
            else:
                self._log(f"[Camera]   -> {err}", file=sys.stderr)

        if working_url is None:
            self._log(f"[Camera] ERROR: No working stream URL found.", file=sys.stderr)
            self._running = False
            return

        # Open the stream with urllib
        try:
            req = urllib.request.Request(
                working_url,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': '*/*',
                }
            )
            response = urllib.request.urlopen(req, timeout=10.0)
        except Exception as e:
            self._log(f"[Camera] ERROR: Failed to open stream: {e}", file=sys.stderr)
            self._running = False
            return

        self._log(f"[Camera] Stream connected!")

        window_name = "Robot Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        buf = b''
        frame_count = 0
        last_frame_time = time.time()

        try:
            while self._running:
                try:
                    chunk = response.read(8192)
                    if not chunk:
                        # No data - check if we've been idle too long
                        if time.time() - last_frame_time > 5.0:
                            self._log("[Camera] No data for 5s, reconnecting...", file=sys.stderr)
                            break
                        time.sleep(0.01)
                        continue
                except Exception as e:
                    self._log(f"[Camera] Read error: {e}", file=sys.stderr)
                    break

                buf += chunk
                last_frame_time = time.time()

                # Extract JPEG frames from buffer
                frames_decoded = 0
                while True:
                    # Find JPEG start marker
                    soi = buf.find(b'\xff\xd8')
                    if soi < 0:
                        break

                    # Find JPEG end marker after start
                    eoi = buf.find(b'\xff\xd9', soi + 2)
                    if eoi < 0:
                        break

                    # Extract JPEG data (include end marker)
                    jpeg_data = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]

                    # Limit buffer size to prevent memory growth
                    if len(buf) > 1048576:  # 1MB max
                        buf = buf[-524288:]  # Keep last 512KB

                    # Decode frame
                    try:
                        frame = cv2.imdecode(
                            np.frombuffer(jpeg_data, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            continue

                        frames_decoded += 1
                        frame_count += 1

                        with self._frame_lock:
                            self._frame = frame.copy()

                        if self._recording and self._video_writer is not None:
                            self._video_writer.write(frame)

                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            self._running = False
                            break
                        elif key == ord('p'):
                            self.take_photo()
                        elif key == ord('r'):
                            if self._recording:
                                self.stop_recording()
                            else:
                                self.start_recording()

                    except Exception:
                        pass

                if not self._running:
                    break

        except Exception as e:
            self._log(f"[Camera] Stream loop error: {e}", file=sys.stderr)
        finally:
            try:
                response.close()
            except Exception:
                pass
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None
            cv2.destroyAllWindows()
            self._running = False
            self._recording = False
            self._log(f"[Camera] Stream closed. Decoded {frame_count} frames.")

    def take_photo(self) -> str:
        if not HAS_CV2:
            return ""
        with self._frame_lock:
            if self._frame is None:
                self._log("[Camera] No frame available", file=sys.stderr)
                return ""
            frame = self._frame.copy()
        filename = f"photo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self._photo_dir, filename)
        cv2.imwrite(filepath, frame)
        self._log(f"[Camera] Photo saved: {filepath}")
        return filepath

    def start_recording(self) -> bool:
        if not HAS_CV2 or self._recording:
            return False
        with self._frame_lock:
            if self._frame is None:
                return False
            height, width = self._frame.shape[:2]
        filename = f"video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(self._video_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
        if not self._video_writer.isOpened():
            self._video_writer = None
            return False
        self._recording = True
        self._log(f"[Camera] Recording started: {filepath}")
        return True

    def stop_recording(self) -> str:
        if not self._recording:
            return ""
        self._recording = False
        filepath = ""
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self._log("[Camera] Recording stopped.")
        return filepath

    @property
    def is_running(self):
        return self._running

    @property
    def is_recording(self):
        return self._recording


# --------------------------------------------------------------------------- #
#  VoiceRobot — Simple single-main-loop architecture
#  - Main loop: read audio -> VAD -> ASR -> command -> HTTP (fire-and-forget)
#  - No queue, no executor thread, no concurrency bugs
#  - HTTP calls never block the main loop
# --------------------------------------------------------------------------- #

class VoiceRobot:
    def __init__(self, args):
        self.sample_rate = SAMPLE_RATE
        self.device = args.device
        self.output_file = args.output
        self.audio_threshold = args.threshold
        self.audio_gain = args.gain
        self.robot_ip = args.robot_ip
        self.robot_speed = args.speed
        self.robot_duration = args.duration
        self.distance_factor = args.distance_factor
        self.turn_factor = args.turn_factor
        self.default_turn_duration = args.default_turn_duration  # Default duration for simple turns
        self.model_dir = args.model_dir
        self.quiet = args.quiet
        self.no_warmup = args.no_warmup
        self.delay_seconds = args.delay

        self._running = False
        self._model = None
        self._recorder_proc = None
        self._needs_wakeup = False
        self._camera = None
        self._auto_start_camera = args.camera
        self._camera_url = args.camera_url

        # Wake/sleep state
        self._awake = not args.wake_word  # Start awake unless wake-word mode enabled
        self._wake_word_mode = args.wake_word
        self._idle_timeout = args.idle_timeout
        self._last_activity_time = time.time()

        # Complex command sequencer state
        self._sequence_thread = None
        self._sequence_running = False
        self._sequence_stop_requested = False

    def _log(self, msg, file=None, flush=True):
        if not self.quiet:
            print(msg, file=file, flush=flush)

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
                        return full_path
                    self._log(f"[Model] Model dir exists but empty: {full_path}", file=sys.stderr)

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
            raise RuntimeError(
                "funasr not installed. Run: pip install funasr\n"
                "Then download model: python3 -c \"from funasr import AutoModel; "
                "AutoModel(model='iic/SenseVoiceSmall', use_onnx=True)\""
            )

        local_path = self._find_local_model()
        if not local_path:
            raise RuntimeError(
                "No local SenseVoice model found!\n"
                "Checked: ~/.cache/modelscope/hub/models/iic/SenseVoiceSmall\n"
                "         ~/.cache/sensevoice/\n"
                "Use --model-dir /path/to/model or download first."
            )

        self._log(f"[Model] Using: {local_path}")
        try:
            self._model = AutoModel(
                model=local_path,
                model_type="asr",
                use_onnx=True,
                disable_pbar=True,
                disable_update=True,
                trust_remote_code=True,
            )
            self._log("[Model] Loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        if not getattr(self, 'no_warmup', False):
            self._log("[Model] Warming up...")
            warmup_result = [None]
            def do_warmup():
                try:
                    warmup = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
                    self._model.generate(warmup)
                    warmup_result[0] = "ok"
                except Exception as e:
                    warmup_result[0] = str(e)
            t = threading.Thread(target=do_warmup)
            t.daemon = True
            t.start()
            t.join(timeout=10)
            if warmup_result[0] == "ok":
                self._log("[Model] Ready.")
            elif warmup_result[0] is None:
                self._log("[Model] Warm-up timed out (10s), continuing anyway.")
            else:
                self._log(f"[Model] Warm-up note: {warmup_result[0]}")
        else:
            self._log("[Model] Warm-up skipped.")

    def _detect_alsa_devices(self):
        """Detect available ALSA capture devices."""
        devices = []
        try:
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if "card" in line and "device" in line:
                    parts = line.split(':')
                    card_part = parts[0].strip()
                    card_num = card_part.split(' ')[1]
                    devices.append(int(card_num))
        except Exception:
            pass
        return sorted(set(devices))

    def _start_recorder(self):
        if os.path.exists(PCM_FILE):
            os.remove(PCM_FILE)

        # Kill any existing arecord processes that might be using the device
        try:
            subprocess.run(["pkill", "-9", "-f", "arecord"], 
                          capture_output=True, timeout=1.0)
            time.sleep(0.3)
        except Exception:
            pass

        available_devices = self._detect_alsa_devices()
        if available_devices:
            self._log(f"[Recorder] Available devices: {available_devices}", file=sys.stderr)
        else:
            self._log("[Recorder] No ALSA devices detected, will try defaults", file=sys.stderr)

        # Build list of devices to try
        devices_to_try = []
        if self.device is not None:
            devices_to_try.append(('plughw', self.device))
            devices_to_try.append(('hw', self.device))
        
        # Add other available devices
        for dev_num in available_devices:
            if dev_num != self.device:
                devices_to_try.append(('plughw', dev_num))
                devices_to_try.append(('hw', dev_num))
        
        # Add default (no -D parameter)
        devices_to_try.append((None, None))

        for dev_type, dev_num in devices_to_try:
            if dev_type and dev_num is not None:
                device_str = f"{dev_type}:{dev_num},0"
                cmd = ["arecord", "-r", str(self.sample_rate), "-f", "S16_LE",
                       "-c", "1", "-t", "raw", "-D", device_str]
            else:
                device_str = "default"
                cmd = ["arecord", "-r", str(self.sample_rate), "-f", "S16_LE",
                       "-c", "1", "-t", "raw"]

            self._log(f"[Recorder] Trying: arecord -D {device_str}")

            try:
                self._recorder_proc = subprocess.Popen(
                    cmd,
                    stdout=open(PCM_FILE, "wb"),
                    stderr=subprocess.PIPE,
                    bufsize=4096,
                )

                import select
                ready, _, _ = select.select([self._recorder_proc.stderr], [], [], 1.0)
                if ready:
                    err = self._recorder_proc.stderr.read1(1024).decode('utf-8', errors='ignore')
                    if err and ("error" in err.lower() or "fail" in err.lower() or 
                               "busy" in err.lower() or "忙" in err or "没有" in err):
                        self._log(f"[Recorder] Error on {device_str}: {err.strip()}", file=sys.stderr)
                        self._recorder_proc.terminate()
                        self._recorder_proc.wait(timeout=2)
                        time.sleep(0.3)
                        continue

                time.sleep(0.5)
                if self._recorder_proc.poll() is None:
                    self._log(f"[Recorder] Recording on {device_str}.")
                    return True
                else:
                    stderr_output = ""
                    try:
                        if self._recorder_proc.stderr:
                            stderr_output = self._recorder_proc.stderr.read().decode('utf-8', errors='ignore').strip()
                    except Exception:
                        pass
                    self._log(f"[Recorder] Failed on {device_str}: {stderr_output or 'exited prematurely'}", file=sys.stderr)

            except Exception as e:
                self._log(f"[Recorder] Exception on {device_str}: {e}", file=sys.stderr)

        self._log("[Recorder] Failed to start on any device!", file=sys.stderr)
        return False

    def _stop_recorder(self):
        if self._recorder_proc and self._recorder_proc.poll() is None:
            self._recorder_proc.terminate()
            try:
                self._recorder_proc.wait(timeout=2)
            except Exception:
                pass
        self._log("[Recorder] Stopped.")

    def _dispatch_command(self, text: str, action: str = None):
        if action is None:
            action = understand_command(text)

        # Check for complex commands first (before simple commands)
        complex_action = detect_complex_command(text)
        if complex_action:
            if not self.robot_ip:
                self._log("[Sequence] No robot IP configured", file=sys.stderr)
                return False
            self._run_complex_sequence(complex_action, text)
            return True

        if not action:
            try:
                write_log(text, self.output_file)
            except Exception:
                pass
            return False

        camera_actions = {"camera_on", "camera_off", "photo", "record_start", "record_stop"}
        if action in camera_actions:
            self._handle_camera_command(action, text)
            return True

        # If stop command received, also stop any running sequence
        if action == "stop" and self._sequence_running:
            self._stop_sequence()
            return True

        # Warn about commands that may not be supported by robot firmware
        if action == "down":
            self._log("[Robot] NOTE: Backward command sent. Some robots don't support backward movement.", file=sys.stderr)

        params = extract_parameters(text, self.robot_speed)

        dur = compute_action_duration(
            params, action, self.robot_duration, self.distance_factor
        )
        turn_dur = compute_turn_duration(
            params, action, params["speed"], self.turn_factor,
            self.default_turn_duration  # Use default for simple turns
        )
        if turn_dur is not None:
            dur = turn_dur

        label = ACTION_LABEL.get(action, action)
        detail_parts = []
        if params["distance"] is not None:
            detail_parts.append(f"{params['distance']}m")
        if params["angle"] is not None:
            detail_parts.append(f"{params['angle']}\u00b0")
        if params["duration"] is not None:
            detail_parts.append(f"{params['duration']}s")
        if params["speed"] != self.robot_speed:
            detail_parts.append(f"speed={params['speed']}")
        detail = ", ".join(detail_parts)

        self._log(f"[Command] {label} ({action})" + (f" [{detail}]" if detail else ""))

        if self.robot_ip:
            is_stop = (action == "stop")
            
            if action == "stop":
                self._stop_sequence()
                
                url = f"http://{self.robot_ip}/api/control?action=stop&speed={self.robot_speed}"
                self._log(f"[Robot] Sending STOP (x3)...")
                for i in range(3):
                    try:
                        req = urllib.request.Request(url, method='GET')
                        resp = urllib.request.urlopen(req, timeout=1.0)
                        resp.read()
                    except Exception as e:
                        self._log(f"[Robot] STOP attempt {i+1} failed: {e}", file=sys.stderr)
                    if i < 2:
                        time.sleep(0.05)
                self._log(f"[Robot] {label} ({action}) -> OK")
                self._needs_wakeup = True
            else:
                # Some robots need a brief ping after stop to re-enable motors
                if self._needs_wakeup:
                    self._needs_wakeup = False
                    self._log("[Robot] Waking up motor controller...")
                    send_robot_command_async(
                        self.robot_ip, "stop", self.robot_speed, None,
                        log_func=None, label="", detail="",
                        output_file=None, text="",
                        is_stop=True,
                    )
                    time.sleep(0.1)
                send_robot_command_async(
                    self.robot_ip, action, params["speed"], dur,
                    log_func=self._log, label=label, detail=detail,
                    output_file=self.output_file, text=text,
                    is_stop=False,
                )

        return True

    def _handle_camera_command(self, action: str, text: str):
        label = ACTION_LABEL.get(action, action)
        self._log(f"[Command] {label} ({action})")

        if not self.robot_ip:
            self._log("[Camera] No robot IP configured", file=sys.stderr)
            return

        if self._camera is None:
            self._camera = CameraStream(self.robot_ip, log_func=self._log, custom_url=self._camera_url)

        if action == "camera_on":
            ok = self._camera.start()
            if not ok:
                self._log("[Camera] Failed to start camera stream", file=sys.stderr)
        elif action == "camera_off":
            self._camera.stop()
        elif action == "photo":
            if not self._camera.is_running:
                self._camera.start()
                time.sleep(1.0)
            self._camera.take_photo()
        elif action == "record_start":
            if not self._camera.is_running:
                self._camera.start()
                time.sleep(1.0)
            self._camera.start_recording()
        elif action == "record_stop":
            self._camera.stop_recording()

    def _run_complex_sequence(self, action: str, text: str):
        """Run a complex multi-step command sequence in a background thread."""
        if self._sequence_running:
            self._log("[Sequence] Another sequence already running, stopping it first...")
            self._sequence_stop_requested = True
            time.sleep(0.3)

        steps = generate_complex_sequence(
            action, text,
            base_speed=self.robot_speed,
            distance_factor=self.distance_factor,
            turn_factor=self.turn_factor,
        )

        if not steps:
            self._log("[Sequence] No steps generated.", file=sys.stderr)
            return

        label = COMPLEX_ACTION_LABEL.get(action, action)
        self._log(f"[Sequence] Starting: {label} ({len(steps)} steps)")
        print(f">>> [执行复杂指令: {label}] {len(steps)}步", flush=True)

        self._sequence_running = True
        self._sequence_stop_requested = False

        def _sequence_worker():
            try:
                self._needs_wakeup = False
                for i, step in enumerate(steps):
                    if self._sequence_stop_requested:
                        self._log(f"[Sequence] Stopped at step {i+1}/{len(steps)}")
                        break

                    step_action = step["action"]
                    step_speed = step["speed"]
                    step_duration = step.get("duration")
                    step_label = step.get("label", "")

                    is_stop = (step_action == "stop")

                    self._log(f"[Sequence] Step {i+1}/{len(steps)}: {step_label or step_action}")

                    if is_stop:
                        send_robot_command_async(
                            self.robot_ip, "stop", step_speed, None,
                            log_func=None, label="", detail="",
                            output_file=None, text="",
                            is_stop=True,
                        )
                        self._needs_wakeup = True
                    else:
                        if self._needs_wakeup:
                            self._needs_wakeup = False
                            send_robot_command_async(
                                self.robot_ip, "stop", step_speed, None,
                                log_func=None, label="", detail="",
                                output_file=None, text="",
                                is_stop=True,
                            )
                            time.sleep(0.1)
                        send_robot_command_async(
                            self.robot_ip, step_action, step_speed, step_duration,
                            log_func=None, label="", detail="",
                            output_file=None, text="",
                            is_stop=False,
                        )

                    # Wait for the step to complete
                    if step_duration is not None:
                        wait_time = step_duration + step.get("delay_after", 0.1)
                        elapsed = 0.0
                        while elapsed < wait_time:
                            if self._sequence_stop_requested:
                                break
                            time.sleep(0.05)
                            elapsed += 0.05
                    else:
                        time.sleep(step.get("delay_after", 0.1))

                if not self._sequence_stop_requested:
                    self._log(f"[Sequence] Complete: {label}")
                    print(f">>> [复杂指令完成: {label}]", flush=True)

            except Exception as e:
                self._log(f"[Sequence] Error: {e}", file=sys.stderr)
            finally:
                self._sequence_running = False
                self._sequence_stop_requested = False

        self._sequence_thread = threading.Thread(target=_sequence_worker, daemon=True)
        self._sequence_thread.start()

    def _stop_sequence(self):
        """Stop the currently running complex sequence."""
        if self._sequence_running:
            self._sequence_stop_requested = True
            if self.robot_ip:
                self._log("[Sequence] Stopping sequence - sending STOP...")
                for i in range(3):
                    url = f"http://{self.robot_ip}/api/control?action=stop&speed={self.robot_speed}"
                    try:
                        req = urllib.request.Request(url, method='GET')
                        resp = urllib.request.urlopen(req, timeout=1.0)
                        resp.read()
                    except Exception:
                        pass
                    if i < 2:
                        time.sleep(0.05)
            self._log("[Sequence] Stop requested.")

    def _send_stop_blocking(self):
        """Send stop command synchronously with retry."""
        url = f"http://{self.robot_ip}/api/control?action=stop&speed={self.robot_speed}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, method='GET')
                resp = urllib.request.urlopen(req, timeout=2.0)
                resp.read()
                return True
            except Exception:
                if attempt < 2:
                    time.sleep(0.1)
        return False

    def _process_segment(self, speech_segment, output_history):
        audio_full = np.concatenate(speech_segment)
        min_samples = int(self.sample_rate * 0.1)
        audio_duration = len(audio_full) / self.sample_rate

        if len(audio_full) < min_samples:
            self._log(f"[ASR] Skipped: too short ({audio_duration:.2f}s < 0.1s)", file=sys.stderr)
            return

        self._log(f"[ASR] Processing {audio_duration:.2f}s segment...", file=sys.stderr)
        result = [None]
        def _do_generate():
            try:
                result[0] = self._model.generate(audio_full)
            except Exception as e:
                result[0] = e

        t_gen = threading.Thread(target=_do_generate, daemon=True)
        t_gen.start()
        t_gen.join(timeout=5.0)

        if t_gen.is_alive():
            self._log("[ASR] Model inference timed out, skipping.", file=sys.stderr)
            return

        if isinstance(result[0], Exception):
            self._log(f"[ASR] Error: {result[0]}", file=sys.stderr)
            return

        text = result[0][0].get('text', '').strip() if result[0] and len(result[0]) > 0 else ""
        if not text:
            self._log("[ASR] Empty result from model", file=sys.stderr)
            return

        text = clean_asr_text(text)
        if not text:
            self._log("[ASR] Empty after cleaning tokens", file=sys.stderr)
            return

        if is_hallucination(text):
            self._log(f"[ASR] Skipped hallucination: '{text}'", file=sys.stderr)
            return

        text = remove_internal_repeats(text).strip()
        if not text:
            self._log("[ASR] Empty after removing repeats", file=sys.stderr)
            return

        # Apply ASR error corrections
        corrected = correct_asr_errors(text)
        if corrected != text:
            self._log(f"[ASR] Corrected '{text}' -> '{corrected}'", file=sys.stderr)
            text = corrected

        for old_text, old_time in output_history:
            if time.time() - old_time > 10.0:
                continue
            if is_duplicate(text, old_text):
                self._log(f"[ASR] Skipped duplicate: '{text}'", file=sys.stderr)
                return

        output_history.append((text, time.time()))
        if len(output_history) > 10:
            output_history.pop(0)

        # Check wake/sleep words first (always active)
        if is_wake_word(text):
            if not self._awake:
                self._awake = True
                self._last_activity_time = time.time()
                print(f">>> {text}  [已唤醒 / WAKE UP]", flush=True)
                self._log("[Wake] System activated by wake word.")
            else:
                print(f">>> {text}  [已唤醒]", flush=True)
            return

        if is_sleep_word(text):
            if self._awake:
                self._awake = False
                print(f">>> {text}  [已休眠 / SLEEP]", flush=True)
                self._log("[Sleep] System deactivated by sleep word.")
            else:
                print(f">>> {text}  [休眠中]", flush=True)
            return

        # If in sleep mode, show text but don't execute commands
        if not self._awake:
            print(f">>> {text}  [休眠中 - 说唤醒词激活]", flush=True)
            return

        # System is awake - normal command processing
        print(f">>> {text}", flush=True)

        # Check idle timeout
        if self._wake_word_mode and self._idle_timeout > 0:
            idle_time = time.time() - self._last_activity_time
            if idle_time > self._idle_timeout:
                self._awake = False
                self._log(f"[Sleep] Idle timeout ({self._idle_timeout}s). Going to sleep.")
                print(">>> [系统休眠 / System sleeping - say wake word to activate]", flush=True)
                return

        action = understand_command(text)
        if not action:
            self._log(f"[Command] No action detected for: '{text}'", file=sys.stderr)
            try:
                write_log(text, self.output_file)
            except Exception:
                pass
            self._last_activity_time = time.time()
            return

        self._last_activity_time = time.time()
        self._dispatch_command(text, action)

    def run(self):
        self._log("=" * 55)
        self._log("  Voice Controlled Robot Car")
        self._log("=" * 55)
        self._log(f"  Audio device : {self.device}")
        self._log(f"  Sample rate  : {self.sample_rate} Hz")
        self._log(f"  Delay        : {self.delay_seconds}s")
        if self.robot_ip:
            self._log(f"  Robot IP     : {self.robot_ip}")
            self._log(f"  Speed        : {self.robot_speed}")
            self._log(f"  Calibration  : {self.distance_factor} m/s at speed=50")
            self._log(f"  Turn calib   : {self.turn_factor}s per 90° at speed=50")
            self._log("  Calibrate    : Say '左转九十度' and measure actual turn")
            self._log("                 If >90°: lower --turn-factor")
            self._log("                 If <90°: raise --turn-factor")
        if self._wake_word_mode:
            self._log(f"  Wake mode    : ON (idle timeout: {self._idle_timeout}s)")
            self._log("  Wake words   : 机器人, 小车, 助手, wake up, hello, 你好...")
            self._log("  Sleep words  : 睡觉, 休眠, 休息, sleep, pause, 再见...")
        else:
            self._log("  Wake mode    : OFF (always listening)")
        self._log("  Press Ctrl+C to stop")
        self._log("=" * 55)

        self._load_model()
        if not self._start_recorder():
            return

        time.sleep(0.5)
        if self._recorder_proc and self._recorder_proc.poll() is not None:
            stderr_output = ""
            try:
                if self._recorder_proc.stderr:
                    stderr_output = self._recorder_proc.stderr.read().decode('utf-8', errors='ignore').strip()
            except Exception:
                pass
            if stderr_output:
                self._log(f"[Recorder] Failed to start: {stderr_output}", file=sys.stderr)
            else:
                self._log(f"[Recorder] Failed to start, exit code: {self._recorder_proc.returncode}", file=sys.stderr)
            return

        # Test robot connectivity before accepting commands
        if self.robot_ip:
            self._log("[Robot] Testing connection...")
            robot_ok = False
            last_err = None
            for attempt in range(3):
                try:
                    url = f"http://{self.robot_ip}/api/control?action=stop&speed={self.robot_speed}"
                    req = urllib.request.Request(url, method='GET')
                    resp = urllib.request.urlopen(req, timeout=2.0)
                    resp.read()
                    robot_ok = True
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(0.3)
            if robot_ok:
                self._log("[Robot] Connection OK")
            else:
                self._log(f"[Robot] WARNING: Cannot connect to {self.robot_ip}, commands may fail! ({last_err})", file=sys.stderr)

        bytes_per_sec = self.sample_rate * SAMPLE_WIDTH
        buffer_delay = min(self.delay_seconds, 0.1)
        delay_bytes = int(bytes_per_sec * buffer_delay)

        frame_size = 512
        frame_bytes = frame_size * SAMPLE_WIDTH
        max_segment_frames = int(self.sample_rate * 3 / frame_size)
        silence_frames_threshold = 3
        vad_start_frames = 1
        command_cooldown = 1.0

        buffer_wait_timeout = 10.0
        buffer_wait_start = time.time()
        buffer_ready = False
        recorder_restarted = False
        
        while True:
            elapsed = time.time() - buffer_wait_start
            if elapsed >= buffer_wait_timeout:
                self._log(f"[Recorder] Buffer wait timed out after {buffer_wait_timeout}s", file=sys.stderr)
                
                if not recorder_restarted:
                    self._log("[Recorder] Attempting to restart recorder...", file=sys.stderr)
                    self._stop_recorder()
                    if os.path.exists(PCM_FILE):
                        try:
                            os.remove(PCM_FILE)
                        except Exception:
                            pass
                    if self._start_recorder():
                        recorder_restarted = True
                        buffer_wait_start = time.time()
                        continue
                    else:
                        self._log("[Recorder] Restart failed", file=sys.stderr)
                
                if self._recorder_proc and self._recorder_proc.poll() is not None:
                    self._log(f"[Recorder] Recorder process died (exit code: {self._recorder_proc.returncode})", file=sys.stderr)
                else:
                    self._log(f"[Recorder] Recorder process still running but file not growing", file=sys.stderr)
                break
            
            if self._recorder_proc and self._recorder_proc.poll() is not None:
                self._log(f"[Recorder] Recorder process died unexpectedly", file=sys.stderr)
                
                if not recorder_restarted:
                    self._log("[Recorder] Attempting to restart recorder...", file=sys.stderr)
                    if self._start_recorder():
                        recorder_restarted = True
                        buffer_wait_start = time.time()
                        continue
                
                break
            
            if os.path.exists(PCM_FILE):
                try:
                    file_size = os.path.getsize(PCM_FILE)
                    if file_size >= delay_bytes:
                        buffer_ready = True
                        break
                    if file_size > 0:
                        self._log(f"[Recorder] Buffer filling... {file_size}/{delay_bytes} bytes", file=sys.stderr)
                except OSError:
                    pass
            
            time.sleep(0.02)

        if not buffer_ready:
            self._log("[ASR] Cannot start listening - audio buffer not ready", file=sys.stderr)
            return

        self._log("[ASR] Listening...")

        if self._auto_start_camera and self.robot_ip:
            self._handle_camera_command("camera_on", "")

        fd = open(PCM_FILE, "rb")
        read_pos = 0
        output_history = []

        speech_segment = []
        vad_frames = 0
        silence_frames = 0
        last_file_size = 0
        no_growth_count = 0
        self._running = True
        loop_counter = 0
        last_command_time = 0

        try:
            while self._running:
                try:
                    loop_counter += 1
                    if loop_counter >= 100:
                        loop_counter = 0
                        self._log("[ASR] heartbeat")

                    file_size = os.path.getsize(PCM_FILE)
                except OSError:
                    time.sleep(0.02)
                    continue

                if file_size == last_file_size:
                    no_growth_count += 1
                    if no_growth_count > 250:
                        if self._recorder_proc and self._recorder_proc.poll() is not None:
                            self._log("[Recorder] Process died, stopping.", file=sys.stderr)
                            break
                        no_growth_count = 0
                else:
                    last_file_size = file_size
                    no_growth_count = 0

                available = file_size - read_pos - delay_bytes
                if available < frame_bytes:
                    time.sleep(0.02)
                    continue

                frames_to_read = min(available // frame_bytes, 8)
                read_len = frame_bytes * frames_to_read

                fd.seek(read_pos)
                raw_data = fd.read(read_len)
                if len(raw_data) < frame_bytes:
                    time.sleep(0.02)
                    continue
                read_pos += len(raw_data)

                for i in range(0, len(raw_data), frame_bytes):
                    chunk = raw_data[i:i+frame_bytes]
                    if len(chunk) < frame_bytes:
                        break

                    audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    if self.audio_gain != 1.0:
                        audio_np = np.clip(audio_np * self.audio_gain, -1.0, 1.0)

                    rms = np.sqrt(np.mean(audio_np ** 2))

                    if rms > self.audio_threshold:
                        vad_frames += 1
                        silence_frames = 0
                        if vad_frames >= vad_start_frames:
                            speech_segment.append(audio_np)
                    else:
                        silence_frames += 1
                        if vad_frames >= vad_start_frames:
                            speech_segment.append(audio_np)

                    segment_ready = (silence_frames >= silence_frames_threshold
                                     or len(speech_segment) >= max_segment_frames)
                    if segment_ready and len(speech_segment) > 0:
                        self._process_segment(speech_segment, output_history)
                        speech_segment = []
                        vad_frames = 0
                        silence_frames = 0

        except KeyboardInterrupt:
            self._log("\nStopping...")
        finally:
            self._running = False
            # Stop any running complex sequence
            if self._sequence_running:
                self._stop_sequence()
                time.sleep(0.3)
            if self.robot_ip:
                self._log("[Shutdown] Sending STOP to robot (x3)...")
                url = f"http://{self.robot_ip}/api/control?action=stop&speed={self.robot_speed}"
                for i in range(3):
                    try:
                        req = urllib.request.Request(url, method='GET')
                        resp = urllib.request.urlopen(req, timeout=1.0)
                        resp.read()
                    except Exception as e:
                        self._log(f"[Shutdown] STOP attempt {i+1} failed: {e}", file=sys.stderr)
                    if i < 2:
                        time.sleep(0.05)
                self._log("[Shutdown] Robot stopped.")
            fd.close()
            self._stop_recorder()
            if self._camera:
                self._camera.stop()
            if os.path.exists(PCM_FILE):
                try:
                    os.remove(PCM_FILE)
                except Exception:
                    pass
            self._log("Done.")



# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def list_alsa_devices():
    print("ALSA Capture Devices:")
    print("-" * 60)
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        print(result.stdout or "  (none found)")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    print("Usage: --device <card_number>")
    print("Example: --device 2  (uses plughw:2,0)")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Controlled Robot Car — microphone → ASR → HTTP robot control"
    )
    parser.add_argument("--device", type=int, default=None,
                        help="ALSA card number (use --list-alsa-devices to see)")
    parser.add_argument("--robot-ip", default=None,
                        help="Robot car IP (e.g. 192.168.4.1)")
    parser.add_argument("--speed", type=int, default=50,
                        help="Robot speed 0-100 (default: 50)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Action duration in seconds (optional)")
    parser.add_argument("--distance-factor", type=float, default=0.3,
                        help="Meters per second at speed=50 (default: 0.3, typical small robot)")
    parser.add_argument("--turn-factor", type=float, default=0.5,
                        help="Seconds to turn 90° at speed=50 (default: 0.5)")
    parser.add_argument("--default-turn-duration", type=float, default=1.0,
                        help="Default duration for simple turns without angle (default: 1.0s)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Recorder→transcriber delay in seconds (default: 0.1, optimized)")
    parser.add_argument("--threshold", type=float, default=0.003,
                        help="VAD RMS threshold (default: 0.003, lower = more sensitive)")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Audio gain multiplier (default: 1.0)")
    parser.add_argument("--model-dir", default=None,
                        help="Path to local SenseVoice model")
    parser.add_argument("--output", default="voice_output.txt",
                        help="Output log file (default: voice_output.txt)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show transcribed text (suppress logs)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip model warm-up (use if loading hangs)")
    parser.add_argument("--list-alsa-devices", action="store_true",
                        help="List ALSA capture devices and exit")
    parser.add_argument("--test-commands", action="store_true",
                        help="Test command matching with sample phrases and exit")
    parser.add_argument("--camera", action="store_true",
                        help="Auto-open camera stream on startup")
    parser.add_argument("--camera-url", default=None,
                        help="Custom camera stream URL (default: http://robot_ip/api/camera/stream)")
    parser.add_argument("--wake-word", action="store_true",
                        help="Enable wake-word mode (system sleeps until wake word detected)")
    parser.add_argument("--idle-timeout", type=float, default=30.0,
                        help="Idle timeout in seconds before auto-sleep (default: 30, requires --wake-word)")

    args = parser.parse_args()

    if args.list_alsa_devices:
        list_alsa_devices()
        return

    if args.test_commands:
        test_phrases = [
            # Movement commands
            "前进", "向前走", "往前走", "直走", "走", "走两米",
            "后退", "往后退", "倒车", "退", "退回去",
            # Turn commands
            "左转", "向左转", "往左拐", "左拐", "往左走",
            "右转", "向右转", "往右拐", "右拐", "往右走",
            # Priority actions
            "停止", "停下", "停", "站住", "别动",
            "抓取", "抓起来", "拿起来", "抓", "夹起来",
            "释放", "放下", "放下来", "松开", "放",
            # Combined commands
            "请前进到红色方块处", "左转一下然后停止",
            "把那个东西抓起来", "放到桌子上",
            "往左走三米", "向右走两米",
            # Speed modifiers (no action)
            "快点", "慢慢", "慢点",
            # Unclear/no-action phrases
            "权认", "你好", "随便说说",
            "今天天气很好", "无意义",
        ]
        print("Semantic Command Understanding Test")
        print("=" * 70)
        print(f"  {'Phrase':<30s} {'Action':<10s} {'Label':<8s}")
        print(f"  {'-'*30} {'-'*10} {'-'*8}")
        for phrase in test_phrases:
            action = understand_command(phrase)
            label = ACTION_LABEL.get(action, "—") if action else "无动作"
            status = "✓" if action else "○"
            print(f"  {status}  {phrase:<30s} {action:<10s} {label:<8s}")
        print("=" * 70)
        print("  ✓ = command detected, ○ = no action (correct for unclear phrases)")
        print()

        param_phrases = [
            "往前走二米",
            "前进三米",
            "后退两米",
            "向前走五点五米",
            "前进五秒钟",
            "左转九十度",
            "右转四十五度",
            "快点前进",
            "慢慢后退",
            "往前走二米快点",
            "左转三十度慢点",
            "前进5米",
            "右转90度",
            "后退3秒",
        ]
        print("Parameter Extraction Test")
        print("=" * 75)
        print(f"  {'Phrase':<25s} {'Action':<8s} {'Dist':>5s} {'Dur':>5s} {'Angle':>6s} {'Speed':>6s}")
        print(f"  {'-'*25} {'-'*8} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")
        for phrase in param_phrases:
            action = match_action(phrase)
            params = extract_parameters(phrase, 50)
            dur = compute_action_duration(params, action, None)
            turn_dur = compute_turn_duration(params, action, params["speed"])
            if turn_dur is not None:
                dur = turn_dur
            dist_str = f"{params['distance']}m" if params["distance"] is not None else "—"
            dur_str = f"{dur:.1f}s" if dur is not None else "—"
            angle_str = f"{params['angle']}°" if params["angle"] is not None else "—"
            speed_str = f"{params['speed']}"
            print(f"  {phrase:<25s} {action:<8s} {dist_str:>5s} {dur_str:>5s} {angle_str:>6s} {speed_str:>6s}")
        print("=" * 75)
        return

    robot = VoiceRobot(args)
    robot.run()


if __name__ == "__main__":
    main()
