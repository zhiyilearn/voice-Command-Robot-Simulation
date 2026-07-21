# Voice Command Robot

A Python application that enables voice control of a robot car via Chinese speech commands. It captures audio from a microphone, transcribes speech using the SenseVoice ASR model, and sends movement commands to a robot car over HTTP.

## Features

- **Real-time Speech Recognition**: Uses SenseVoice Small (FunASR) for accurate Chinese speech-to-text transcription
- **Streaming ASR Display**: Shows interim transcription results in real-time as you speak
- **Voice Activity Detection (VAD)**: Automatically detects speech start/end with ~0.6s silence threshold
- **Wake Word Mode**: Optional sleep/wake system activated by wake words (机器人, 小车, wake up, etc.)
- **Complex Commands**: Supports multi-step trajectories (square, circle, triangle, figure-8, S-curve, forward-back repeat)
- **Camera Control**: View live stream, take photos, and record video from robot camera
- **Non-blocking HTTP**: Robot commands run asynchronously so audio capture never blocks
- **Command Deduplication**: Prevents accidental repeated command execution with cooldown
- **Timing Logs**: Tracks latency from voice acceptance to CLI output to robot execution

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| PC / Embedded | Laptop or embedded board (e.g., RK3588) |
| OS | Ubuntu / Linux |
| Microphone | Built-in or USB microphone |
| Robot | Car robot with HTTP API (e.g., IP: 192.168.4.1) |

### Check Microphone Devices

```bash
arecord -l
```

Example output:
```
card 1: PCH [HDA Intel PCH], device 0: ALC233 Analog [ALC233 Analog]
  子设备: 1/1
  子设备 #0: subdevice #0
card 2: Audio [UGREEN CM564 USB Audio], device 0: USB Audio [USB Audio]
  子设备: 1/1
  子设备 #0: subdevice #0
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch + torchaudio
- FunASR (SenseVoice model)
- NumPy
- OpenCV (optional, for camera features)
- `arecord` (ALSA utilities, usually pre-installed on Ubuntu)

## Quick Start

### Basic Usage

```bash
python3 voiceCommandRobot.py --device 2 --robot-ip 192.168.4.1
```

### Calibrated Usage

```bash
python3 voiceCommandRobot.py \
  --device 2 \
  --robot-ip 192.168.4.1 \
  --distance-factor 0.08 \
  --turn-factor-left 1.8 \
  --turn-factor-right 1.5
```

### With Wake Word Mode

```bash
python3 voiceCommandRobot.py \
  --device 2 \
  --robot-ip 192.168.4.1 \
  --wake-word \
  --idle-timeout 30
```

### List Available Audio Devices

```bash
python3 voiceCommandRobot.py --list-alsa-devices
```

### Test Command Matching (No Robot Required)

```bash
python3 voiceCommandRobot.py --test-commands
```

### Test Robot Connection

```bash
python3 voiceCommandRobot.py --robot-ip 192.168.4.1 --test-connection
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `None` | ALSA card number (see `--list-alsa-devices`) |
| `--robot-ip` | `None` | Robot car IP address |
| `--speed` | `50` | Robot speed 0-100 |
| `--distance-factor` | `0.3` | Meters per second at speed=50 |
| `--turn-factor` | `0.5` | Seconds per 90° turn at speed=50 |
| `--turn-factor-left` | `None` | Left turn calibration (falls back to `--turn-factor`) |
| `--turn-factor-right` | `None` | Right turn calibration (falls back to `--turn-factor`) |
| `--default-turn-duration` | `1.0` | Default duration for simple turns without angle |
| `--delay` | `0.02` | Recorder-to-transcriber delay (lower = more responsive) |
| `--threshold` | `0.003` | VAD sensitivity threshold (lower = more sensitive) |
| `--gain` | `1.0` | Audio gain multiplier |
| `--model-dir` | `None` | Path to local SenseVoice model |
| `--output` | `voice_output.txt` | Output log file |
| `--quiet` | `False` | Suppress non-essential logs |
| `--no-warmup` | `False` | Skip model warm-up |
| `--camera` | `False` | Auto-open camera stream on startup |
| `--camera-url` | `None` | Custom camera stream URL |
| `--wake-word` | `False` | Enable wake-word mode |
| `--idle-timeout` | `30.0` | Auto-sleep timeout (requires `--wake-word`) |
| `--no-streaming` | `False` | Disable real-time streaming ASR display |
| `--list-alsa-devices` | `False` | List audio devices and exit |
| `--test-commands` | `False` | Test command matching and exit |
| `--test-connection` | `False` | Test robot connection and exit |
| `--test-robot` | `False` | Test robot movement commands |

## Supported Voice Commands

### Movement Commands

All movement commands should be prefixed with **机器人** when the system is awake.

| Command (Chinese) | Action | Example |
|-------------------|--------|---------|
| 前进 / 向前走 / 往前走 | Move forward | 机器人向前走一米 |
| 后退 / 向后退 / 往后退 | Move backward | 机器人后退零点五米 |
| 左转 / 向左转 | Turn left | 机器人左转九十度 |
| 右转 / 向右转 | Turn right | 机器人右转四十五度 |
| 停 / 停止 / 停下 | Stop | 机器人停 |
| 抓 / 拿 | Grab | 机器人抓 |
| 放 / 松开 | Release | 机器人放 |

### Parameter Examples

- Distance: 向前走**一米**, 后退**零点五米**, 前进**两米**
- Angle: 左转**九十度**, 右转**四十五度**
- Speed: 前进**速度八十**, 以**五十**的速度前进
- Duration: 前进**三秒**, 走**五秒钟**

### Complex Trajectory Commands

| Command (Chinese) | Description |
|-------------------|-------------|
| 前后往返重复 | Forward-backward repeat |
| 左右旋转360度 | Spin left-right 360° |
| 对角斜线 | Diagonal line |
| 变速前进 | Variable speed forward |
| 全速紧急刹停 | Full speed emergency stop |
| 正方形轨迹 | Square trajectory |
| 圆形轨迹 | Circle trajectory |
| 三角形轨迹 | Triangle trajectory |
| 数字8轨迹 | Figure-8 trajectory |
| S型蜿蜒曲线 | S-curve trajectory |

### Camera Commands

| Command (Chinese) | Action |
|-------------------|--------|
| 拍照 / 照相 | Take photo |
| 录像 / 开始录像 | Start recording |
| 停止录像 | Stop recording |
| 打开摄像头 | Open camera stream |
| 关闭摄像头 | Close camera stream |

### Wake / Sleep Words

| Type | Words |
|------|-------|
| Wake | 机器人, 小车, 助手, wake up, hello, 你好 |
| Sleep | 睡觉, 休眠, 休息, sleep, pause, 再见 |

## Calibration Guide

To achieve accurate movement distances and turns:

1. **Distance calibration**: Say `机器人向前走一米` and measure actual distance
   - If actual < 1m: increase `--distance-factor`
   - If actual > 1m: decrease `--distance-factor`

2. **Turn calibration**: Say `机器人左转九十度` and measure actual turn angle
   - If turn > 90°: decrease `--turn-factor-left` / `--turn-factor-right`
   - If turn < 90°: increase `--turn-factor-left` / `--turn-factor-right`

## Robot API

The robot must expose this HTTP endpoint:

```
GET http://{robot_ip}/api/control?action={action}&speed={speed}[&time={ms}]
```

**Actions**: `up`, `down`, `left`, `right`, `stop`, `grab`, `release`

**Camera stream** (optional): `http://{robot_ip}/api/camera/stream`

## Program Files

| File | Description |
|------|-------------|
| `voiceCommandRobot.py` | Main program (latest version with streaming ASR, timing logs, 10s max segment) |
| `voiceCommandRobot_v5.py` | Alternative version with similar features |

## Architecture

```
Microphone (arecord) → Raw PCM → VAD → ASR (SenseVoice) → Command Parser → HTTP Robot API
                                            ↓
                                    Streaming Display (real-time interim results)
```

- **Audio capture**: `arecord` writes raw PCM to `/tmp/voicerobot_audio.raw`
- **VAD**: Frame-level energy-based voice activity detection
- **ASR**: SenseVoice Small model via FunASR
- **Command dispatch**: Non-blocking HTTP requests to robot

## Tips

- Use `--quiet` for cleaner output showing only transcribed commands
- Enable `--wake-word` in noisy environments to reduce false triggers
- Use `--no-streaming` if terminal display issues occur
- Adjust `--threshold` if speech is not detected or too much noise triggers it
- The program automatically suppresses duplicate commands within a 10-second window

## License

MIT License
