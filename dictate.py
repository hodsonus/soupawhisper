#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper.
Hold the hotkey to record, release to transcribe and copy to clipboard.
"""

import argparse
import configparser
import subprocess
import tempfile
import threading
import signal
import sys
import os
from pathlib import Path

import selectors

import evdev
from evdev import ecodes
from faster_whisper import WhisperModel

__version__ = "0.1.0"

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"


def load_config():
    config = configparser.ConfigParser()

    # Defaults
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "key": "f12",
        "key_mode": "toggle",
        "max_record_seconds": "600",
        "auto_type": "true",
        "notifications": "true",
    }

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "key_mode": config.get("hotkey", "key_mode", fallback=defaults["key_mode"]),
        "max_record_seconds": config.getint("hotkey", "max_record_seconds", fallback=int(defaults["max_record_seconds"])),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
    }


CONFIG = load_config()


def get_hotkey_code(key_name):
    """Map key name to evdev key code."""
    key_name = key_name.lower()
    # Map common key names to evdev key codes
    key_map = {
        "f1": ecodes.KEY_F1, "f2": ecodes.KEY_F2, "f3": ecodes.KEY_F3,
        "f4": ecodes.KEY_F4, "f5": ecodes.KEY_F5, "f6": ecodes.KEY_F6,
        "f7": ecodes.KEY_F7, "f8": ecodes.KEY_F8, "f9": ecodes.KEY_F9,
        "f10": ecodes.KEY_F10, "f11": ecodes.KEY_F11, "f12": ecodes.KEY_F12,
        "scroll_lock": ecodes.KEY_SCROLLLOCK,
        "pause": ecodes.KEY_PAUSE,
        "insert": ecodes.KEY_INSERT,
        "home": ecodes.KEY_HOME,
        "end": ecodes.KEY_END,
    }
    if key_name in key_map:
        return key_map[key_name]
    # Try evdev KEY_ constant directly
    attr = f"KEY_{key_name.upper()}"
    if hasattr(ecodes, attr):
        return getattr(ecodes, attr)
    print(f"Unknown key: {key_name}, defaulting to f12")
    return ecodes.KEY_F12


HOTKEY = get_hotkey_code(CONFIG["key"])
HOTKEY_NAME = CONFIG["key"].upper()
MODEL_SIZE = CONFIG["model"]
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
KEY_MODE = CONFIG["key_mode"]  # "hold" or "toggle"
MAX_RECORD_SECONDS = CONFIG["max_record_seconds"]
AUTO_TYPE = CONFIG["auto_type"]
NOTIFICATIONS = CONFIG["notifications"]


class Dictation:
    def __init__(self):
        self.recording = False
        self.record_process = None
        self.record_timer = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        self.notify_id = None

        # Load model in background
        print(f"Loading Whisper model ({MODEL_SIZE})...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            self.model_loaded.set()
            print(f"Model loaded. Ready for dictation!")
            if KEY_MODE == "hold":
                print(f"Hold [{HOTKEY_NAME}] to record, release to transcribe.")
            else:
                print(f"Press [{HOTKEY_NAME}] to start/stop recording.")
            print("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            print(f"Failed to load model: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                print("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def _close_notification(self):
        """Close the current persistent notification via DBus."""
        if self.notify_id is not None:
            subprocess.run(
                [
                    "gdbus", "call", "--session",
                    "--dest", "org.freedesktop.Notifications",
                    "--object-path", "/org/freedesktop/Notifications",
                    "--method", "org.freedesktop.Notifications.CloseNotification",
                    str(self.notify_id),
                ],
                capture_output=True
            )
            self.notify_id = None

    def notify(self, title, message, icon="dialog-information", timeout=2000, persistent=False):
        """Send a desktop notification."""
        if not NOTIFICATIONS:
            return
        self._close_notification()
        cmd = [
            "notify-send",
            "-a", "SoupaWhisper",
            "-i", icon,
            "-t", str(timeout),
        ]
        if persistent:
            cmd += ["-u", "critical", "-p"]
        cmd += [title, message]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if persistent and result.stdout.strip():
            self.notify_id = int(result.stdout.strip())

    def start_recording(self):
        if self.recording or self.model_error:
            return

        self.recording = True
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        # Record using arecord (ALSA) - works on most Linux systems
        self.record_process = subprocess.Popen(
            [
                "arecord",
                "-f", "S16_LE",  # Format: 16-bit little-endian
                "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
                "-c", "1",       # Mono
                "-t", "wav",
                self.temp_file.name
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Auto-stop after max duration
        self.record_timer = threading.Timer(MAX_RECORD_SECONDS, self.stop_recording)
        self.record_timer.start()

        print("Recording...")
        if KEY_MODE == "hold":
            hint = f"Release {HOTKEY_NAME} when done"
        else:
            hint = f"Press {HOTKEY_NAME} again to stop"
        self.notify("Recording...", hint, "audio-input-microphone", 0, persistent=True)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_timer:
            self.record_timer.cancel()
            self.record_timer = None

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        print("Transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 0, persistent=True)

        # Wait for model if not loaded yet
        self.model_loaded.wait()

        if self.model_error:
            print(f"Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        # Transcribe
        try:
            segments, info = self.model.transcribe(
                self.temp_file.name,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            if text:
                # Save current clipboard contents
                old_clipboard = None
                try:
                    result = subprocess.run(
                        ["wl-paste", "--no-newline"],
                        capture_output=True, timeout=2
                    )
                    if result.returncode == 0:
                        old_clipboard = result.stdout
                except (subprocess.TimeoutExpired, OSError):
                    pass

                # Copy to clipboard using wl-copy (Wayland)
                process = subprocess.Popen(
                    ["wl-copy"],
                    stdin=subprocess.PIPE
                )
                process.communicate(input=text.encode())

                # Type it into the active input field
                if AUTO_TYPE:
                    subprocess.run(["wtype", text])

                # Restore previous clipboard contents
                if old_clipboard is not None:
                    restore = subprocess.Popen(
                        ["wl-copy"],
                        stdin=subprocess.PIPE
                    )
                    restore.communicate(input=old_clipboard)

                print(f"Copied: {text}")
                self.notify("Copied!", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                print("No speech detected")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            print(f"Error: {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            # Cleanup temp file
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)

    def _find_keyboards(self):
        """Find keyboard devices from /dev/input/."""
        keyboards = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                # Check if device has EV_KEY capability with actual key codes
                if ecodes.EV_KEY in caps:
                    key_codes = caps[ecodes.EV_KEY]
                    # Must have common keyboard keys (not just a mouse with buttons)
                    if ecodes.KEY_A in key_codes and ecodes.KEY_Z in key_codes:
                        keyboards.append(dev)
                    else:
                        dev.close()
                else:
                    dev.close()
            except (PermissionError, OSError):
                continue
        return keyboards

    def stop(self):
        print("\nExiting...")
        self.running = False
        os._exit(0)

    def run(self):
        keyboards = self._find_keyboards()
        if not keyboards:
            print("Error: No keyboard devices found.")
            print("Make sure your user is in the 'input' group:")
            print("  sudo usermod -aG input $USER")
            print("Then log out and back in.")
            sys.exit(1)

        print(f"Listening on: {', '.join(dev.name for dev in keyboards)}")

        selector = selectors.DefaultSelector()
        for dev in keyboards:
            selector.register(dev, selectors.EVENT_READ)

        try:
            while self.running:
                for key, mask in selector.select(timeout=1):
                    dev = key.fileobj
                    for event in dev.read():
                        if event.type == ecodes.EV_KEY and event.code == HOTKEY:
                            if KEY_MODE == "hold":
                                if event.value == 1:  # Key down
                                    self.start_recording()
                                elif event.value == 0:  # Key up
                                    self.stop_recording()
                            else:  # toggle
                                if event.value == 1:  # Key down
                                    if self.recording:
                                        self.stop_recording()
                                    else:
                                        self.start_recording()
        finally:
            selector.close()
            for dev in keyboards:
                dev.close()


def check_dependencies():
    """Check that required system commands are available."""
    missing = []

    for cmd in ["arecord", "wl-copy"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            pkg = "alsa-utils" if cmd == "arecord" else "wl-clipboard"
            missing.append((cmd, pkg))

    if AUTO_TYPE:
        if subprocess.run(["which", "wtype"], capture_output=True).returncode != 0:
            missing.append(("wtype", "wtype"))

    if missing:
        print("Missing dependencies:")
        for cmd, pkg in missing:
            print(f"  {cmd} - install with: sudo apt install {pkg}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SoupaWhisper - Push-to-talk voice dictation"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.parse_args()

    print(f"SoupaWhisper v{__version__}")
    print(f"Config: {CONFIG_PATH}")

    check_dependencies()

    dictation = Dictation()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
