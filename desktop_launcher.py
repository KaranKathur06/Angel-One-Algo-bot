"""
Desktop launcher for the Angel One SmartAPI RSI Trading Bot.

Goals:
- Double-clickable .exe (packaged via PyInstaller with --onefile --noconsole)
- Starts Streamlit programmatically on a dynamic port
- Opens the UI automatically (pywebview if available, else default browser)
- Loads credentials from .env placed next to the .exe; if missing, prompts once
- Never logs or prints secrets
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv


# ------------------------- Path helpers ------------------------- #
def get_base_dir() -> Path:
    """Return the directory that contains the Python files (inside _MEIPASS when frozen)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def get_user_dir() -> Path:
    """
    Return the directory where the user keeps the .exe (and where .env/config.json should live).
    For frozen builds this is the folder containing the .exe; for dev runs, fall back to base_dir.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return get_base_dir()


# ------------------------- Environment handling ------------------------- #
REQUIRED_ENV_KEYS = ["SMART_API_KEY", "SMART_API_CLIENT_ID", "SMART_API_PIN", "SMART_API_TOTP_SECRET"]


def prompt_for_env(env_path: Path) -> None:
    """
    Prompt the user for required SmartAPI credentials using a minimal Tkinter form.
    Writes the resulting .env next to the executable.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        # If Tk is unavailable, raise a clear error so the user can install it.
        raise RuntimeError("Tkinter is required to capture credentials on first run.")

    root = tk.Tk()
    root.title("Angel One SmartAPI Setup")
    root.geometry("420x260")
    root.resizable(False, False)

    fields = {
        "SMART_API_KEY": ("API Key", ""),
        "SMART_API_CLIENT_ID": ("Client ID", ""),
        "SMART_API_PIN": ("PIN (will be masked)", ""),
        "SMART_API_TOTP_SECRET": ("TOTP Secret (Base32)", ""),
    }

    entries: Dict[str, tk.Entry] = {}

    tk.Label(root, text="Enter your SmartAPI credentials", font=("Segoe UI", 11, "bold")).pack(pady=(10, 5))
    form = tk.Frame(root)
    form.pack(padx=16, pady=8, fill="x")

    for idx, (env_key, (label, default)) in enumerate(fields.items()):
        tk.Label(form, text=label, anchor="w").grid(row=idx, column=0, sticky="w", pady=4)
        entry = tk.Entry(form, show="*" if "PIN" in env_key or "SECRET" in env_key else "", width=40)
        entry.insert(0, default)
        entry.grid(row=idx, column=1, sticky="ew", pady=4)
        entries[env_key] = entry
    form.columnconfigure(1, weight=1)

    status_var = tk.StringVar(value="Your secrets stay local on this machine.")
    status_label = tk.Label(root, textvariable=status_var, fg="#666")
    status_label.pack(pady=(6, 0))

    def on_save() -> None:
        values = {k: v.get().strip() for k, v in entries.items()}
        missing = [k for k, v in values.items() if not v]
        if missing:
            status_var.set(f"Missing: {', '.join(missing)}")
            status_label.config(fg="#b00")
            return

        lines = [
            "SMART_API_ENABLED=true",
            f"SMART_API_KEY={values['SMART_API_KEY']}",
            f"SMART_API_CLIENT_ID={values['SMART_API_CLIENT_ID']}",
            f"SMART_API_PIN={values['SMART_API_PIN']}",
            f"SMART_API_TOTP_SECRET={values['SMART_API_TOTP_SECRET']}",
        ]
        try:
            env_path.write_text("\n".join(lines) + "\n")
            messagebox.showinfo("Saved", f"Credentials saved to {env_path.name}. Keep this file safe.")
            root.destroy()
        except Exception as exc:  # pragma: no cover - UI only path
            status_var.set(f"Failed to save: {exc}")
            status_label.config(fg="#b00")

    tk.Button(root, text="Save & Launch", command=on_save, width=18).pack(pady=(12, 8))
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


def ensure_env_file() -> Path:
    """
    Ensure .env exists next to the executable and contains required keys.
    If missing, prompt the user once. Secrets are never printed to stdout/stderr.
    """
    user_dir = get_user_dir()
    env_path = user_dir / ".env"

    # Load existing values if present
    load_dotenv(env_path, override=False)
    missing = [key for key in REQUIRED_ENV_KEYS if not os.getenv(key)]
    if missing:
        prompt_for_env(env_path)
        load_dotenv(env_path, override=False)
        still_missing = [key for key in REQUIRED_ENV_KEYS if not os.getenv(key)]
        if still_missing:
            raise RuntimeError(f"Missing required credentials: {', '.join(still_missing)}")

    return env_path


# ------------------------- Single-instance lock helpers ------------------------- #
LOCKFILE_NAME = "AngelOneTradingBot.lock"
FIXED_PORT = 8501


def get_lock_path() -> Path:
    """Return the OS temp-based lock file path for single-instance enforcement."""
    return Path(tempfile.gettempdir()) / LOCKFILE_NAME


def acquire_single_instance_lock() -> Optional[int]:
    """
    Create a lock file with O_EXCL semantics in the OS temp directory.
    Returns the file descriptor if acquired, else None if another instance created it first.
    The fd must stay open for the life of the process.
    """
    lock_path = get_lock_path()
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    try:
        fd = os.open(lock_path, flags)
        # Optionally write our PID for debugging
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        except Exception:
            pass
        return fd
    except FileExistsError:
        # Another instance already holds the lock
        return None
    except Exception:
        # On any other failure, play safe and refuse to start another instance
        return None


def release_single_instance_lock(lock_fd: Optional[int]) -> None:
    """Remove the lock file and close the descriptor."""
    lock_path = get_lock_path()
    if lock_fd is not None:
        try:
            os.close(lock_fd)
        except Exception:
            pass
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


# ------------------------- Streamlit bootstrap ------------------------- #


def start_streamlit(app_path: Path, work_dir: Path, port: int) -> subprocess.Popen:
    """Launch Streamlit as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--server.runOnSave",
        "false",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]

    log_path = work_dir / "launcher.log"
    stdout = log_path.open("a", encoding="utf-8")
    # Keep stderr with stdout to avoid console windows; logs stay local
    process = subprocess.Popen(cmd, cwd=work_dir, stdout=stdout, stderr=subprocess.STDOUT)
    return process


def wait_for_server(url: str, timeout: int = 45) -> bool:
    """Poll the URL until it responds or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.8)
    return False


def open_ui(url: str) -> None:
    """
    Open the UI using pywebview if available; otherwise fall back to the default browser.
    pywebview provides a clean desktop window with no address bar.
    """
    try:
        import webview

        window = webview.create_window("Angel One Trading Bot", url)
        webview.start(gui="qt", debug=False)
        return
    except Exception:
        # Fall back to system browser
        webbrowser.open(url, new=1, autoraise=True)


# ------------------------- Shutdown handling ------------------------- #
def register_shutdown(
    proc: subprocess.Popen,
    lock_fd: Optional[int],
) -> None:
    """Terminate the Streamlit subprocess when the launcher exits."""
    def _stop() -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        release_single_instance_lock(lock_fd)

    atexit.register(_stop)

    def _handle_signal(signum: int, frame: Optional[object]) -> None:  # pragma: no cover - signal path
        _stop()
        sys.exit(0)

    for sig in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _handle_signal)


# ------------------------- Main entry ------------------------- #
def main() -> None:
    # Acquire single-instance lock in OS temp directory.
    # If already locked, exit immediately (no new server, no new browser tab).
    lock_fd = acquire_single_instance_lock()
    if lock_fd is None:
        # Another instance is already running; behave like a no-op.
        return

    base_dir = get_base_dir()
    user_dir = get_user_dir()

    # Ensure secrets are available before starting the server
    ensure_env_file()
    # Prefer bundled copy (PyInstaller _MEIPASS), fallback to folder with the exe
    candidates = [
        base_dir / "streamlit_app.py",
        user_dir / "streamlit_app.py",
    ]
    app_path = next((p for p in candidates if p.exists()), None)
    if app_path is None:
        raise FileNotFoundError(
            "streamlit_app.py not found. Ensure it is packaged with the exe "
            "or placed alongside the .exe file."
        )

    port = FIXED_PORT
    url = f"http://127.0.0.1:{port}"

    proc = start_streamlit(app_path, user_dir, port)
    register_shutdown(proc, lock_fd)

    # Wait for server readiness asynchronously while opening UI ASAP
    ready_flag = threading.Event()

    def _wait_and_open() -> None:
        if wait_for_server(url):
            ready_flag.set()
        open_ui(url)

    threading.Thread(target=_wait_and_open, daemon=True).start()

    # Keep the launcher alive while the Streamlit process is running
    try:
        while proc.poll() is None:
            time.sleep(1)
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()

