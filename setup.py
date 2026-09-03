#!/usr/bin/env python3
"""
RAG It! — One-Click Setup Script
=================================
Automates the full environment setup for first-time users.

Usage:
    python setup.py

What this script does:
    1. Checks Python version (3.11+ required)
    2. Creates a virtual environment (.venv)
    3. Upgrades pip
    4. Installs all requirements from requirements.txt
    5. Creates required data directories
    6. Checks for Ollama installation
    7. Checks for model weights (Visualized_m3.pth)
    8. Prints a final status summary
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


# ============================================================
# Colours for terminal output
# ============================================================

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  [OK]{RESET} {msg}")
def warn(msg): print(f"{YELLOW}  [!]{RESET}  {msg}")
def err(msg):  print(f"{RED}  [ERROR]{RESET} {msg}")
def info(msg): print(f"  {msg}")
def header(msg):
    print(f"\n{BOLD}{msg}{RESET}")
    print("─" * 60)


# ============================================================
# Paths (relative to this script's directory)
# ============================================================

PROJECT_ROOT  = Path(__file__).resolve().parent
REQUIREMENTS  = PROJECT_ROOT / "requirements.txt"
MODELS_DIR    = PROJECT_ROOT / "models"
WEIGHTS_FILE  = MODELS_DIR / "Visualized_m3.pth"
VENV_DIR      = PROJECT_ROOT / ".venv"

IS_WINDOWS    = platform.system() == "Windows"

if IS_WINDOWS:
    PYTHON_BIN = VENV_DIR / "Scripts" / "python.exe"
    PIP_BIN    = VENV_DIR / "Scripts" / "pip.exe"
else:
    PYTHON_BIN = VENV_DIR / "bin" / "python"
    PIP_BIN    = VENV_DIR / "bin" / "pip"


# ============================================================
# Helpers
# ============================================================

def run(cmd: list, cwd=None, check=True):
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        err(f"Command failed: {' '.join(str(c) for c in cmd)}")
        print(result.stderr[-2000:])
        sys.exit(1)
    return result


def run_pip(args: list):
    return run([str(PIP_BIN)] + args)


# ============================================================
# Step 1 — Python version check
# ============================================================

def check_python():
    header("Step 1: Checking Python version")
    major, minor = sys.version_info.major, sys.version_info.minor
    info(f"Detected Python {major}.{minor}.{sys.version_info.micro}")
    if major < 3 or (major == 3 and minor < 11):
        err(f"Python 3.11+ is required. You have {major}.{minor}.")
        err("Download from https://python.org")
        sys.exit(1)
    ok(f"Python {major}.{minor} is supported.")


# ============================================================
# Step 2 — Create virtual environment
# ============================================================

def create_venv():
    header("Step 2: Creating virtual environment (.venv)")
    if VENV_DIR.exists():
        warn(".venv already exists — skipping creation.")
        return
    info("Creating .venv ...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    ok(".venv created successfully.")


# ============================================================
# Step 3 — Upgrade pip
# ============================================================

def upgrade_pip():
    header("Step 3: Upgrading pip")
    run_pip(["install", "--upgrade", "pip", "--quiet"])
    ok("pip is up to date.")


# ============================================================
# Step 4 — Install requirements
# ============================================================

def install_requirements():
    header("Step 4: Installing Python dependencies (requirements.txt)")
    if not REQUIREMENTS.exists():
        err(f"requirements.txt not found at: {REQUIREMENTS}")
        sys.exit(1)
    info("This may take several minutes on first run (PyTorch, Docling, etc.)")
    info("Downloading & installing packages ...")
    run_pip([
        "install",
        "-r", str(REQUIREMENTS),
        "--quiet",
    ])
    ok("All Python dependencies installed.")


# ============================================================
# Step 5 — Create required directories
# ============================================================

def create_directories():
    header("Step 5: Creating required directories")
    dirs = [
        PROJECT_ROOT / "documents",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "qdrant_data",
        PROJECT_ROOT / "data" / "parsed",
        PROJECT_ROOT / "data" / "images",
        PROJECT_ROOT / "data" / "tables",
        PROJECT_ROOT / "data" / "metadata",
        PROJECT_ROOT / "data" / "chunks",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        ok(f"Created: {d.relative_to(PROJECT_ROOT)}")


# ============================================================
# Step 6 — Check Ollama
# ============================================================

def check_ollama():
    header("Step 6: Checking Ollama installation")
    result = run(["ollama", "--version"], check=False)
    if result.returncode == 0:
        ok(f"Ollama found: {result.stdout.strip()}")
        info("Remember to pull a model: ollama pull qwen3.5:4b-q4_K_M")
    else:
        warn("Ollama not found in PATH.")
        warn("Download and install from: https://ollama.com")
        warn("After install, run: ollama pull qwen3.5:4b-q4_K_M")


# ============================================================
# Step 7 — Check model weights
# ============================================================

def check_weights():
    header("Step 7: Checking Visualized-BGE model weights")
    if WEIGHTS_FILE.exists():
        size_gb = WEIGHTS_FILE.stat().st_size / (1024 ** 3)
        ok(f"Visualized_m3.pth found ({size_gb:.2f} GB)")
    else:
        warn(f"Model weights NOT found at: {WEIGHTS_FILE}")
        warn("Download Visualized_m3.pth (~1.74 GB) from:")
        warn("  https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth")
        warn(f"Place it at: {WEIGHTS_FILE}")


# ============================================================
# Final summary
# ============================================================

def print_summary():
    header("Setup Complete!")
    print(f"""
{BOLD}To activate the virtual environment:{RESET}""")
    if IS_WINDOWS:
        print("  .venv\\Scripts\\Activate.ps1       (PowerShell)")
        print("  .venv\\Scripts\\activate.bat        (Command Prompt)")
    else:
        print("  source .venv/bin/activate")

    print(f"""
{BOLD}To launch the RAG It! desktop app:{RESET}
  python ui/main.py

{BOLD}To ingest documents via CLI:{RESET}
  python scripts/ingest.py

{BOLD}To run the interactive CLI chat:{RESET}
  python tests/test_rag.py

{GREEN}Happy RAG-ing! 🚀{RESET}
""")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  RAG It! — Setup Script{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    check_python()
    create_venv()
    upgrade_pip()
    install_requirements()
    create_directories()
    check_ollama()
    check_weights()
    print_summary()
