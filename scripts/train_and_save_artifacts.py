"""
Legacy entry point: trains all three models and saves artifacts.
Redirects to train_all.py for backward compatibility.
Run: python scripts/train_and_save_artifacts.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_all import main

if __name__ == "__main__":
    main()
