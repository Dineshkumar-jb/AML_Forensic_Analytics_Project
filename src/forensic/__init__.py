"""Forensic Analytics package (Benford's Law, RSF, Duplicate detection, Velocity, Typologies)."""

from src.forensic.engine import run_forensic_suite
from src.forensic.benford import analyze_benford
from src.forensic.rsf import calculate_rsf
from src.forensic.duplicates import detect_duplicates
from src.forensic.velocity import analyze_velocity
from src.forensic.typologies import analyze_all_typologies

__all__ = [
    "run_forensic_suite",
    "analyze_benford",
    "calculate_rsf",
    "detect_duplicates",
    "analyze_velocity",
    "analyze_all_typologies",
]
