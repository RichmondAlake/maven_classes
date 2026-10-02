"""Compatibility entry point for the current interaction-only browser checks."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('check_interaction_browser.py')), run_name='__main__')
