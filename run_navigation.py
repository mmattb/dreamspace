#!/usr/bin/env python3
"""Launcher for the animated dreamspace navigation example.

This script automatically sets up the Python path and runs the example.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import and run the example
if __name__ == "__main__":
    from examples.animated_navigation import main, main_with_args
    
    # Check if we have command line arguments (other than script name)
    if len(sys.argv) > 1:
        main_with_args()
    else:
        main()
