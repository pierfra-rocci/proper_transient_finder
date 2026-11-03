#!/usr/bin/env python3
# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Run the standalone tiled survey-reference preparer from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proper_transient_finder.survey_mosaic import main

if __name__ == "__main__":
    raise SystemExit(main())
