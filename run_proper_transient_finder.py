#!/usr/bin/env python3
# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

"""Launch Proper Transient Finder from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    from proper_transient_finder import ProperTransientFinder, main
except ModuleNotFoundError as error:
    if error.name != "proper_transient_finder":
        raise
    raise SystemExit(
        "Astronomy pipeline is not installed. Run scripts/bootstrap.ps1; see USER_GUIDE.md."
    ) from error

__all__ = ["ProperTransientFinder", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
