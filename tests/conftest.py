from __future__ import annotations

import sys
from pathlib import Path

ARIA_ROOT = Path(__file__).resolve().parent.parent
if str(ARIA_ROOT) not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT))
