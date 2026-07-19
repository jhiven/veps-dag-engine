"""Root conftest.py.

Adds the project root to ``sys.path`` so that ``tests.support.*`` imports
resolve correctly within test sub-packages.  This file must live at the
project root, not inside the ``tests/`` directory, so that pytest picks it
up before collecting any test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `tests` importable as a top-level package when running pytest from
# the project root or from any sub-directory.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
