"""Project root conftest - makes `core`, `environment`, etc. importable
when running scripts directly (python benchmarks/xxx.py) as well as
under pytest."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
