import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "data"):
    _path = os.path.join(_ROOT, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
