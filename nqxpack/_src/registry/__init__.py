from nqxpack._src.registry.versioninfo import VERSION
from nqxpack._src.registry._loader import load_all_available

# stdlib has no external dependencies — always load it at import time.
from nqxpack._src.registry import stdlib  # noqa: F401
