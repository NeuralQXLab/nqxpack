import threading
import importlib

# Maps logical name → (registry module path, guard package to probe).
# guard=None means no external dependency; always load.
_BUILTIN = {
    "stdlib": ("nqxpack._src.registry.stdlib", None),
    "jax": ("nqxpack._src.registry.jax", "jax"),
    "flax": ("nqxpack._src.registry.flax", "flax"),
    "hydra": ("nqxpack._src.registry.hydra", "omegaconf"),
    "netket": ("nqxpack._src.registry.netket", "netket"),
    "netket_operator": ("nqxpack._src.registry.netket_operator", "netket"),
}

_initialized = False
_lock = threading.Lock()


def load_all_available() -> None:
    """
    Load every built-in registry module whose guard package is installed.
    Safe to call multiple times — subsequent calls return immediately.
    """
    global _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        for _module_path, _guard in _BUILTIN.values():
            if _guard is not None:
                try:
                    importlib.import_module(_guard)
                except ImportError:
                    continue  # package not installed; skip silently
            importlib.import_module(_module_path)
        _initialized = True
