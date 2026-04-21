import threading
import importlib
import importlib.metadata
import warnings

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

# Third-party packages register serializers by declaring an entry point in
# this group pointing to a module whose import has the side effect of
# registering their types:
#
#   [project.entry-points."nqxpack_registry"]
#   mypackage = "mypackage._nqxpack_registry"
#
ENTRY_POINT_GROUP = "nqxpack_registry"

_initialized = False
_lock = threading.Lock()


def load_all_available() -> None:
    """
    Load every built-in registry module whose guard package is installed,
    plus any third-party registries declared via the 'nqxpack_registry'
    entry point group.

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

        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            # ep.name is the namespace/guard: the package whose types this
            # registry handles. Skip if that package is not installed.
            try:
                importlib.import_module(ep.name)
            except ImportError:
                continue
            try:
                ep.load()
            except Exception as exc:
                warnings.warn(
                    f"nqxpack: failed to load registry plugin '{ep.name}' "
                    f"from '{ep.value}': {exc}",
                    ImportWarning,
                    stacklevel=2,
                )

        _initialized = True
