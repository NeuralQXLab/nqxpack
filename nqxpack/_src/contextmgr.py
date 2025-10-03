from typing import TYPE_CHECKING

import threading
from functools import wraps

if TYPE_CHECKING:
    from nqxpack._src.lib_v1.asset_lib import AssetManager

# Thread-local storage to keep track of the context per thread
_local = threading.local()

# Dictionary holding the versions of the packages used in the serialization
# until now. We cache it to speed up serialisation, and assume that package
# versions cannot change during runtime.
if not hasattr(_local, "packages"):
    _local.packages = {}


class PackingContext:
    def __init__(self, asset_manager: "AssetManager | None" = None, metadata=None):
        if metadata is None:
            metadata = {}

        self._path_stack = []  # Stack to track traversal paths
        self._metadata = metadata  # Storage for metadata
        self._asset_manager = (
            asset_manager  # Example asset manager (could be any object)
        )

    def enter_path(self, path):
        """Pushes a path onto the stack."""
        self._path_stack.append(path)

    def exit_path(self):
        """Pops a path from the stack."""
        if self._path_stack:
            self._path_stack.pop()

    @property
    def path(self):
        """Returns the current path (top of the stack) or None if empty."""
        path = "/".join(map(str, self._path_stack))
        return path

    @property
    def raw_path(self):
        """Returns the raw path stack."""
        return tuple(self._path_stack)

    @property
    def package_versions(self):
        return _local.packages

    @property
    def saved_file_package_versions(self):
        return self._metadata["packages"]

    def set_metadata(self, key, value):
        """Stores metadata in the context."""
        self._metadata[key] = value

    def get_metadata(self, key=None):
        """Retrieves stored metadata. If key is None, returns all metadata."""
        return self._metadata if key is None else self._metadata.get(key)

    @property
    def asset_manager(self) -> "AssetManager":
        """Retrieves the asset manager object."""
        return self._asset_manager

    def __enter__(self):
        set_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        clear_context()
        pass  # Nothing to clean up in this simple case


# Singleton access for the context manager
def current_context() -> PackingContext:
    if not hasattr(_local, "context"):
        _local.context = PackingContext()
    return _local.context


def set_context(context):
    _local.context = context


def clear_context():
    if hasattr(_local, "context"):
        del _local.context


def autopath(func):
    """Decorator to automatically manage context paths."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        __tracebackhide__ = True

        ctx = current_context()
        path = kwargs.pop("path", None)
        if path:
            ctx.enter_path(path)
        try:
            return func(*args, **kwargs)
        finally:
            if path:
                ctx.exit_path()

    return wrapper
