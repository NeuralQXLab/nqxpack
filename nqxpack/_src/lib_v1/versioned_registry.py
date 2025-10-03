from typing import Callable, Any
from nqxpack._src.contextmgr import current_context


class VersionedDeserializationRegistry:
    """
    Registry for versioned deserialization functions.

    Stores deserialization functions keyed by class path strings, with support
    for multiple versions of the same class.

    Each entry maps a class path string to a list of tuples containing:
    - minimum version (inclusive) as a tuple (major, minor, patch)
    - deserialization function

    When deserializing, the registry selects the appropriate function based on
    the package version stored in the context.
    """

    def __init__(self):
        # Dict[str, List[Tuple[Tuple[int, int, int], Callable]]]
        self._registry = {}

    def register(
        self,
        class_path: str,
        deserialization_fun: Callable[[dict], Any],
        min_version: tuple[int, int, int] = (0, 0, 0),
    ):
        """
        Register a deserialization function for a specific class path and version range.

        Args:
            class_path: The fully qualified class path as a string (e.g., "package.module.Class")
            deserialization_fun: Function that takes a dict and returns an instance
            min_version: Minimum version (inclusive) for which this deserializer is valid
        """
        if class_path not in self._registry:
            self._registry[class_path] = []

        # Insert in sorted order by version (highest to lowest)
        entry = (min_version, deserialization_fun)
        entries = self._registry[class_path]

        # Find insertion point - keep sorted by version descending
        insert_idx = 0
        for i, (existing_version, _) in enumerate(entries):
            if min_version >= existing_version:
                insert_idx = i
                break
            insert_idx = i + 1

        entries.insert(insert_idx, entry)

    def get(self, class_path: str, package_name: str = None) -> Callable | None:
        """
        Get the appropriate deserialization function for a class path.

        Args:
            class_path: The fully qualified class path as a string
            package_name: Name of the package to check version for. If None,
                         extracts from class_path

        Returns:
            The deserialization function, or None if not found
        """
        if class_path not in self._registry:
            return None

        # Extract package name from class path if not provided
        if package_name is None:
            # Assume format "package.module.Class"
            parts = class_path.split(".")
            if len(parts) > 0:
                package_name = parts[0]
            else:
                package_name = None

        # Get the version from the saved file metadata
        # The context manager already returns parsed tuples
        ctx = current_context()
        try:
            saved_versions = ctx.saved_file_package_versions
            file_version = saved_versions.get(package_name, (0, 0, 0))
        except (AttributeError, KeyError, TypeError):
            # If no version info available, use (0, 0, 0)
            file_version = (0, 0, 0)

        # Find the appropriate deserializer
        # Entries are sorted by version descending, so we find the first one
        # where min_version <= file_version
        for min_version, deserialization_fun in self._registry[class_path]:
            if file_version >= min_version:
                return deserialization_fun

        # No suitable version found
        return None

    def __contains__(self, class_path: str) -> bool:
        """Check if a class path is registered."""
        return class_path in self._registry


# Global instance
VERSIONED_DESERIALIZATION_REGISTRY = VersionedDeserializationRegistry()
