"""Utilities for parsing and handling version strings."""

import re


def parse_version(version: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Parse a version string or tuple into a tuple of (major, minor, patch).

    Args:
        version: Either a version string like "1.2.3" or a tuple like (1, 2, 3)

    Returns:
        A tuple of (major, minor, patch)

    Examples:
        >>> parse_version("1.2.3")
        (1, 2, 3)
        >>> parse_version("1.2")
        (1, 2, 0)
        >>> parse_version("1.2.3rc1")
        (1, 2, 3)
        >>> parse_version("")
        (0, 0, 0)
        >>> parse_version((1, 2, 3))
        (1, 2, 3)
    """
    if isinstance(version, tuple):
        return version

    if isinstance(version, str):
        if version == "":
            return (0, 0, 0)
        parts = version.split(".")
        # Handle versions with fewer than 3 parts
        while len(parts) < 3:
            parts.append("0")
        try:
            return tuple(int(p) for p in parts[:3])
        except ValueError:
            # If parsing fails (e.g., "1.2.3rc1"), try to extract just the numbers
            numbers = []
            for part in parts[:3]:
                match = re.match(r"(\d+)", part)
                if match:
                    numbers.append(int(match.group(1)))
                else:
                    numbers.append(0)
            while len(numbers) < 3:
                numbers.append(0)
            return tuple(numbers)

    return (0, 0, 0)
