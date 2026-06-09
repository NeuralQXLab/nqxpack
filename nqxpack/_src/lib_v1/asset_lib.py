from abc import ABC, abstractmethod
from pathlib import Path

from zipfile import ZipFile

import jax
from flax import serialization

from nqxpack._src.contextmgr import current_context


class AssetManager(ABC):
    """
    Used to store binary blobs or large serialized objects, which may be needed
    when serializing some custom types.

    This is passed in as a parameter to the `serialize` and `deserialize` functions.
    """

    @abstractmethod
    def _write(self, key: str, value: bytes):
        """
        Commits a binary blob to the asset manager under key `key`.

        This should be implemented for the specific asset manager.
        """
        pass

    @abstractmethod
    def _read(self, key: str) -> bytes:
        """
        Reads a binary blob to the asset manager under key `key`.

        This should be implemented for the specific asset manager.
        """
        pass

    @abstractmethod
    def _has(self, key: str) -> bool:
        """
        Returns whether a binary blob exists under the resolved key `key`.

        This should be implemented for the specific asset manager.
        """
        pass

    def _resolve_key(self, key: str) -> str:
        """Map a logical ``path/asset_name`` key to the backend storage key.

        The default is the identity; backends that strip a ``remove_root`` or
        prepend a ``path`` prefix override this so that ``_write``/``_read``/
        ``_has`` all agree on the same resolved key.
        """
        return key

    def write_asset(self, asset_name, value: bytes, path: tuple[str, ...] = None):
        """
        Write an asset to the backend

        Args:
            asset_name: Name of the asset
            value: The asset to write, which should be a binary blob.
            path: Path to the asset, as a tuple of strings. This is optional.
        """
        if path is None:
            path = current_context().path
        return self._write(self._resolve_key(f"{path}/{asset_name}"), value)

    def read_asset(self, asset_name, path: tuple[str, ...] = None):
        if path is None:
            path = current_context().path
        return self._read(self._resolve_key(f"{path}/{asset_name}"))

    def has_asset(self, asset_name, path: tuple[str, ...] = None) -> bool:
        """
        Returns whether an asset exists in the backend.

        Lets a deserializer branch on the presence of an optional, self-describing
        payload (e.g. one written only when a save-time option was set) instead of
        reading back a persisted flag.

        Args:
            asset_name: Name of the asset
            path: Path to the asset, as a tuple of strings. This is optional.
        """
        if path is None:
            path = current_context().path
        return self._has(self._resolve_key(f"{path}/{asset_name}"))

    def write_msgpack(self, asset_name, value: dict, path: tuple[str, ...] = None):
        """
        Write a dictionary of msgpack-serializable objects to the asset manager.

        Args:
            asset_name: Name of the asset
            value: The asset to write, which should be a dictionary of msgpack-serializable objects.
            path: Path to the asset, as a tuple of strings. This is optional.
        """
        if jax.process_index() == 0:
            self.write_asset(asset_name, serialization.msgpack_serialize(value), path)

    def read_msgpack(self, asset_name, path: tuple[str, ...] = None):
        """
        Reads a dictionary of data serialized with msgpack to the asset manager.

        Args:
            asset_name: Name of the asset
            value: The asset to write, which should be a dictionary of msgpack-serializable objects.
            path: Path to the asset, as a tuple of strings. This is optional.
        """
        return serialization.msgpack_restore(self.read_asset(asset_name, path))


class InMemoryAssetManager(AssetManager):
    """
    Asset manager that stores assets in memory.
    """

    def __init__(self):
        self._assets = {}

    def _write(self, key: str, value: bytes):
        self._assets[key] = value

    def _read(self, key: str) -> bytes:
        return self._assets[key]

    def _has(self, key: str) -> bool:
        return key in self._assets


class FolderAssetManager(AssetManager):
    def __init__(self, folder, path, remove_root=None):
        """
        Constructs an asset manager backed by a folder.

        Args:
            folder: a directory to store the asset.
            root: A prefix to remove from all keys when writing to the archive. This is optional.
        """
        if not isinstance(folder, Path):
            folder = Path(folder)
        self.folder = folder
        self.path = path
        self.remove_root = remove_root

    def _resolve_key(self, key: str) -> str:
        if self.remove_root is not None and key.startswith(self.remove_root):
            key = key[len(self.remove_root) :]
        if self.path is not None:
            key = self.path + key
        return key

    def _write(self, key: str, value: bytes):
        if jax.process_index() == 0:
            if not (self.folder / key).parent.exists():
                (self.folder / key).parent.mkdir(parents=True)
            with open(self.folder / key, "wb") as f:
                f.write(value)

    def _read(self, key: str) -> bytes:
        with open(self.folder / key, "rb") as f:
            return f.read()

    def _has(self, key: str) -> bool:
        return (self.folder / key).exists()


class ArchiveAssetManager(AssetManager):
    """
    Asset manager that writes to a zip archive.
    """

    def __init__(
        self, archive: ZipFile, path: str | None = None, remove_root: str | None = None
    ):
        """
        Constructs an asset manager backed by a zip file archive.

        Args:
            archive: an open zip file object.
            remove_root: A prefix to remove from all keys when writing to the archive. This is optional.
            path: A prefix to add to all keys when writing to the archive. This is optional.
        """
        self.archive = archive
        self.path = path
        self.remove_root = remove_root

    def _resolve_key(self, key: str) -> str:
        if self.remove_root is not None and key.startswith(self.remove_root):
            key = key[len(self.remove_root) :]
        if self.path is not None:
            key = self.path + key
        return key

    def _write(self, key: str, value: bytes):
        if jax.process_index() == 0:
            with self.archive.open(key, "w") as f:
                f.write(value)

    def _read(self, key: str) -> bytes:
        if key not in self.archive.namelist():
            raise FileNotFoundError(f"Asset {key} not found in archive.")

        with self.archive.open(key, "r") as f:
            return f.read()

    def _has(self, key: str) -> bool:
        return key in self.archive.namelist()
