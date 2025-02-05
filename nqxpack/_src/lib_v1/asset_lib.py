from abc import ABC, abstractmethod
from pathlib import Path

from zipfile import ZipFile

from flax import serialization

from nqxpack._src.distributed import is_master_process


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

    def write_asset(self, asset_name, value: bytes, path: tuple[str, ...] = ()):
        """
        Write an asset to the backend

        Args:
            asset_name: Name of the asset
            value: The asset to write, which should be a binary blob.
            path: Path to the asset, as a tuple of strings. This is optional.
        """
        key = "/".join(path + (asset_name,))
        return self._write(key, value)

    def read_asset(self, asset_name, path: tuple[str, ...] = ()):
        key = "/".join(path + (asset_name,))
        return self._read(key)

    def write_msgpack(self, asset_name, value: dict, path: tuple[str, ...] = ()):
        """
        Write a dictionary of msgpack-serializable objects to the asset manager.

        Args:
            asset_name: Name of the asset
            value: The asset to write, which should be a dictionary of msgpack-serializable objects.
            path: Path to the asset, as a tuple of strings. This is optional.
        """
        if is_master_process():
            self.write_asset(asset_name, serialization.msgpack_serialize(value), path)

    def read_msgpack(self, asset_name, path: tuple[str, ...] = ()):
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

    def _write(self, key: str, value: bytes):
        if self.remove_root is not None:
            if key.startswith(self.remove_root):
                key = key[len(self.remove_root) :]
        if self.path is not None:
            key = self.path + key

        if is_master_process():
            if not (self.folder / key).parent.exists():
                (self.folder / key).parent.mkdir(parents=True)
            with open(self.folder / key, "wb") as f:
                f.write(value)

    def _read(self, key: str) -> bytes:
        if self.remove_root is not None:
            if key.startswith(self.remove_root):
                key = key[len(self.remove_root) :]
        if self.path is not None:
            key = self.path + key

        with open(self.folder / key, "rb") as f:
            return f.read()


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

    def _write(self, key: str, value: bytes):
        if self.remove_root is not None:
            if key.startswith(self.remove_root):
                key = key[len(self.remove_root) :]
        if self.path is not None:
            key = self.path + key
        if is_master_process():
            with self.archive.open(key, "w") as f:
                f.write(value)

    def _read(self, key: str) -> bytes:
        if self.remove_root is not None:
            if key.startswith(self.remove_root):
                key = key[len(self.remove_root) :]
        if self.path is not None:
            key = self.path + key

        if key not in self.archive.namelist():
            raise FileNotFoundError(f"Asset {key} not found in archive.")

        with self.archive.open(key, "r") as f:
            return f.read()
