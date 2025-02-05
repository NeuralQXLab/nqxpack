from pathlib import Path

from zipfile import ZipFile
import shutil


from nqxpack._src.lib_v1.asset_lib import ArchiveAssetManager, FolderAssetManager
from nqxpack._src.distributed import is_master_process


class ZipArchive:
    """
    Asset manager that writes to a zip archive.
    """

    def __init__(
        self,
        path,
        *args,
        mode="w",
        **kwargs,
    ):
        """
        Constructs an asset manager backed by a zip file archive.

        Args:
            archive: an open zip file object.
            remove_root: A prefix to remove from all keys when writing to the archive. This is optional.
            path: A prefix to add to all keys when writing to the archive. This is optional.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if mode == "w" and not is_master_process():
            self._archive = None
        else:
            # Remove the directory and all its contents
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
            self._archive = ZipFile(path, *args, mode=mode, **kwargs)

    def __enter__(self):
        if self._archive is None:
            return self
        self._archive.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._archive.__exit__(exc_type, exc_value, traceback)

    def __contains__(self, key):
        return key in self._archive.namelist()

    def open(self, key, mode):
        return self._archive.open(key, mode)

    def read(self, key):
        return self._archive.read(key)

    def write(self, key, value):
        return self._archive.write(key, value)

    def close(self):
        self._archive.close()

    def create_asset_manager(self, *args, **kwargs):
        return ArchiveAssetManager(self._archive, *args, **kwargs)


class DirectoryArchive:
    """
    Asset manager that writes to a zip archive.
    """

    def __init__(self, path, mode="w"):
        """
        Constructs an asset manager backed by a zip file archive.

        Args:
            archive: an open zip file object.
            remove_root: A prefix to remove from all keys when writing to the archive. This is optional.
            path: A prefix to add to all keys when writing to the archive. This is optional.
        """
        self.path = Path(path)
        if is_master_process():
            if mode == "w":
                # Remove the directory and all its contents
                if self.path.exists() and self.path.is_dir():
                    shutil.rmtree(self.path)
                elif self.path.exists():
                    self.path.unlink()
                self.path.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def __contains__(self, key):
        return (self.path / key).exists()

    def open(self, key, mode):
        mode = mode + "b"
        path = self.path / key
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        return path.open(mode)

    def read(self, key):
        path = self.path / key

        return path.read_bytes()

    def write(self, key, value):
        path = self.path / key
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        return path.write(value)

    def close(self):
        self.path.close()

    def create_asset_manager(self, *args, **kwargs):
        return FolderAssetManager(self.path, *args, **kwargs)
