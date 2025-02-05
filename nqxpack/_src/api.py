import orjson
import json

from pathlib import Path


from nqxpack._src.lib_v1 import (
    serialize_object,
    deserialize_object,
)
from nqxpack._src import distributed
from nqxpack._src.contextmgr import PackingContext
from nqxpack._src.metadata.generate_metadata import generate_metadata

from nqxpack._src.io import DirectoryArchive, ZipArchive
from nqxpack._src.errors import FutureVersionError
from nqxpack._src.lib_v1 import VERSION as LIB_VERSION
from nqxpack._src.registry import VERSION as REGISTRY_VERSION
from nqxpack._src.metadata.generate_metadata import VERSION as METADATA_VERSION

_CONFIG_FILENAME = "object.json"
_METADATA_FILENAME = "metadata.json"

_FORMAT_VERSION = 1.3

versioninfo_info = {
    "lib": LIB_VERSION,
    "registry": REGISTRY_VERSION,
    "metadata": METADATA_VERSION,
}


def save(object, path, *, zip: bool = True):
    """
    Saves an object to a file, using the NQXPack format.

    .. warning::

        The file will record the versions of installed packages whose objects are serialized to the file.
        A compatible version of those libraries will be required to load the file. By 'compatible', it is
        intended a version of libraries that is API-compatible with the version that was used to save the file.

        In general, we recomend to use the same version of the libraries that were used to save the file.

    The NQXPack format is a zip file that contains the following files:
    - metadata.json: Contains metadata about the file, including the format version and the versions of the
        libraries used to save the file.
    - object.json: A json file, in a format similar to Hydra's, that contains the information necessary to
        reconstruct the object.
    - assets/: A directory containing any object that cannot easily be serialized to json. This will generally
        contain large objects, like numpy arrays, stored into a binary blob.

    .. warning::

        The NQXPack format is in beta not guaranteed to be stable at the moment.

        It attempts to be backwards-compatible, but not forwards-compatible. This means that you can load
        files saved with older versions of NQXPack, but you cannot load files saved with newer versions. An
        informative error will be thrown in that case.
        However, the guarantee is, for the time-being, best-effort only.

        The NQXPack format is not secure. Do not load files from untrusted sources!

    Args:
        object: The object to save.
        path: The path to save the object to. If the path does not have a .nk extension, it will be added.
        zip: If True (default), the object will be saved in a zip file. If False, it will be saved in a directory.
    """
    if not isinstance(path, Path):
        path = Path(path)

    # Check that extension is .nk
    if path.suffix != ".nk":
        path = path.with_suffix(".nk")

    if zip:
        archive = ZipArchive(path, mode="w")
    else:
        archive = DirectoryArchive(path, mode="w")

    orjson_options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
    with archive:
        with PackingContext(
            asset_manager=archive.create_asset_manager(path="assets/"),
        ):
            object_json = serialize_object(object)

            if distributed.is_master_process():
                with archive.open(_CONFIG_FILENAME, "w") as f:
                    data = orjson.dumps(object_json, option=orjson_options)
                    f.write(data)

            metadata_json = generate_metadata(
                format_version=_FORMAT_VERSION, versioninfo=versioninfo_info
            )
            if distributed.is_master_process():
                with archive.open(_METADATA_FILENAME, "w") as f:
                    f.write(orjson.dumps(metadata_json, option=orjson_options))


def load(path):
    """
    Loads an nqxpack file.

    Args:
        object: The object to save.
        path: The path to save the object to. If the path does not have a .nk extension, it will be added.
        zip: If True (default), the object will be saved in a zip file. If False, it will be saved in a directory.
    """

    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists() and path.suffix != ".nk":
        new_path = path.with_suffix(".nk")
        if new_path.exists():
            path = new_path

    if path.is_dir() and (path / _METADATA_FILENAME).exists():
        archive = DirectoryArchive(path, mode="r")
    elif path.exists():
        archive = ZipArchive(path, mode="r")
    else:
        raise FileNotFoundError(f"File not found: {path}")

    with archive:
        if _METADATA_FILENAME not in archive:
            raise ValueError(
                f"Invalid file format: archive does not contain {_METADATA_FILENAME}"
            )

        metadata = archive.read(_METADATA_FILENAME)
        metadata = json.loads(metadata)

        # validate
        if metadata["format"] != "NetKet":
            raise ValueError("Invalid file format.")

        file_version = metadata["format_version"]
        if file_version > _FORMAT_VERSION:
            # Do not attempt to load files from future versions
            raise FutureVersionError(file_version, _FORMAT_VERSION)
        elif file_version <= 1.1:
            # v1.0 and v1.1 of the format did not have the asset manager, and where just storing the state object
            # directly in the zip file.
            # Also, the object was stored in a file named config.json, not object.json
            _CONFIG_FILENAME = "config.json"
            asset_manager = archive.create_asset_manager(remove_root="state/")
        else:
            _CONFIG_FILENAME = "object.json"
            asset_manager = archive.create_asset_manager(path="assets/")

        config = archive.read(_CONFIG_FILENAME)
        with PackingContext(asset_manager=asset_manager, metadata=metadata):

            state_obj_dict = json.loads(config)

            # Fix a but that appeared briefly
            # TODO: Put this somewhere reasoanbly
            if "_target_" in state_obj_dict:
                if state_obj_dict["_target_"] == "netket.vqs.mc.mc_state.state.MCState":
                    state_obj_dict["_target_"] = "#netket.vqs.mc.mc_state.state.MCState"

            state = deserialize_object(state_obj_dict)

    return state
