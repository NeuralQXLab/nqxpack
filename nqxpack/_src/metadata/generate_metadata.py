import time
import socket
import sys
import os


import jax


from nqxpack._src.contextmgr import current_context

VERSION = 1.1
METADATA_VERSION = VERSION


def generate_metadata(format_version, versioninfo, extra_metadata=None):
    format_version = format_version  # ".".join(map(str, format_version))
    # versioninfo = {k: ".".join(map(str, v)) for k, v in versioninfo.items()}
    versioninfo["metadata"] = METADATA_VERSION  # ".".join(map(str, METADATA_VERSION))

    package_versions = current_context().package_versions

    if extra_metadata is None:
        extra_metadata = {}

    host_info = get_host_info()

    return {
        "format": "NetKet",
        "format_version": format_version,
        "versions": versioninfo,
        "packages": package_versions,
        "host": host_info,
        "extra": extra_metadata,
    }


def get_host_info():
    return {
        "python_version": sys.version,
        "time": time.time(),
        "hostname": socket.gethostname(),
        "n_devices": jax.device_count(),
        "uname": " ".join(os.uname()),
    }


def validate_metadata(metadata):
    if metadata["format"] != "NetKet":
        raise ValueError(f"Invalid metadata format: {metadata['format']}")
