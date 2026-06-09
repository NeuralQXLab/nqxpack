"""Unit tests for ``AssetManager.has_asset`` and the ``_resolve_key`` refactor.

``has_asset`` lets a deserializer probe for an optional, self-describing payload.
The key point of the refactor is that ``_write``/``_read``/``_has`` all route the
logical ``path/asset_name`` key through the same ``_resolve_key`` hook, so they
agree on the backend key by construction.

The folder and archive backends share the exact same ``_resolve_key`` body; the
archive path is additionally exercised end-to-end by the ``zip=True`` integration
tests, so here we cover the in-memory and folder backends directly.
"""

import pytest

from nqxpack._src.lib_v1.asset_lib import (
    InMemoryAssetManager,
    FolderAssetManager,
)

from .. import common


def _make_inmemory(tmp_path, path, remove_root):
    # InMemory has the identity `_resolve_key`, so path/remove_root don't apply.
    return InMemoryAssetManager()


def _make_folder(tmp_path, path, remove_root):
    return FolderAssetManager(tmp_path, path=path, remove_root=remove_root)


# (path prefix, remove_root) combos that exercise `_resolve_key`.
RESOLUTIONS = [
    pytest.param(None, None, id="bare"),
    pytest.param("assets/", None, id="path"),
    pytest.param("assets/", "data/", id="path+remove_root"),
]

BACKENDS = [
    pytest.param(_make_inmemory, id="inmemory"),
    pytest.param(_make_folder, id="folder"),
]


@common.skipif_distributed
@pytest.mark.parametrize("path,remove_root", RESOLUTIONS)
@pytest.mark.parametrize("make", BACKENDS)
def test_write_has_read_agree(make, path, remove_root, tmp_path):
    am = make(tmp_path, path, remove_root)
    blob = b"payload"

    # Absent before writing.
    assert am.has_asset("blob.bin", path="data") is False

    # `_write`, `_has` and `_read` all resolve to the same backend key.
    am.write_asset("blob.bin", blob, path="data")
    assert am.has_asset("blob.bin", path="data") is True
    assert am.read_asset("blob.bin", path="data") == blob


@common.skipif_distributed
@pytest.mark.parametrize("make", BACKENDS)
def test_has_asset_false_for_missing(make, tmp_path):
    am = make(tmp_path, "assets/", None)
    am.write_asset("present.bin", b"x", path="data")
    assert am.has_asset("present.bin", path="data") is True
    assert am.has_asset("absent.bin", path="data") is False
