"""Tests for per-call serialization options (see design/serialization_options_proposal.md)."""

import numpy as np
import pytest

import nqxpack
from nqxpack.registry import (
    register_serialization,
    SaveOption,
    LoadOption,
    current_context,
)

from .. import common


class Widget:
    """Tiny object with an optional heavy payload (`extra`) gated by a save option."""

    def __init__(self, value, extra=None):
        self.value = value
        self.extra = extra

    def __eq__(self, other):
        return (
            isinstance(other, Widget)
            and self.value == other.value
            and np.array_equal(self.extra, other.extra)
        )


def _serialize_widget(w):
    ctx = current_context()
    if ctx.option("test_include_extra"):
        ctx.asset_manager.write_msgpack("extra.msgpack", {"extra": np.asarray(w.extra)})
    return {"value": w.value}


def _deserialize_widget(obj):
    am = current_context().asset_manager
    w = Widget(obj["value"])
    if am.has_asset("extra.msgpack"):  # self-describing, no flag read back
        w.extra = am.read_msgpack("extra.msgpack")["extra"]
    return w


register_serialization(
    Widget,
    _serialize_widget,
    _deserialize_widget,
    options=[
        SaveOption(
            "test_include_extra",
            bool,
            default=False,
            doc="Store the optional `extra` payload.",
        ),
        LoadOption(
            "test_widget_policy",
            bool,
            default=False,
            doc="A defensive load-time flag (allowed to go unused).",
        ),
    ],
)


@common.skipif_distributed
@pytest.mark.parametrize("zip", [True, False])
def test_save_option_toggles_payload(tmp_path, zip):
    w = Widget(3, extra=np.arange(64.0))
    path = tmp_path / "w.nk"

    # Default: option off -> no payload, extra not recovered.
    nqxpack.save(w, path, zip=zip)
    assert nqxpack.load(path).extra is None

    # Opt in -> payload written, recovered on a plain load (self-describing).
    nqxpack.save(w, path, zip=zip, options={"test_include_extra": True})
    loaded = nqxpack.load(path)
    assert loaded == w
    np.testing.assert_array_equal(loaded.extra, w.extra)


@common.skipif_distributed
def test_unknown_option_suggests(tmp_path):
    with pytest.raises(KeyError, match="test_include_extra"):
        nqxpack.save(Widget(1), tmp_path / "w.nk", options={"test_include_xtra": True})


@common.skipif_distributed
def test_wrong_direction(tmp_path):
    with pytest.raises(ValueError, match="load-time option"):
        nqxpack.save(Widget(1), tmp_path / "w.nk", options={"test_widget_policy": True})


@common.skipif_distributed
def test_wrong_type(tmp_path):
    with pytest.raises(TypeError, match="expects bool"):
        nqxpack.save(
            Widget(1), tmp_path / "w.nk", options={"test_include_extra": "yes"}
        )


@common.skipif_distributed
def test_unused_save_option_warns(tmp_path):
    # No Widget is saved, so no serializer reads the option this call.
    with pytest.warns(UserWarning, match="no serializer used it"):
        nqxpack.save(
            np.arange(3), tmp_path / "x.nk", options={"test_include_extra": True}
        )


@common.skipif_distributed
def test_load_option_silent_when_unused(tmp_path, recwarn):
    nqxpack.save(Widget(1), tmp_path / "w.nk")
    nqxpack.load(tmp_path / "w.nk", options={"test_widget_policy": True})
    assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)


@common.skipif_distributed
def test_list_options():
    save_names = {s.name for s in nqxpack.list_options("save")}
    load_names = {s.name for s in nqxpack.list_options("load")}
    assert "test_include_extra" in save_names
    assert "test_widget_policy" in load_names
    assert "test_widget_policy" not in save_names


@common.skipif_distributed
def test_undeclared_option_read_raises(tmp_path):
    # Reading an option the dispatching serializer did not declare is a bug.
    def _bad_serialize(w):
        current_context().option("test_widget_policy")  # not declared here
        return {"value": w.value}

    class Gadget:
        def __init__(self, value):
            self.value = value

    register_serialization(Gadget, _bad_serialize, options=[])
    with pytest.raises(LookupError, match="did not declare it"):
        nqxpack.save(Gadget(1), tmp_path / "g.nk")
