"""Regression test for lazy / co-located deserializer registration.

When a custom deserializer is registered as an import side effect of the module
that defines the class (the pattern produced by ``register_serialization`` with
a ``deserialization_fun``), the load path must still find it. On save the
instance's module is already imported, so the registration has run; on load
nqxpack only has the ``_target_`` qualname. ``deserialize_custom_object`` must
therefore import the module (resolving the qualname) and *then* re-check the
versioned registry, instead of giving up and falling back to the default
constructor-based deserialization.
"""

import sys
import textwrap

from nqxpack._src.lib_v1 import (
    serialize_object,
    deserialize_object,
)

from .. import common


# A module whose *import* has the side effect of registering a custom
# serializer/deserializer for the class it defines. The deserializer tags the
# instance so we can tell it ran (instead of the default constructor path).
_MODULE_SRC = textwrap.dedent(
    """
    from nqxpack.registry import register_serialization


    class LazilyRegistered:
        def __init__(self, value, via=None):
            self.value = value
            self.via = via


    def _serialize(obj):
        return {"value": obj.value}


    def _deserialize(data):
        obj = LazilyRegistered(data["value"], via="custom")
        return obj


    register_serialization(
        LazilyRegistered,
        _serialize,
        _deserialize,
    )
    """
)


@common.skipif_distributed
def test_lazy_colocated_deserializer_is_used_on_load(tmp_path, monkeypatch):
    mod_name = "lazy_reg_fixture_mod"
    (tmp_path / f"{mod_name}.py").write_text(_MODULE_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))

    # Import the module to serialize an instance (mirrors save: the class's
    # module is necessarily imported because we hold an instance).
    import importlib

    mod = importlib.import_module(mod_name)
    obj = mod.LazilyRegistered(42)
    blob = serialize_object(obj)

    # Now simulate a *fresh* load process: the module has never been imported,
    # so its co-located registration has not run yet. We must drop both the
    # module and its registry entry to reproduce the bug.
    from nqxpack._src.lib_v1.versioned_registry import (
        VERSIONED_DESERIALIZATION_REGISTRY,
    )
    from nqxpack._src.lib_v1.custom_types import TYPE_SERIALIZATION_REGISTRY

    target = f"{mod_name}.LazilyRegistered"
    VERSIONED_DESERIALIZATION_REGISTRY._registry.pop(target, None)
    TYPE_SERIALIZATION_REGISTRY.pop(mod.LazilyRegistered, None)
    sys.modules.pop(mod_name, None)

    assert target not in VERSIONED_DESERIALIZATION_REGISTRY

    # Loading must re-import the module (running the registration) and then use
    # the custom deserializer rather than the default constructor path.
    restored = deserialize_object(blob)

    assert restored.value == 42
    assert restored.via == "custom"  # proves the custom deserializer fired
