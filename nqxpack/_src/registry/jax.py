from nqxpack._src.lib_v1.closure import register_closure_simple_serialization
from nqxpack._src.lib_v1.custom_types import register_serialization

import jax

register_closure_simple_serialization(
    jax.nn.initializers.normal,
    "init",
    original_qualname="jax._src.nn.initializers.normal",
)
register_closure_simple_serialization(
    jax.nn.initializers.variance_scaling,
    "init",
    original_qualname="jax._src.nn.initializers.variance_scaling",
)


def serialize_PyTreeDef(obj):
    # __getstate__ returns (registry, data). We only serialize the data part
    # since we'll use the default_registry during deserialization
    _registry, data = obj.__getstate__()
    return {"data": data}


def deserialize_PyTreeDef(serialized):
    obj = jax.tree_util.PyTreeDef.__new__(jax.tree_util.PyTreeDef)
    # Reconstruct state using default_registry
    obj.__setstate__((jax.tree_util.default_registry, serialized["data"]))
    return obj


register_serialization(
    jax.tree_util.PyTreeDef, serialize_PyTreeDef, deserialize_PyTreeDef
)
