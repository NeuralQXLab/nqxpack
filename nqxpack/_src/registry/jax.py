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
    return {
        "node_data": obj.node_data(),
        "children": obj.children(),
    }


def deserialize_PyTreeDef(obj):
    return jax.tree_util.PyTreeDef.make_from_node_data_and_children(
        jax.tree_util.default_registry, obj["node_data"], obj["children"]
    )


register_serialization(
    jax.tree_util.PyTreeDef, serialize_PyTreeDef, deserialize_PyTreeDef
)
