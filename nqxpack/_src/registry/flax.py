import jax

import flax
from flax import nnx
from flax import serialization

from nqxpack._src.lib_v1.custom_types import (
    register_automatic_serialization,
)
from nqxpack._src.lib_v1 import register_serialization
from nqxpack._src.contextmgr import current_context


# Graph

# flax.nnx
from flax.nnx.graph import HashableMapping

register_automatic_serialization(HashableMapping, mapping="_mapping")

flax_version = tuple(int(x) for x in flax.__version__.split("."))[:3]
if flax_version >= (0, 10, 4):
    from flax.nnx.graph import IndexesPytreeDef

    register_automatic_serialization(IndexesPytreeDef, *IndexesPytreeDef._fields)

# Sequence
register_automatic_serialization(nnx.Sequential, _args_="layers")


def serialize_linear(
    layer: nnx.Linear,
) -> dict:
    asset_manager = current_context().asset_manager
    _, data = nnx.split(layer)
    state_dict = serialization.to_state_dict(data.to_pure_dict())

    asset_manager.write_msgpack("state.msgpack", state_dict)

    return {
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "use_bias": layer.use_bias,
        "dtype": layer.dtype,
        "param_dtype": layer.param_dtype,
        "precision": layer.precision,
        "kernel_init": layer.kernel_init,
        "bias_init": layer.bias_init,
        "dot_general": layer.dot_general,
        "promote_dtype": layer.promote_dtype,
        # "preferred_element_type": layer.preferred_element_type,
    }


def deserialize_linear(obj: dict) -> nnx.Linear:
    asset_manager = current_context().asset_manager

    layer = nnx.Linear(
        obj["in_features"],
        obj["out_features"],
        use_bias=obj["use_bias"],
        dtype=obj.get("dtype", jax.numpy.float32),
        param_dtype=obj.get("param_dtype", jax.numpy.float32),
        precision=obj.get("precision", None),
        kernel_init=obj.get("kernel_init", nnx.initializers.lecun_normal()),
        bias_init=obj.get("bias_init", nnx.initializers.zeros),
        dot_general=obj.get("dot_general", None),
        promote_dtype=obj.get("promote_dtype", False),
        # preferred_element_type=obj.get("preferred_element_type", None),
        rngs=nnx.Rngs(0),
    )
    graph, data = nnx.split(layer)
    state_dict = asset_manager.read_msgpack("state.msgpack")
    state = serialization.from_state_dict(data.to_pure_dict(), state_dict)
    return nnx.merge(graph, state)


register_serialization(
    nnx.Linear, serialize_linear, deserialization_fun=deserialize_linear
)
