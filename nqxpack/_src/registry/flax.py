from nqxpack._src.lib_v1.custom_types import (
    register_automatic_serialization,
)

# Graph

# flax.nnx
import flax
from flax.nnx.graph import HashableMapping

register_automatic_serialization(HashableMapping, mapping="_mapping")

flax_version = tuple(int(x) for x in flax.__version__.split("."))[:3]
if flax_version >= (0, 10, 4):
    from flax.nnx.graph import IndexesPytreeDef

    register_automatic_serialization(IndexesPytreeDef, *IndexesPytreeDef._fields)
