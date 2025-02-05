from nqxpack._src.lib_v1.custom_types import (
    register_automatic_serialization,
)

# Graph

# flax.nnx
from flax.nnx.graph import HashableMapping

register_automatic_serialization(HashableMapping, mapping="_mapping")
