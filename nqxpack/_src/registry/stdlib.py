from functools import partial
import io
import numpy as np

from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
    register_automatic_serialization,
)
from netket.utils import HashableArray
from nqxpack._src.contextmgr import current_context


def serialize_partial(par):
    res = {
        "name": par.func,
    }
    if par.args:
        res["args"] = par.args
    if par.keywords:
        res["kwargs"] = par.keywords
    return res


def deserialize_partial(obj):
    return partial(obj["name"], *obj.get("args", ()), **obj.get("kwargs", {}))


register_serialization(partial, serialize_partial, deserialize_partial)


# frozenset
def serialize_frozenset(obj):
    return {"elements": list(obj)}


def deserialize_frozenset(obj):
    return frozenset(obj["elements"])


register_serialization(frozenset, serialize_frozenset, deserialize_frozenset)


# complex
register_automatic_serialization(complex, "real", "imag")


# numpy
def serialize_np_bool_(obj):
    return {"value": bool(obj)}


def deserialize_np_bool_(obj):
    return np.bool_(obj["value"])


register_serialization(np.bool_, serialize_np_bool_, deserialize_np_bool_)


def serialize_hashable_array(obj):
    asset_manager = current_context().asset_manager

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(obj))
    asset_manager.write_asset("array.npy", buffer.getvalue())
    return {}


def deserialize_hashable_array(obj):
    asset_manager = current_context().asset_manager
    array = np.load(io.BytesIO(asset_manager.read_asset("array.npy")))

    return HashableArray(array)


register_serialization(
    HashableArray,
    serialization_fun=serialize_hashable_array,
    deserialization_fun=deserialize_hashable_array,
)
