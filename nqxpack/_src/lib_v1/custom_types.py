from typing import Any
from collections.abc import Callable
from functools import partial, wraps

# needed to support parametric classes

import numpy as np
from typing import TypeVar
import jax.numpy as jnp


from nqxpack._src.lib_v1.asset_lib import (
    AssetManager,
)
from nqxpack._src.lib_v1.resolution import (
    _qualname,
)


TYPE_SERIALIZATION_REGISTRY = {}
TYPE_DESERIALIZATION_REGISTRY = {}


T = TypeVar("T")
PathT = tuple[str, ...]
SerializationFun = Callable[[T, PathT, AssetManager], dict]
ConversionFun = Callable[[T], Any]
DeserializationFun = Callable[[dict, PathT, AssetManager], T]


def register_serialization(
    cls: T,
    serialization_fun: SerializationFun | ConversionFun,
    deserialization_fun: DeserializationFun = None,
    reconstruct_type: bool = True,
    override: bool = False,
):
    """
    Register a custom serialization function and deserialization function for a given class.

    Args:
        cls: The class to register the serialization for
        serialization_fun: A function that takes an instance of the class and returns a dictionary
            with the serialized data. If the dictionary has a key "_target_", it will be used to
            deserialize the object. If it does not have it, it will be added.
            If the function does not return a dictionary, no `_target_` entry is added.
        deserialization_fun: A function that takes the serialized dictionary and returns an instance
            of the class. If unspecified, the deserialization will use the "_target_" key to resolve
            the function to call.
        reconstruct_type: If True (default) the output of ``serialization_fun`` must be a dictionary, and
            the ``_target_`` field will be used to reconstruct the original field. If False, instead, the
            output can be any type handled by the serialisation library, and will not be reserialized. This
            can be used to serialize some types that you do not actually want to serialise, and convert them
            to default types.
    """
    if reconstruct_type:
        if deserialization_fun is not None:
            _target_qualname = "#" + _qualname(cls, skip_register=True)
        else:
            _target_qualname = _qualname(cls, skip_register=True)

        @wraps(serialization_fun)
        def _serialize_fun(obj):
            dict_data = serialization_fun(obj)
            assert isinstance(dict_data, dict)

            if "_target_" not in dict_data:
                dict_data["_target_"] = _target_qualname
            return dict_data

    else:
        _serialize_fun = serialization_fun

    if cls in TYPE_SERIALIZATION_REGISTRY and not override:
        raise ValueError(f"Type {cls} is already registered for serialization")

    TYPE_SERIALIZATION_REGISTRY[cls] = _serialize_fun
    if deserialization_fun is not None:
        TYPE_DESERIALIZATION_REGISTRY[cls] = deserialization_fun


def _simple_serialize(
    cls,
    attrs,
    attr_map,
    obj,
    array_to_list,
):
    if array_to_list:

        def _getattr(obj, attr):
            val = getattr(obj, attr)
            if (
                isinstance(val, np.ndarray)
                or isinstance(val, jnp.ndarray)
                and val.ndim <= 2
            ):
                return val.tolist()
            return val

    else:
        _getattr = getattr
    res = {k: _getattr(obj, k) for k in attrs}
    for k, v in attr_map.items():
        res[k] = _getattr(obj, v)
    res["_target_"] = _qualname(obj)
    return res


def register_automatic_serialization(
    cls, *attrs, array_to_list: bool = False, **attr_map
):
    """
    Register a simple serialization function for a class that just serializes the given attributes.

    Args:
        cls: The class to register the serialization for
        *attrs: The attributes to serialize and pass to the constructor directily
        **attr_map: A mapping of the attribute name to the name of the attribute in the object
        array_to_list: If True, arrays with ndim <= 1 will be converted to lists
    """
    register_serialization(
        cls,
        serialization_fun=partial(
            _simple_serialize, cls, attrs, attr_map, array_to_list=array_to_list
        ),
    )


#


def serialize_np(cls, obj):
    if obj.size < 30:
        data = obj.tolist()
    else:
        data = obj.tobytes()

    return {
        "shape": obj.shape,
        "dtype": obj.dtype.name,
        "data": data,
    }


def deserialize_np(cls, obj):
    if isinstance(obj["data"], list):
        return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    else:
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])


register_serialization(
    np.ndarray, partial(serialize_np, np.ndarray), partial(deserialize_np, np.ndarray)
)


def StringKeyDict(*mapping):
    """
    Used to rebuild a dictionary with string keys from a list of tuples.
    """
    return dict(mapping)
