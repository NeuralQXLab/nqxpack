from types import FunctionType
from functools import partial

# needed to support parametric classes
from plum.parametric import type_unparametrized

import numpy as np

import dataclasses

from nqxpack._src.lib_v1.resolution import (
    _fname,
    _resolve_qualname,
    _qualname,
)
from nqxpack._src.lib_v1.custom_types import (
    TYPE_SERIALIZATION_REGISTRY,
    TYPE_DESERIALIZATION_REGISTRY,
)
from nqxpack._src.lib_v1.closure import (
    is_closure,
    CLOSURE_SERIALIZATION_REGISTRY,
)
from nqxpack._src.errors import (
    SerializationError,
)
from nqxpack._src.contextmgr import autopath, current_context

PLAIN_TYPES = (int, float, str, bool, type(None))

NUMERIC_TYPES = (int, float, complex)


@autopath
def serialize_object(obj):
    """
    Serialize an object to a JSON-serializable form similar to what hydra can process.

    To deserialize the object, use `deserialize_object`.

    This also accepts an ``asset_manager`` argument that can be used to store
    large objects that should not be serialized in the JSON file. If the asset manager is
    not specified, this function might fail.

    Args:
        obj: The object to serialize.
        path: The path to the object, as a tuple of strings. This is optional, and is used
            internally to provide better error messages.
        asset_manager: An instance of an AssetManager, which is used to store large objects
    """
    if obj is None:
        return obj
    elif isinstance(obj, PLAIN_TYPES):
        return obj
    elif isinstance(obj, list):
        return [serialize_object(x, path=i) for i, x in enumerate(obj)]
    elif isinstance(obj, tuple):
        if len(obj) == 0:
            return {"_target_": "builtins.tuple"}
        else:
            return {
                "_target_": "builtins.tuple",
                "_args_": [
                    [
                        serialize_object(
                            x,
                            path=("_args_", i),
                        )
                        for i, x in enumerate(obj)
                    ],
                ],
            }
    elif isinstance(obj, dict):
        # It's rare (though it happnens, for example in nnx.NodeMapping) but some dicts
        # have non-string keys. We need to convert them to strings. We do this by wrapping
        if any(not isinstance(k, str) for k in obj.keys()):
            return {
                "_target_": "nqxpack._src.lib_v1.custom_types.StringKeyDict",
                "_args_": [
                    [
                        serialize_object(k, path="_args_/{i}"),
                        serialize_object(v, path=k),
                    ]
                    for i, (k, v) in enumerate(obj.items())
                ],
            }
        else:
            return {k: serialize_object(v, path=k) for k, v in obj.items()}

    # I don't think this is needed, as it's handled by serializing the type itself.
    # elif any(obj is t for t in NUMERIC_TYPES):
    #    return np.dtype(obj).name
    elif isinstance(obj, np.dtype):
        return obj.name
    elif isinstance(obj, type):
        # special case NoneType, because it does not exist in builtins but it does in types
        if obj is type(None):
            return "< types.NoneType >"
        return "< " + _fname(obj) + " >"
    else:
        return serialize_object(serialize_custom_object(obj))


@autopath
def deserialize_object(obj):
    """
    Deserialize an object from a JSON-serializable produced by `serialize_object`.

    Args:
        obj: The object to deserialize.
        asset_manager: An instance of an AssetManager.
    """
    if obj is None:
        return obj
    elif isinstance(obj, str):
        # Check if it is a function
        if obj.startswith("<") and obj.endswith(">"):
            return _resolve_qualname(obj[1:-1].strip())
        return obj
    elif isinstance(obj, PLAIN_TYPES):
        return obj
    elif isinstance(obj, list):
        return [deserialize_object(x, path=i) for i, x in enumerate(obj)]
    elif isinstance(obj, tuple):
        return tuple(deserialize_object(x, path=i) for i, x in enumerate(obj))
    elif isinstance(obj, dict):
        obj = {k: deserialize_object(v, path=k) for k, v in obj.items()}
        if "_target_" in obj:
            obj = deserialize_custom_object(obj)
        return obj
    else:
        global_path = current_context().path
        raise NotImplementedError(
            f"Cannot deserialize object of type {type(obj)} : {obj} found at path `{global_path}`."
        )


def serialize_custom_object(obj):
    """
    Function to serialize custom objects. This is used by `serialize_object` to serialize
    custom objects that are not handled by the default serialization.

    Should not be used directly.
    """

    if hasattr(obj, "__to_json__"):
        dict_data = obj.__to_json__()
        if "_target_" not in dict_data:
            dict_data["_target_"] = _qualname(obj)
        return dict_data
    elif type_unparametrized(obj) in TYPE_SERIALIZATION_REGISTRY:
        typ = type_unparametrized(obj)
        dict_data = TYPE_SERIALIZATION_REGISTRY[typ](obj)
        return dict_data
    elif dataclasses.is_dataclass(obj):
        # no: this recurses
        # dict_data = dataclasses.asdict(obj)
        dict_data = {"_target_": _qualname(obj)}
        for field in dataclasses.fields(obj):
            dict_data[field.name] = serialize_object(
                getattr(obj, field.name), path=field.name
            )
        return dict_data
    elif is_closure(obj):
        serialization_fun = CLOSURE_SERIALIZATION_REGISTRY.get(_fname(obj), None)
        if serialization_fun is not None:
            return serialization_fun(obj)
        else:
            global_path = current_context().path
            raise NotImplementedError(
                f"""
                    Cannot serialize closure object of type `{_qualname(obj)}` : `{obj}`
                    found at path `{global_path}`.

                You should register a serialization function for this closure by adding to
                netket_project/utils/serialization/serialize_v1.py the following line:

                register_closure_simple_serialization(
                    parent_function, {obj.__name__}, original_qualname='{_qualname(obj)}'
                    )
                """
            )

    elif isinstance(obj, FunctionType):
        if obj.__name__ == obj.__qualname__:  # top-level function
            return "< " + _fname(obj) + " >"
        else:
            # staticmethod
            return "< " + _fname(obj) + " >"
    else:
        global_path = current_context().path
        raise SerializationError(_qualname(obj), obj, path=global_path)


def deserialize_custom_object(obj):
    if "_target_" in obj:
        target_str = obj["_target_"]
        del obj["_target_"]

        is_custom = target_str.startswith("#")
        if is_custom:
            target_str = target_str[1:]
        target = _resolve_qualname(target_str)

        if is_custom:
            return TYPE_DESERIALIZATION_REGISTRY[target](obj)
        else:
            deserialization_fun = TYPE_DESERIALIZATION_REGISTRY.get(
                target, partial(default_deserialization, target)
            )
            return deserialization_fun(obj)
    else:
        global_path = current_context().path
        raise ValueError(
            f"Invalid serialization format for custom object at `{global_path}`."
        )


def default_deserialization(target, obj):
    try:
        if "_args_" in obj:
            args = obj["_args_"]
            del obj["_args_"]
        else:
            args = ()
        return target(*args, **obj)
    except Exception as err:
        global_path = current_context().path
        raise ValueError(
            f"Impossible to reconstruct object at {global_path}:\n obj: {target}({args}, {obj})"
        ) from err
