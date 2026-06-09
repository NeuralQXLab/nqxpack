from typing import Any
from collections.abc import Callable
from functools import partial, wraps

# needed to support parametric classes

import numpy as np
from typing import TypeVar
import jax.numpy as jnp
from plum import type_unparametrized


from nqxpack._src.lib_v1.asset_lib import (
    AssetManager,
)
from nqxpack._src.lib_v1.resolution import (
    _qualname,
)
from nqxpack._src.lib_v1.versioned_registry import (
    VERSIONED_DESERIALIZATION_REGISTRY,
)
from nqxpack._src.lib_v1.options import OptionSpec, register_option
from nqxpack._src.contextmgr import current_context


def _with_option_scope(fun, specs):
    """Wrap ``fun`` so that ``specs`` are the option declarations in scope for its body.

    When ``specs`` is empty the function is returned unchanged (no wrapper, no
    overhead) -- only (de)serializers that declared options pay for a scope push.
    """
    if not specs:
        return fun

    @wraps(fun)
    def scoped(obj):
        with current_context()._option_scope(specs):
            return fun(obj)

    return scoped


TYPE_SERIALIZATION_REGISTRY = {}


T = TypeVar("T")
PathT = tuple[str, ...]
SerializationFun = Callable[[T, PathT, AssetManager], dict]
ConversionFun = Callable[[T], Any]
DeserializationFun = Callable[[dict], T]


def has_custom_serializer(obj_or_cls) -> bool:
    """Whether serializing this object dispatches to a custom serializer.

    Returns ``True`` if serializing ``obj_or_cls`` would be handled by a
    user-provided serializer -- either a ``__to_json__`` method or a function
    registered via :func:`register_serialization` -- rather than falling through
    to the default handling (dataclass / closure / qualified-name fallback).

    Accepts either an instance or a class. This mirrors the dispatch decision
    made in ``serialize_custom_object`` so callers do not need to reach into the
    internal registry. Use it, for example, to decide whether a framework-wrapped
    model should be unwrapped so that a serializer registered for its concrete
    class can fire.
    """
    if hasattr(obj_or_cls, "__to_json__"):
        return True
    if isinstance(obj_or_cls, type):
        cls = obj_or_cls
    else:
        cls = type_unparametrized(obj_or_cls)
    return cls in TYPE_SERIALIZATION_REGISTRY


def register_serialization(
    cls: type[T],
    serialization_fun: SerializationFun | ConversionFun,
    deserialization_fun: DeserializationFun | None = None,
    reconstruct_type: bool = True,
    override: bool = False,
    min_version: tuple[int, int, int] = (0, 0, 0),
    *,
    options: list[OptionSpec] | None = None,
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
        min_version: Minimum package version (inclusive) that was used to save/serialize data in this format.
            Default is (0, 0, 0). This indicates "data saved by package version >= min_version uses this format".
            When loading, the deserialization function will be selected based on the version that was used to save,
            not the current package version. The deserialization function will be registered in the versioned
            registry with this min_version.
        options: Per-call options accepted by this (de)serializer, declared with
            :func:`~nqxpack.registry.SaveOption` / :func:`~nqxpack.registry.LoadOption`.
            Save options are readable (via :func:`~nqxpack.registry.option`) inside
            ``serialization_fun``, load options inside ``deserialization_fun``.
    """
    save_specs = {s.name: s for s in (options or []) if s.direction == "save"}
    load_specs = {s.name: s for s in (options or []) if s.direction == "load"}
    for s in options or []:
        register_option(s)

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

    # Resolve `option()` reads inside each body against this registration's
    # declarations (no-op when nothing was declared).
    _serialize_fun = _with_option_scope(_serialize_fun, save_specs)

    if cls in TYPE_SERIALIZATION_REGISTRY and not override:
        raise ValueError(f"Type {cls} is already registered for serialization")

    TYPE_SERIALIZATION_REGISTRY[cls] = _serialize_fun
    if deserialization_fun is not None:
        register_deserialization(
            class_path=cls,
            deserialization_fun=_with_option_scope(deserialization_fun, load_specs),
            min_version=min_version,
        )


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


def register_deserialization(
    class_path: str | type,
    deserialization_fun: DeserializationFun,
    min_version: tuple[int, int, int] = (0, 0, 0),
):
    """
    Register a deserialization function for a deprecated or older version of a class.

    This function allows registering deserialization logic without a corresponding
    serialization function, useful for maintaining backwards compatibility with
    older serialized formats.

    Args:
        class_path: Either a fully qualified class path as a string (e.g., "package.module.OldClass")
                   or a class object which will be converted to a class path.
        deserialization_fun: Function that takes a dict and returns an instance
        min_version: Minimum package version (inclusive) that was used to save/serialize data in this format.
                    Default is (0, 0, 0). This indicates "data saved by package version >= min_version uses this deserializer".
                    When loading, this deserializer will be selected based on the version that was used to save the data.
    """
    if not isinstance(class_path, str):
        class_path = _qualname(class_path, skip_register=True)

    VERSIONED_DESERIALIZATION_REGISTRY.register(
        class_path=class_path,
        deserialization_fun=deserialization_fun,
        min_version=min_version,
    )


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
    data = obj["data"]
    if isinstance(data, (list, float, int, complex)):
        return np.array(data, dtype=obj["dtype"]).reshape(obj["shape"])
    else:
        return np.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"])


register_serialization(
    np.ndarray, partial(serialize_np, np.ndarray), partial(deserialize_np, np.ndarray)
)


def StringKeyDict(*mapping):
    """
    Used to rebuild a dictionary with string keys from a list of tuples.
    """
    return dict(mapping)
