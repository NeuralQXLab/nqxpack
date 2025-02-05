from functools import partial
from types import FunctionType

from nqxpack._src.lib_v1.resolution import (
    _fname,
    _resolve_qualname,
)

CLOSURE_SERIALIZATION_REGISTRY = {}


def is_closure(obj):
    return isinstance(obj, FunctionType) and obj.__closure__ is not None


def _serialize_closure(parent_fun, vars_mapping, closure):
    closure_vars = closure.__closure__

    res = {}
    for i, varname in enumerate(closure.__code__.co_freevars):
        varname = vars_mapping.get(varname, varname)
        res[varname] = closure_vars[i].cell_contents

    return {"_target_": _fname(parent_fun), **res}


def register_closure_simple_serialization(
    parent_fun,
    closure_name,
    vars_mapping={},
    original_qualname=None,
    override: bool = False,
):
    """
    Register the closure resulting from a function to have a simple serialization.

    This is usually enough for jax.nn.initializers, for example. The requirement
    for this to work is that the closure shows has a name of the form
    <parent_fun>.<locals>.<closure_name>.

    To register, you must provide the parent function from which the closure
    was generated, the name of the closure, and optionally a mapping of the closure
    variable names to the names you want to use in the serialization.

    Args:
        parent_fun: The parent function of the closure
        closure_name: The name of the closure
        vars_mapping: A mapping of the closure variable names to the names you want
            to use in the serialization
        original_qualname: The original qualname of the parent function. Jax usually
            hides the ._src part of the qualname, and you must provide it here to
            resolve the original function.
    """
    if original_qualname is not None:
        assert _resolve_qualname(original_qualname) == parent_fun
        qualname = original_qualname
    else:
        qualname = _fname(parent_fun)

    closure_qualname = qualname + ".<locals>." + closure_name

    if closure_qualname in CLOSURE_SERIALIZATION_REGISTRY and not override:
        raise ValueError(
            f"Closure '{closure_qualname}' is already registered for serialization"
        )

    CLOSURE_SERIALIZATION_REGISTRY[closure_qualname] = partial(
        _serialize_closure, parent_fun, vars_mapping
    )
