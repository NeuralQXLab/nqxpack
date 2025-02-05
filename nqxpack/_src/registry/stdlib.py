from functools import partial

from nqxpack._src.lib_v1.custom_types import register_serialization


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
