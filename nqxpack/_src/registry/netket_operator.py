from functools import partial

import numpy as np
from scipy.sparse import issparse, coo_matrix

import io

# flake8: noqa: E402
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
)
from nqxpack._src.contextmgr import current_context


def _pack_array(array):
    if isinstance(array, np.ndarray):
        return array
    elif issparse(array):
        array = array.tocoo()
        return {
            "_target_": "scipy.sparse.coo_matrix",
            "data": array.data,
            "row": array.row,
            "col": array.col,
            "shape": list(array.shape),
        }
    else:
        raise ValueError(f"Unsupported array type: {type(array)}")


def _unpack_array(array, dtype=None):
    if isinstance(array, np.ndarray):
        if dtype is not None:
            array = array.astype(dtype)
        return array
    elif isinstance(array, dict) and array.get("_target_") == "scipy.sparse.coo_matrix":
        return coo_matrix(
            (array["data"], (array["row"], array["col"])),
            shape=array["shape"],
            dtype=dtype,
        )


## LocalOperator
from netket.operator import LocalOperatorJax
from netket.operator._local_operator.numba import LocalOperator as LocalOperatorNumba


def serialize_LocalOperator(op):

    operators = [_pack_array(term) for term in op.operators]
    current_context().asset_manager.write_msgpack("operators.msgpack", operators)

    # if hasattr(op, "_acting_on"):
    #     acting_on = op._acting_on
    #     if not isinstance(acting_on, list):
    #         acting_on = acting_on.tolist()
    # else:
    acting_on = [list(int(o) for o in ao) for ao in op.acting_on]

    return {
        "hilbert": op.hilbert,
        # "operators": op.operators,
        "acting_on": acting_on,
        "constants": op.constant.item(),
        "dtype": op.dtype,
    }


def deserialize_LocalOperator(cls, obj):

    operators = current_context().asset_manager.read_msgpack("operators.msgpack")
    operators = [_unpack_array(arr, dtype=obj["dtype"]) for arr in operators]

    return cls(
        hilbert=obj["hilbert"],
        operators=operators,
        acting_on=obj["acting_on"],
        constant=obj["constants"],
        dtype=obj["dtype"],
    )


register_serialization(
    LocalOperatorNumba,
    serialize_LocalOperator,
    partial(deserialize_LocalOperator, LocalOperatorNumba),
)
register_serialization(
    LocalOperatorJax,
    serialize_LocalOperator,
    partial(deserialize_LocalOperator, LocalOperatorJax),
)

## PauliStrings
from netket.operator import PauliStringsJax
from netket.operator._pauli_strings.numba import PauliStrings as PauliStringsNumba


def serialize_PauliStrings(op):
    ctx = current_context()
    asset_manager = ctx.asset_manager

    buffer = io.BytesIO()
    np.savez_compressed(buffer, operators=op.operators, weights=op.weights)
    asset_manager.write_asset("data.npz", buffer.getvalue())

    return {
        "hilbert": op.hilbert,
        # "operators": op.operators,
        # "weights": op._weights,
        "cutoff": op._cutoff,
        "dtype": op.dtype,
    }


def deserialize_PauliStrings(cls, obj):
    ctx = current_context()
    asset_manager = ctx.asset_manager

    data = np.load(io.BytesIO(asset_manager.read_asset("data.npz")))
    operators = data["operators"]
    weights = data["weights"]

    return cls(
        hilbert=obj["hilbert"],
        operators=operators,
        weights=weights,
        cutoff=obj["cutoff"],
        dtype=obj["dtype"],
    )


register_serialization(
    PauliStringsNumba,
    serialize_PauliStrings,
    partial(deserialize_PauliStrings, PauliStringsNumba),
)
register_serialization(
    PauliStringsJax,
    serialize_PauliStrings,
    partial(deserialize_PauliStrings, PauliStringsJax),
)

## Ising
from netket.operator import IsingJax
from netket.operator._ising.numba import Ising as IsingNumba


def serialize_Ising(op):
    return {
        "hilbert": op.hilbert,
        "graph": op.edges.tolist(),
        "h": op.h.item(),
        "J": op.J.item(),
        "dtype": op.dtype,
    }


register_serialization(IsingNumba, serialize_Ising)
register_serialization(IsingJax, serialize_Ising)
