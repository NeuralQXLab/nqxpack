import numpy as np
from pathlib import Path

import jax

import pytest

import nqxpack
import netket as nk
from nqxpack._src import distributed

hi = nk.hilbert.Spin(0.5, 4)
g = nk.graph.Chain(4)

operators = {}
operators["IsingNumba"] = nk.operator.IsingNumba(hi, h=1.0, graph=g)
operators["IsingJax"] = nk.operator.IsingJax(hi, h=1.0, graph=g)
operators["LocalOperatorNumba"] = operators["IsingNumba"].to_local_operator()
operators["LocalOperatorJax"] = operators["IsingJax"].to_local_operator()
operators["PauliStringsNumba"] = operators["LocalOperatorNumba"].to_pauli_strings()
operators["PauliStringsJax"] = operators["LocalOperatorJax"].to_pauli_strings()

operators_params = [pytest.param(op, id=name) for name, op in operators.items()]


@pytest.mark.parametrize("operator", operators_params)
def test_save_mcstate(operator, tmpdir):
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))

    nqxpack.save(operator, tmpdir / "operator.mpack")
    distributed.barrier("barrier 1")
    loaded_operator = nqxpack.load(tmpdir / "operator.mpack")

    assert operator.hilbert == loaded_operator.hilbert
    if not isinstance(operator, nk.operator.DiscreteJaxOperator):
        operator = operator.to_jax_operator()
        loaded_operator = loaded_operator.to_jax_operator()

    op_flat, treestruct = jax.tree.flatten(operator)
    loaded_op_flat, loaded_treestruct = jax.tree.flatten(loaded_operator)
    for a, b in zip(op_flat, loaded_op_flat):
        np.testing.assert_allclose(a, b)
    assert treestruct == loaded_treestruct
