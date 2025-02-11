import numpy as np
from pathlib import Path

import jax

import pytest

import nqxpack
import netket as nk
from netket_pro import distributed

hi = nk.hilbert.Spin(0.5, 4)
g = nk.graph.Chain(4)

operators = {}
operators["Ising"] = nk.operator.Ising(hi, h=1.0, graph=g)
operators["LocalOperator"] = operators["Ising"].to_local_operator()
operators["LocalOperatorJax"] = operators["LocalOperator"].to_jax_operator()
operators["PauliStrings"] = operators["LocalOperator"].to_pauli_strings()
operators["PauliStringsJax"] = operators["PauliStrings"].to_jax_operator()

operators_params = [pytest.param(op, id=name) for name, op in operators.items()]


@pytest.mark.parametrize("operator", operators_params)
def test_save_mcstate(operator, tmpdir):
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))
    elif distributed.mode() == "mpi":
        tmpdir = nk.utils.mpi.MPI_py_comm.bcast(str(tmpdir), root=0)
        tmpdir = Path(tmpdir)

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
