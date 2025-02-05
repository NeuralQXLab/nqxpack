import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen

import pytest

import nqxpack
import netket as nk

hi = nk.hilbert.Spin(0.5, 4)
g = nk.graph.Chain(4)

models = {}
models["RBM"] = nk.models.RBM(param_dtype=complex)
models["SymmSymExp"] = nk.nn.blocks.SymmExpSum(
    nk.models.RBM(param_dtype=complex), symm_group=g.translation_group()
)


def build_nnx_model():
    return nnx.Sequential(
        nnx.Linear(
            in_features=4,
            out_features=2,
            rngs=nnx.Rngs(1),
        ),
        # nnx.relu,
        nnx.Linear(
            in_features=2,
            out_features=1,
            rngs=nnx.Rngs(1),
        ),
        jnp.squeeze,
    )


models["NNX-Sequential"] = build_nnx_model

models_params = [pytest.param(model, id=name) for name, model in models.items()]


@pytest.mark.parametrize("model", models_params)
def test_save_mcstate(model, tmpdir):
    if not isinstance(model, linen.Module):
        model = model()
    sa = nk.sampler.MetropolisLocal(hi, n_chains=4)
    vs = nk.vqs.MCState(sa, model, n_samples=64)

    nqxpack.save(vs, tmpdir / "mcstate.mpack")
    nqxpack.save(vs, "mcstate.mpack")
    new_vs = nqxpack.load(tmpdir / "mcstate.mpack")

    assert vs.hilbert == new_vs.hilbert
    jax.tree.map(np.testing.assert_allclose, vs.parameters, new_vs.parameters)
    jax.tree.map(np.testing.assert_allclose, vs.samples, new_vs.samples)
