import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn

import pytest

import nqxpack
from nqxpack._src.errors import JaxArraySerializationError


def _build_model():
    return nn.Sequential([nn.Dense(features=4)])


def test_save_raises_targeted_error_for_jax_arrays(tmp_path):
    model = _build_model()
    variables = model.init(jax.random.key(0), jnp.ones((1, 4)))

    with pytest.raises(JaxArraySerializationError) as excinfo:
        nqxpack.save(
            {"model": model, "variables": variables}, tmp_path / "checkpoint.nk"
        )

    msg = str(excinfo.value)

    assert "variables/params/layers_0/kernel" in msg
    assert "nqxpack only supports NumPy arrays" in msg
    assert "jax.tree.map(np.asarray, pytree)" in msg
    assert "shape: (4, 4)" in msg
    assert "dtype: float32" in msg


def test_save_accepts_manual_numpy_conversion_for_flax_variables(tmp_path):
    model = _build_model()
    variables = model.init(jax.random.key(0), jnp.ones((1, 4)))
    variables_np = jax.tree.map(np.asarray, variables)

    path = tmp_path / "checkpoint.nk"
    nqxpack.save({"model": model, "variables": variables_np}, path)

    assert path.exists()
