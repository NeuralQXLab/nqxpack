"""Regression test for nqxpack MCState serialization with >= 10 list-valued
variable entries.

``flax.serialization.to_state_dict`` converts Python lists to dicts with
string keys (``{'0': ..., '1': ..., '10': ..., '2': ...}``).  When the list
has >= 10 elements, alphabetical key order diverges from numeric order, so
``jax.tree.flatten`` on the loaded dict produces leaves in the wrong order,
corrupting the deserialized parameters.  The fix (``_sort_numeric_string_keys``)
must convert those string-key dicts back to lists before flattening.
"""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx


import nqxpack
from nqxpack._src import distributed
from nqxpack._src.registry.netket import _sort_numeric_string_keys
import netket as nk

# ---------------------------------------------------------------------------
# Unit tests for the helper
# ---------------------------------------------------------------------------


def test_sort_numeric_string_keys_shallow():
    """Dict with numeric string keys should be converted to a sorted list."""
    d = {str(i): i * 10 for i in range(11)}  # keys: '0'..'10' in insert order
    result = _sort_numeric_string_keys(d)
    assert result == [i * 10 for i in range(11)]


def test_sort_numeric_string_keys_nested():
    """Numeric-string dicts nested inside a regular-key dict are converted."""
    d = {"params": {str(i): float(i) for i in range(11)}}
    result = _sort_numeric_string_keys(d)
    assert isinstance(result, dict)
    assert result["params"] == [float(i) for i in range(11)]


def test_sort_numeric_string_keys_non_numeric_unchanged():
    """Dicts whose keys are not all numeric strings must be left as dicts."""
    d = {"kernel": 1.0, "bias": 2.0}
    result = _sort_numeric_string_keys(d)
    assert result == d


def test_sort_numeric_string_keys_fewer_than_10_unchanged_value():
    """Lists with fewer than 10 entries should also round-trip correctly."""
    d = {str(i): i for i in range(5)}
    result = _sort_numeric_string_keys(d)
    assert result == list(range(5))


# ---------------------------------------------------------------------------
# Integration test: MCState with nnx.Sequential of 11 layers
# ---------------------------------------------------------------------------


def _build_deep_nnx_model(n_layers: int = 11):
    """nnx.Sequential stores its submodules as a Python list, so with >= 10
    layers the serialised variables dict has numeric string keys whose
    alphabetical order diverges from numeric order."""
    layers = [
        nnx.Linear(in_features=4, out_features=4, rngs=nnx.Rngs(i))
        for i in range(n_layers)
    ]
    layers.append(jnp.squeeze)
    return nnx.Sequential(*layers)


hi = nk.hilbert.Spin(0.5, 4)


def test_mcstate_nnx_deep_save_load_params(tmpdir):
    """Parameters must be recovered exactly after a nqxpack save/load cycle
    when the model has >= 10 layers (triggering the numeric-string-key bug)."""
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))

    model = _build_deep_nnx_model(n_layers=11)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=4)
    vs = nk.vqs.MCState(sa, model, n_samples=64)

    distributed.barrier("deep_barrier_1")
    nqxpack.save(vs, tmpdir / "mcstate_deep.mpack")
    distributed.barrier("deep_barrier_2")

    new_vs = nqxpack.load(tmpdir / "mcstate_deep.mpack")

    # Parameters must match exactly — before the fix, layers >= index 10 were
    # mapped to the wrong weight matrices.
    jax.tree.map(np.testing.assert_allclose, vs.parameters, new_vs.parameters)
