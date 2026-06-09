"""Round-trip tests for the `save_sample_cache` MCState save option.

Runs on multi-process (sharding) setups too: the sample cache must come back
sharded along the chain axis exactly as it was before saving.
"""

import numpy as np
from pathlib import Path

import pytest

import nqxpack
from nqxpack._src import distributed
import netket as nk

hi = nk.hilbert.Spin(0.5, 4)


def _make_state():
    sa = nk.sampler.MetropolisLocal(hi, n_chains=4)
    return nk.vqs.MCState(sa, nk.models.RBM(param_dtype=complex), n_samples=64)


def _save_load(vs, tmpdir, **kwargs):
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))
    path = tmpdir / "mcstate.nk"

    distributed.barrier("save start")
    nqxpack.save(vs, path, **kwargs)
    distributed.barrier("save end")
    return nqxpack.load(path)


@pytest.mark.parametrize("save_cache", [False, True])
def test_save_sample_cache(save_cache, tmpdir):
    vs = _make_state()
    _ = vs.samples  # populate the sample cache

    options = {"save_sample_cache": True} if save_cache else None
    new_vs = _save_load(vs, tmpdir, options=options)

    if not save_cache:
        # Default: the cache is not persisted, just regenerated lazily on access.
        assert new_vs._samples is None
    else:
        # Persisted: restored, value-equal, and sharded exactly as before.
        assert new_vs._samples is not None
        orig, _ = distributed.allgather(vs._samples)
        restored, _ = distributed.allgather(new_vs._samples)
        np.testing.assert_allclose(orig, restored)
        assert new_vs._samples.sharding == vs._samples.sharding


def test_save_sample_cache_without_samples(tmpdir):
    # Option on, but the cache was never populated -> nothing to persist.
    vs = _make_state()
    assert vs._samples is None

    new_vs = _save_load(vs, tmpdir, options={"save_sample_cache": True})
    assert new_vs._samples is None
