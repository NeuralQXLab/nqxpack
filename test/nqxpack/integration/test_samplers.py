"""Round-trip tests for netket sampler serialization.

Uses tiny hilbert spaces / replica counts so the tests run quickly.
"""

import numpy as np
from pathlib import Path

import pytest

import nqxpack
from nqxpack._src import distributed
import netket as nk


def _save_load(obj, tmpdir):
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))
    path = tmpdir / "sampler.nk"

    distributed.barrier("save start")
    nqxpack.save(obj, path)
    distributed.barrier("save end")
    return nqxpack.load(path)


hi = nk.hilbert.Spin(0.5, 2)


def test_save_ardirect_sampler(tmpdir):
    sampler = nk.sampler.ARDirectSampler(hi)

    loaded = _save_load(sampler, tmpdir)

    assert type(loaded) is type(sampler)
    assert loaded.hilbert == sampler.hilbert
    assert loaded.dtype == sampler.dtype


# Each case covers a different way to specify the parallel-tempering temperatures.
_pt_samplers = {
    "linear": dict(n_replicas=4),
    "log": dict(n_replicas=4, betas="log"),
    "custom_betas": dict(betas=[1.0, 0.6, 0.3, 0.1]),
}


@pytest.mark.parametrize(
    "kwargs", [pytest.param(kw, id=name) for name, kw in _pt_samplers.items()]
)
def test_save_parallel_tempering_sampler(kwargs, tmpdir):
    sampler = nk.sampler.ParallelTemperingSampler(
        hi, rule=nk.sampler.rules.LocalRule(), n_chains=2, **kwargs
    )

    loaded = _save_load(sampler, tmpdir)

    assert type(loaded) is type(sampler)
    assert loaded.hilbert == sampler.hilbert
    assert type(loaded.rule) is type(sampler.rule)
    assert loaded.n_replicas == sampler.n_replicas
    assert loaded.n_chains == sampler.n_chains
    assert loaded._beta_distribution == sampler._beta_distribution
    np.testing.assert_allclose(loaded.sorted_betas, sampler.sorted_betas)
