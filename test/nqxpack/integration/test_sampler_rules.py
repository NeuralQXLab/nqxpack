"""Round-trip tests for netket sampler rules serialization."""

import numpy as np
from pathlib import Path

import pytest

import nqxpack
from nqxpack._src import distributed
import netket as nk


def _save_load(obj, tmpdir):
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))
    path = tmpdir / "rule.nk"

    distributed.barrier("save start")
    nqxpack.save(obj, path)
    distributed.barrier("save end")
    return nqxpack.load(path)


def _assert_rule_equal(orig, loaded):
    assert type(loaded) is type(orig)
    np.testing.assert_array_equal(
        np.asarray(orig.clusters), np.asarray(loaded.clusters)
    )
    po, pl = orig.probabilities, loaded.probabilities
    if po is None:
        assert pl is None
    else:
        np.testing.assert_array_equal(np.asarray(po), np.asarray(pl))


def test_save_fermion_hop_rule(tmpdir):
    g = nk.graph.Chain(4, pbc=False)
    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
    rule = nk.sampler.rules.FermionHopRule(hi, graph=g)

    loaded = _save_load(rule, tmpdir)
    _assert_rule_equal(rule, loaded)


def test_save_fermion_hop_sampler(tmpdir):
    g = nk.graph.Chain(4, pbc=False)
    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
    rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, rule, n_chains=4)

    loaded = _save_load(sampler, tmpdir)

    assert loaded.hilbert == sampler.hilbert
    _assert_rule_equal(sampler.rule, loaded.rule)


# Simple rules without extra state, or with only plain scalar attributes.
_simple_rules = {
    "GlobalSpinFlipRule": (nk.sampler.rules.GlobalSpinFlipRule(), {}),
    "GaussianRule": (nk.sampler.rules.GaussianRule(sigma=2.5), {"sigma": 2.5}),
    "LangevinRule": (
        nk.sampler.rules.LangevinRule(dt=0.02, chunk_size=8),
        {"dt": 0.02, "chunk_size": 8},
    ),
}


@pytest.mark.parametrize(
    "rule, attrs",
    [pytest.param(r, a, id=name) for name, (r, a) in _simple_rules.items()],
)
def test_save_simple_rule(rule, attrs, tmpdir):
    loaded = _save_load(rule, tmpdir)

    assert type(loaded) is type(rule)
    for name, value in attrs.items():
        assert getattr(loaded, name) == value
