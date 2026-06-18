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
