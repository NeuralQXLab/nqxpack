import json
from pathlib import Path

import jax
import pytest

import nqxpack
from nqxpack._src import distributed
from nqxpack._src.registry.netket import _SPIN_HAS_INVERTED_ORDERING
import netket as nk

hilberts = {
    "plain": nk.hilbert.Spin(0.5, 4),
    "total_sz": nk.hilbert.Spin(0.5, 4, total_sz=0),
    "spin1": nk.hilbert.Spin(1.0, 3),
}
if _SPIN_HAS_INVERTED_ORDERING:
    hilberts["inverted"] = nk.hilbert.Spin(0.5, 4, inverted_ordering=True)

hilberts_params = [pytest.param(hi, id=name) for name, hi in hilberts.items()]


def _tmpdir(tmpdir):
    if distributed.mode() == "sharding":
        tmpdir = Path(distributed.broadcast_string(str(tmpdir)))
    return Path(tmpdir)


def _save_state(hi, path):
    sa = nk.sampler.MetropolisLocal(hi, n_chains=4)
    vs = nk.vqs.MCState(sa, nk.models.RBM(), n_samples=64)
    distributed.barrier("save start")
    nqxpack.save(vs, path, zip=False)
    distributed.barrier("save end")


def _set_spin_field(path, value):
    # Rewrite the Spin entry of the archive as written by another nqxpack version:
    # `value=None` drops the `inverted_ordering` field, otherwise it is set.
    def _patch(obj):
        if isinstance(obj, dict):
            if obj.get("_target_", "").endswith(".Spin"):
                obj.pop("inverted_ordering", None)
                if value is not None:
                    obj["inverted_ordering"] = value
            for v in obj.values():
                _patch(v)
        elif isinstance(obj, list):
            for v in obj:
                _patch(v)

    if jax.process_index() == 0:
        config = path / "object.json"
        data = json.loads(config.read_text())
        _patch(data)
        config.write_text(json.dumps(data))
    distributed.barrier("patch end")


@pytest.mark.parametrize("hi", hilberts_params)
def test_spin_roundtrip(hi, tmpdir):
    path = _tmpdir(tmpdir) / "state.nk"
    _save_state(hi, path)

    vs = nqxpack.load(path)
    assert vs.hilbert == hi
    assert vs.sampler.hilbert == hi


@pytest.mark.parametrize("field", [None, False])
def test_spin_load_default_ordering(field, tmpdir):
    # Files written without the field (new nqxpack on NetKet main) or with the
    # default ordering (old nqxpack) load on every NetKet version.
    hi = nk.hilbert.Spin(0.5, 4, total_sz=0)
    path = _tmpdir(tmpdir) / "state.nk"
    _save_state(hi, path)
    _set_spin_field(path, field)

    vs = nqxpack.load(path)
    assert vs.hilbert == hi


def test_spin_load_inverted_ordering(tmpdir):
    hi = nk.hilbert.Spin(0.5, 4)
    path = _tmpdir(tmpdir) / "state.nk"
    _save_state(hi, path)
    _set_spin_field(path, True)

    if _SPIN_HAS_INVERTED_ORDERING:
        vs = nqxpack.load(path)
        assert vs.hilbert == nk.hilbert.Spin(0.5, 4, inverted_ordering=True)
    else:
        with pytest.raises(RuntimeError) as excinfo:
            nqxpack.load(path)
        err = excinfo.value
        while err.__cause__ is not None:
            err = err.__cause__
        assert isinstance(err, ValueError)
        assert "inverted_ordering=True" in str(err)
