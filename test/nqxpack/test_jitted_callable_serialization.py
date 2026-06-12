"""Serialization of jitted / custom-gradient jax callables.

Activation functions are commonly stored as plain attributes of a model (e.g.
``self.activation = jnp.tanh``), which become static fields in an ``nnx``
graphdef. Many of them are not ``types.FunctionType`` -- ``jnp.tanh`` /
``jax.nn.silu`` / ``jax.nn.sigmoid`` are ``PjitFunction``s and ``jax.nn.relu``
is a ``custom_jvp`` -- but they carry a clean ``__module__`` / ``__qualname__``,
so nqxpack serializes them by reference just like a plain function.
"""

import jax
import jax.numpy as jnp

import pytest

import nqxpack
from nqxpack._src.errors import MainScopeError


# A mix of PjitFunction (tanh/silu/sigmoid), custom_jvp (relu) and a plain
# function (gelu), all stored as a value to be serialized by reference.
JITTED_OR_CUSTOM_GRAD = [
    jnp.tanh,
    jax.nn.tanh,
    jax.nn.silu,
    jax.nn.sigmoid,
    jax.nn.relu,
    jax.nn.gelu,
]


@pytest.mark.parametrize("fn", JITTED_OR_CUSTOM_GRAD, ids=lambda f: f.__qualname__)
def test_jitted_callable_roundtrips_by_reference(fn, tmp_path):
    path = tmp_path / "fn.nk"
    nqxpack.save({"activation": fn}, path)
    loaded = nqxpack.load(path)
    # Resolved back to the exact same object, not a copy.
    assert loaded["activation"] is fn


def test_main_scope_jitted_callable_raises(tmp_path):
    # A callable that cannot be re-imported on load (its module is __main__)
    # must fail loudly at save time rather than produce an unloadable file --
    # the same safeguard the plain-function branch gets from ``_fname``.
    main_scope_fn = jax.jit(lambda x: x + 1)
    main_scope_fn.__module__ = "__main__"

    with pytest.raises(MainScopeError):
        nqxpack.save({"activation": main_scope_fn}, tmp_path / "fn.nk")
