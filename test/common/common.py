# File containing common commands for NetKet Test infrastructure

from typing import Any
import jax

from functools import partial
import os

import pytest

import netket as nk


def _is_true(x):
    if isinstance(x, str):
        xl = x.lower()
        if xl == "1" or x == "true":
            return True
    elif x == 1:
        return True
    else:
        return False


skipif_ci = pytest.mark.skipif(
    _is_true(os.environ.get("CI", False)), reason="Test too slow/broken on CI"
)
"""Use as a decorator to mark a test to be skipped when running on CI.
For example:

Example:
>>> @skipif_ci
>>> def test_my_serial_function():
>>>     your_serial_test()

"""

xfailif_distributed = pytest.mark.xfail(
    nk.config.netket_experimental_sharding and jax.process_count() > 1,
    reason="Broken under distributed.",
)

skipif_sharding = pytest.mark.skipif(
    nk.config.netket_experimental_sharding, reason="Only run without sharding"
)

skipif_distributed = pytest.mark.skipif(
    nk.config.netket_experimental_sharding,
    reason="Skip if distributed",
)


class set_config:
    """
    Temporarily changes the value of the configuration `name`.

    Example:

    >>> with set_config("netket_experimental_disable_ode_jit", True):
    >>>     run_code

    """

    def __init__(self, name: str, value: Any):
        self._name = name.upper()
        self._value = value

    def __enter__(self):
        self._orig_value = nk.config.FLAGS[self._name]
        nk.config.update(self._name, self._value)

    def __exit__(self, exc_type, exc_value, traceback):
        nk.config.update(self._name, self._orig_value)


netket_experimental_fft_autocorrelation = partial(
    set_config, "NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"
)


def hash_for_seed(obj):
    """
    Hash any object into an int that can be used in `np.random.seed`, and does not change between Python sessions.

    Args:
      obj: any object with `repr` defined to show its states.
    """

    bs = repr(obj).encode()
    out = 0
    for b in bs:
        out = (out * 256 + b) % 4294967291  # Output in [0, 2**32 - 1]
    return out


def named_parametrize(argname: str, values: list):
    param_values = [pytest.param(obj, id=f"{argname}={obj}") for obj in values]
    return pytest.mark.parametrize(argname, param_values)
