from functools import lru_cache


import jax

from netket import config as nkconfig
from netket.utils import mpi


@lru_cache
def process_index() -> int:
    """
    Returns the index of this process running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_index()``. If you are running
    with mpi, this is ``nk.utils.mpi.rank``.

    This is an integer between 0 and
    :func:`netket_pro.distributed.process_count()`.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_index()
    else:
        return mpi.rank


def is_master_process() -> bool:
    """
    Returns whether the current process is the master process.
    """
    return process_index() == 0
