from functools import lru_cache

import jax


@lru_cache
def process_index() -> int:
    """
    Returns the index of this process running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_index()``.

    This is an integer between 0 and
    :func:`netket_pro.distributed.process_count()`.
    """
    return jax.process_index()


def is_master_process() -> bool:
    """
    Returns whether the current process is the master process.
    """
    return process_index() == 0
