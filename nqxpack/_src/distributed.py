import numpy as np

import jax
from jax.experimental import multihost_utils

from netket.utils import module_version

if module_version("jax") >= (0, 7, 0):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

from netket import config as nkconfig


def replicate_sharding():
    """
    Create a replicated sharding that works with both old and new JAX versions.
    Local copy to avoid netket_pro dependency.
    """
    # TODO: always use the NamedShardng version
    if module_version("jax") >= (0, 7, 0):
        return NamedSharding(jax.sharding.get_abstract_mesh(), P())
    else:
        from jax.sharding import PositionalSharding

        return PositionalSharding(jax.devices()).replicate()


def mode() -> str:
    """
    Returns the distributed mode used by NetKet.

    This can be one of the following: ``None``, ``"sharding"``
    """
    if nkconfig.netket_experimental_sharding:
        return "sharding"
    else:
        return None


def barrier(name: str):
    """
    Synchronizes all processes. This function ensures that all processes reach this point
    before continuing.

    Args:
        name: A unique string to identify the synchronization point.
    """
    if mode() == "sharding":
        multihost_utils.sync_global_devices(name)


def broadcast_string(s: str, root: int = 0) -> str:
    def _encode_string_to_uint64_array(s):
        """Encodes a string into a NumPy array of uint64."""
        byte_data = s.encode("utf-8")  # Convert to bytes
        padding_size = (
            8 - len(byte_data) % 8
        ) % 8  # Compute padding to make it multiple of 8
        byte_data += b"\x00" * padding_size  # Pad with null bytes
        uint64_array = np.frombuffer(byte_data, dtype=np.uint64)  # Interpret as uint64
        return uint64_array, padding_size

    def _decode_uint64_array_to_string(uint64_array, padding_size):
        """Decodes a NumPy uint64 array back to a string."""
        byte_data = uint64_array.tobytes()  # Convert back to bytes
        return (
            byte_data[:-padding_size].decode("utf-8")
            if padding_size
            else byte_data.decode("utf-8")
        )

    if jax.process_count() > 1:
        if root != 0:
            raise ValueError("Only root=0 is supported in sharding mode")

        encoded_array, pad_size = _encode_string_to_uint64_array(s)
        encoded_array = multihost_utils.broadcast_one_to_all(encoded_array)
        pad_size = multihost_utils.broadcast_one_to_all(pad_size)
        s = _decode_uint64_array_to_string(encoded_array, pad_size)

    return s


def allgather(array, *, axis: int = 0, token=None):
    """
    Gathers (unshard) a distributed (sharded) array to all processes.

    The resulting array will have the same shape as the input array except
    the first axis, which will be :ref:`netket_pro.distributed.process_count`
    times longer.

    .. note::

        An input array of shape :math:`(M, N, ...)` will lead to a gathered
        array of shape :math:`(P \times M, N, ...)`, where :math:`P` is the
        number of processes.

    .. note::

        The resulting array will be unsharded, or fully addressable locally
        and on every process.

    Args:
        array: The array to gather.

    Returns:
        A tuple of the gathered array and the token.

    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported for now. Open a PR.")

    if mode() == "sharding":
        sharding = replicate_sharding()
        array = jax.lax.with_sharding_constraint(array, sharding)
    else:
        pass
    return array, token
