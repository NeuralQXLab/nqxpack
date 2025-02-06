from functools import partial

# flake8: noqa: E402
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
    register_automatic_serialization,
)
from nqxpack._src.contextmgr import current_context

import jax
from flax import serialization

# Graph
from netket.graph import Lattice


def serialize_Lattice(g):
    return {
        "basis_vectors": g.basis_vectors,
        "extent": g.extent.tolist(),
        "pbc": g.pbc.tolist(),
        "site_offsets": g._site_offsets,
        "point_group": g._point_group,
        "max_neighbor_order": g._max_neighbor_order,
    }


register_serialization(Lattice, serialize_Lattice)

# Hilbert
from netket.hilbert import Spin, Qubit, Fock
from netket.hilbert import SpinOrbitalFermions
from netket.hilbert.constraint import ExtraConstraint


def serialize_Spin(hi):
    return {
        "s": hi._s,
        "N": hi.size,
        "total_sz": hi._total_sz,
        "inverted_ordering": hi._inverted_ordering,
        "constraint": hi.constraint if hi._total_sz is None else None,
    }


register_serialization(Spin, serialize_Spin)

register_automatic_serialization(Qubit, N="size")
register_automatic_serialization(
    Fock,
    "n_max",
    "n_particles",
    N="size",  # constraint="constraint"
)


def serialize_SpinOrbitalFermions(hi):
    data = {
        "n_orbitals": hi.n_orbitals,
        "s": hi.spin,
    }
    constraint = hi.constraint
    if hi.spin is None:
        data["n_fermions"] = hi.n_fermions
    else:
        data["n_fermions_per_spin"] = hi.n_fermions_per_spin

    # Set the constraint as None for the default constraint
    if any(s is not None for s in hi.n_fermions_per_spin):
        if isinstance(constraint, ExtraConstraint):
            constraint = constraint.extra_constraint
        else:
            constraint = None
    data["constraint"] = constraint
    return data


register_serialization(SpinOrbitalFermions, serialize_SpinOrbitalFermions)

from netket.hilbert import DoubledHilbert


def serialize_DoubledHilbert(hi):
    return {
        "hilb": hi.physical,
    }


def deserialize_DoubledHilbert(obj):
    return DoubledHilbert(obj["hilb"])


register_serialization(
    DoubledHilbert,
    serialize_DoubledHilbert,
    deserialization_fun=deserialize_DoubledHilbert,
)

# Constraints
from netket.hilbert.constraint import SumConstraint, SumOnPartitionConstraint

register_automatic_serialization(SumConstraint, "sum_value")
register_automatic_serialization(SumOnPartitionConstraint, "sum_values", "sizes")
register_automatic_serialization(ExtraConstraint, "base_constraint", "extra_constraint")

# Sampler
from netket.sampler import MetropolisSampler, ExactSampler

register_automatic_serialization(
    MetropolisSampler,
    "hilbert",
    "rule",
    "sweep_size",
    "reset_chains",
    "n_chains",
    "chunk_size",
    "machine_pow",
    "dtype",
    array_to_list=True,
)
register_automatic_serialization(
    ExactSampler, "hilbert", "machine_pow", "dtype", array_to_list=True
)

# Sampler Rules
from netket.sampler.rules import (
    ExchangeRule,
    FixedRule,
    HamiltonianRule,
    LocalRule,
    MultipleRules,
    TensorRule,
)

register_automatic_serialization(FixedRule)
register_automatic_serialization(LocalRule)
register_automatic_serialization(ExchangeRule, "clusters", array_to_list=True)
register_automatic_serialization(
    MultipleRules, "rules", "probabilities", array_to_list=True
)
register_automatic_serialization(TensorRule, "hilbert", "rules")
register_automatic_serialization(HamiltonianRule, "operator")


# group theory
from netket.graph.space_group import Translation, Permutation


def serialize_translation(t):
    return {
        "permutation": t.permutation.wrapped.tolist(),
        "displacement": t._vector.tolist(),
    }


register_serialization(Translation, serialize_translation)


def serialize_permutation(t):
    return {
        "permutation": t.permutation.wrapped.tolist(),
        "name": t._name,
    }


register_serialization(Permutation, serialize_permutation)


try:
    from netket.utils.model_frameworks.nnx import NNXWrapper
except ModuleNotFoundError:
    raise ImportError(
        "This version of netket pro requires a more recent netket version. Update NETKET! (from github)"
    )

register_automatic_serialization(NNXWrapper, "graphdef")

# mcstate
from netket.vqs import MCMixedState, MCState, FullSumState


def _replicate(x):
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return jax.lax.with_sharding_constraint(
            x, jax.sharding.PositionalSharding(jax.devices()).replicate()
        )
    return x


# For model states using frameworks that
def _unpack_variables(state_dict, obj):
    if "variables_structure" in obj:
        variables_flat, _ = jax.tree_flatten(state_dict["variables"])
        variables = jax.tree.unflatten(obj["variables_structure"], variables_flat)
        del obj["variables_structure"], variables_flat
    else:
        variables = state_dict["variables"]
    return variables


def serialize_mcstate(
    state: MCState,
) -> dict:
    asset_manager = current_context().asset_manager
    path = current_context().raw_path

    state_dict = serialization.to_state_dict(state)
    state_dict = jax.tree.map(_replicate, state_dict)
    variables_structure = jax.tree.structure(state.variables)
    asset_manager.write_msgpack("state.msgpack", state_dict, path=path)

    return {
        "sampler": state.sampler,
        "model": state._model,  # write the bare model
        "variables_structure": variables_structure,
    }


def deserialize_vstate(
    cls,
    obj,
) -> MCState:
    asset_manager = current_context().asset_manager
    path = current_context().raw_path

    state_dict = asset_manager.read_msgpack("state.msgpack", path=path)
    variables = _unpack_variables(state_dict, obj)
    state = cls(**obj, variables=variables)
    state = serialization.from_state_dict(state, state_dict)

    return state


register_serialization(MCState, serialize_mcstate, partial(deserialize_vstate, MCState))


def serialize_mcmixedstate(state: MCMixedState) -> dict:

    asset_manager = current_context().asset_manager
    path = current_context().raw_path

    state_dict = serialization.to_state_dict(state)
    state_dict = jax.tree.map(_replicate, state_dict)
    asset_manager.write_msgpack("state.msgpack", state_dict, path=path)

    return {
        "sampler": state.sampler,
        "sampler_diag": state.diagonal.sampler,
        "model": state._model,  # write the bare model
    }


def deserialize_mcmixedstate(obj) -> MCMixedState:
    asset_manager = current_context().asset_manager
    path = current_context().raw_path

    state_dict = asset_manager.read_msgpack("state.msgpack", path=path)
    variables = _unpack_variables(state_dict, obj)
    state = MCMixedState(**obj, variables=variables)
    state = serialization.from_state_dict(state, state_dict)
    return state


register_serialization(MCMixedState, serialize_mcmixedstate, deserialize_mcmixedstate)


def serialize_fullsumstate(state: FullSumState, *, mixed_state: bool = False) -> dict:
    asset_manager = current_context().asset_manager
    path = current_context().raw_path

    state_dict = serialization.to_state_dict(state)
    state_dict = jax.tree.map(_replicate, state_dict)
    asset_manager.write_msgpack("state.msgpack", state_dict, path=path)

    if not mixed_state:
        hilbert = state.hilbert
    else:
        hilbert = state.hilbert.physical

    return {
        "hilbert": hilbert,
        "model": state._model,  # write the bare model
    }


register_serialization(
    FullSumState, serialize_fullsumstate, partial(deserialize_vstate, FullSumState)
)
