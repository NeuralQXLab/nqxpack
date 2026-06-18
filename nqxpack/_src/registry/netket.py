from functools import partial
import io

import numpy as np

# flake8: noqa: E402
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
    register_automatic_serialization,
    has_custom_serializer,
)
from nqxpack._src.lib_v1.options import SaveOption
from nqxpack._src.contextmgr import current_context
from nqxpack._src.distributed import replicate_sharding

import jax
import jax.numpy as jnp
from flax import serialization

# Graph
from netket.graph import Lattice
from netket.utils.version_check import module_version


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
    elif any(s is not None for s in hi.n_fermions_per_spin):
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

from netket.hilbert.tensor_hilbert import TensorHilbert
from netket.hilbert.tensor_hilbert_discrete import TensorDiscreteHilbert


def serialize_TensorHilbert(hi):
    return {"_args_": hi._hilbert_spaces}


register_serialization(
    TensorHilbert,
    serialize_TensorHilbert,
)
register_serialization(
    TensorDiscreteHilbert,
    serialize_TensorHilbert,
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

# FermionHopRule extends ExchangeRule but requires a SpinOrbitalFermions hilbert
# space in its constructor and re-expands the clusters over the spin subsectors
# when `spin_symmetric=True`. The rule only stores the fully-expanded `clusters`
# (and `probabilities`), not the hilbert space, so we cannot use the automatic
# serialization. On deserialization we rebuild it with `spin_symmetric=False` (so
# the stored clusters are used as-is, without being expanded a second time). The
# constructor still requires a SpinOrbitalFermions instance for an `isinstance`
# check, but with `spin_symmetric=False` it is otherwise unused, so a minimal
# dummy hilbert is sufficient to reconstruct an identical rule.
from netket.sampler.rules.fermion_2nd import FermionHopRule


def serialize_FermionHopRule(rule):
    res = {"clusters": np.asarray(rule.clusters).tolist()}
    if rule.probabilities is not None:
        res["probabilities"] = np.asarray(rule.probabilities).tolist()
    return res


def deserialize_FermionHopRule(data):
    clusters = np.asarray(data["clusters"])
    n_orbitals = int(clusters.max()) + 1 if clusters.size else 1
    dummy_hilbert = SpinOrbitalFermions(n_orbitals)
    rule = FermionHopRule(
        dummy_hilbert, clusters=clusters.tolist(), spin_symmetric=False
    )
    if "probabilities" in data:
        rule = rule.replace(probabilities=jnp.asarray(data["probabilities"]))
    return rule


register_serialization(
    FermionHopRule, serialize_FermionHopRule, deserialize_FermionHopRule
)


# group theory
from netket.graph.space_group import Translation, Permutation, TranslationGroup


def serialize_translation(t):
    if module_version("netket") >= (3, 18, 0):
        return {
            "inverse_permutation_array": t.inverse_permutation_array.tolist(),
            "displacement": t._vector.tolist(),
        }
    else:
        return {
            "permutation": t.permutation.wrapped.tolist(),
            "displacement": t._vector.tolist(),
        }


register_serialization(Translation, serialize_translation)


def serialize_permutation(t):
    if module_version("netket") >= (3, 18, 0):
        return {
            "inverse_permutation_array": t.inverse_permutation_array.tolist(),
            "name": t._name,
        }
    else:
        return {
            "permutation": t.permutation.wrapped.tolist(),
            "name": t._name,
        }


register_serialization(Permutation, serialize_permutation)


# groups
def serialize_translationgroup(tg):
    return {"lattice": tg.lattice, "axes": tg.axes}


def deserialize_translationgroup(obj):
    if "elems" in obj:
        # this was serialized with an old version of ketnet
        from netket.graph.space_group import PermutationGroup

        return PermutationGroup(**obj)
    else:
        return TranslationGroup(**obj)


register_serialization(
    TranslationGroup,
    serialize_translationgroup,
    deserialization_fun=deserialize_translationgroup,
)

try:
    from netket.utils.model_frameworks.nnx import NNXWrapper
except ModuleNotFoundError:
    raise ImportError(
        "This version of netket pro requires a more recent netket version. Update NETKET! (from github)"
    )

register_automatic_serialization(NNXWrapper, "graphdef")

# mcstate
from netket.vqs import MCMixedState, MCState, FullSumState
from netket.jax.sharding import shard_along_axis


def _replicate(x):
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return jax.lax.with_sharding_constraint(x, replicate_sharding())
    return x


# For model states using frameworks that
def _sort_numeric_string_keys(d):
    """Recursively sort dicts with numeric string keys into lists to restore
    the original structure that ``to_state_dict`` converted from lists to dicts.

    ``flax.serialization.to_state_dict`` converts lists to dicts with string
    keys (e.g. ``{'0': a, '1': b, ...}``).  When the number of entries >= 10,
    alphabetical iteration order (``'0','1','10','11',...,'2',...``) no longer
    matches the original numerical order, causing ``jax.tree.flatten`` to
    return leaves in the wrong order.  This helper restores the list form so
    that ``jax.tree.flatten`` produces leaves in the correct order.
    """
    if isinstance(d, dict):
        d = {k: _sort_numeric_string_keys(v) for k, v in d.items()}
        # If all keys are non-negative integer strings, convert to a list
        # sorted by the numeric value.
        if d and all(k.isdigit() for k in d.keys()):
            sorted_items = sorted(d.items(), key=lambda kv: int(kv[0]))
            return [v for _, v in sorted_items]
    elif isinstance(d, (list, tuple)):
        return type(d)(_sort_numeric_string_keys(v) for v in d)
    return d


def _unpack_variables(state_dict, obj):
    if "variables_structure" in obj:
        variables_sd = _sort_numeric_string_keys(state_dict["variables"])
        variables_flat, _ = jax.tree.flatten(variables_sd)
        variables = jax.tree.unflatten(obj["variables_structure"], variables_flat)
        del obj["variables_structure"]
    else:
        variables = state_dict["variables"]
    return variables


def _serialize_model_field(state):
    """Choose what to serialize for a variational state's model.

    By default we write ``state._model`` (the framework-native static model: a
    flax module, an nnx graphdef wrapper, an equinox static partition, ...),
    exactly as before.

    However, some frameworks wrap the user's model into an opaque static
    container (``flax.nnx`` -> graphdef, ``equinox`` -> static partition) that
    hides the original class. If a custom serializer has been registered for
    that original class, we instead serialize ``state.model`` -- the *unwrapped*
    model -- so that the registered serializer fires. This lets packages provide
    a relocatable serialization (e.g. bundling source/config) for their models
    even when netket would otherwise store a graphdef referencing the class by
    import path.

    Models whose concrete class has no registered serializer are completely
    unaffected: the check is on the class (no unwrap is triggered) and the
    returned object is ``state._model`` as before.
    """
    framework = getattr(state, "_model_framework", None)
    static_model = state._model
    if framework is not None:
        concrete_cls = framework.wrapped_model_class(static_model)
    else:
        concrete_cls = type(static_model)

    if has_custom_serializer(concrete_cls):
        # unwrap (e.g. nnx.merge / eqx.combine) so the registered serializer for
        # `concrete_cls` is dispatched on the real model instance.
        return state.model
    return static_model


def serialize_mcstate(
    state: MCState,
) -> dict:
    asset_manager = current_context().asset_manager

    # If saving sample cache, and the cache is there, save it.
    save_samples = (
        current_context().option("save_sample_cache") and state._samples is not None
    )

    state_dict = serialization.to_state_dict(state)
    if save_samples:
        # netket stores `_sampler_state_previous` to reproduce the cached samples
        # on reload; since we store the samples, keep the consistent current state.
        state_dict["sampler_state"] = serialization.to_state_dict(state.sampler_state)
    state_dict = jax.tree.map(_replicate, state_dict)
    variables_structure = jax.tree.structure(state.variables)
    asset_manager.write_msgpack("state.msgpack", state_dict)

    if save_samples:
        asset_manager.write_msgpack(
            "samples.msgpack", {"samples": _replicate(state._samples)}
        )

    return {
        "sampler": state.sampler,
        "model": _serialize_model_field(state),  # bare model (or relocatable export)
        "variables_structure": variables_structure,
    }


def deserialize_vstate(
    cls,
    obj,
) -> MCState:
    asset_manager = current_context().asset_manager

    state_dict = asset_manager.read_msgpack("state.msgpack")
    variables = _unpack_variables(state_dict, obj)
    state = cls(**obj, variables=variables)
    state = serialization.from_state_dict(state, state_dict)

    # Self-describing: the samples cache is present iff it was saved with
    # `save_sample_cache=True`. The restored sampler state is the one consistent
    # with these samples (see `serialize_mcstate`).
    if asset_manager.has_asset("samples.msgpack"):
        samples = jnp.asarray(asset_manager.read_msgpack("samples.msgpack")["samples"])
        state._samples = shard_along_axis(samples, axis=0)

    return state


register_serialization(
    MCState,
    serialize_mcstate,
    partial(deserialize_vstate, MCState),
    options=[
        SaveOption(
            "save_sample_cache",
            bool,
            default=False,
            doc="Store the cached Monte-Carlo samples (`state._samples`) if present "
            "so they need not be regenerated after reload. **Does not change the samples "
            "obtained after reloading**. Just the speed!",
        ),
    ],
)


def serialize_mcmixedstate(state: MCMixedState) -> dict:

    asset_manager = current_context().asset_manager

    state_dict = serialization.to_state_dict(state)
    state_dict = jax.tree.map(_replicate, state_dict)
    asset_manager.write_msgpack("state.msgpack", state_dict)

    return {
        "sampler": state.sampler,
        "sampler_diag": state.diagonal.sampler,
        "model": _serialize_model_field(state),  # bare model (or relocatable export)
    }


def deserialize_mcmixedstate(obj) -> MCMixedState:
    asset_manager = current_context().asset_manager

    state_dict = asset_manager.read_msgpack("state.msgpack")
    variables = _unpack_variables(state_dict, obj)
    state = MCMixedState(**obj, variables=variables)
    state = serialization.from_state_dict(state, state_dict)
    return state


register_serialization(MCMixedState, serialize_mcmixedstate, deserialize_mcmixedstate)


def serialize_fullsumstate(state: FullSumState, *, mixed_state: bool = False) -> dict:
    asset_manager = current_context().asset_manager

    state_dict = serialization.to_state_dict(state)
    state_dict = jax.tree.map(_replicate, state_dict)
    asset_manager.write_msgpack("state.msgpack", state_dict)

    if not mixed_state:
        hilbert = state.hilbert
    else:
        hilbert = state.hilbert.physical

    return {
        "hilbert": hilbert,
        "model": _serialize_model_field(state),  # bare model (or relocatable export)
    }


register_serialization(
    FullSumState, serialize_fullsumstate, partial(deserialize_vstate, FullSumState)
)


# HashableArray
from netket.utils import HashableArray


def serialize_hashable_array(obj):
    asset_manager = current_context().asset_manager
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(obj))
    asset_manager.write_asset("array.npy", buffer.getvalue())
    return {}


def deserialize_hashable_array(obj):
    asset_manager = current_context().asset_manager
    array = np.load(io.BytesIO(asset_manager.read_asset("array.npy")))
    return HashableArray(array)


register_serialization(
    HashableArray,
    serialization_fun=serialize_hashable_array,
    deserialization_fun=deserialize_hashable_array,
)
