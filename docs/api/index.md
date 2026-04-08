# API Reference

Full reference documentation for all public functions and classes in `nqxpack`.

## Top-level functions

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   nqxpack.save
   nqxpack.load
   nqxpack.serialize_object
   nqxpack.deserialize_object
```

## Registry

The `nqxpack.registry` submodule exposes helpers for extending the serialisation
system with custom types.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   nqxpack.registry.register_serialization
   nqxpack.registry.register_automatic_serialization
   nqxpack.registry.register_closure_simple_serialization
   nqxpack.registry.AssetManager
```

## Built-in registry modules

nqxpack ships with serialisers for the following libraries. They are imported
automatically when `nqxpack` is imported.

| Module | Covered types |
|--------|---------------|
| `nqxpack._src.registry.stdlib` | `functools.partial`, `frozenset`, `complex` |
| `nqxpack._src.registry.jax` | JAX arrays, dtypes |
| `nqxpack._src.registry.flax` | `flax.linen` modules, `flax.nnx` graph definitions |
| `nqxpack._src.registry.netket` | `MCState`, Hilbert spaces, samplers, models |
| `nqxpack._src.registry.netket_operator` | NetKet operators |
| `nqxpack._src.registry.hydra` | OmegaConf / Hydra config nodes |
