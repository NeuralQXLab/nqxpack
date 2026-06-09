# Per-call serialization options — design proposal

> Supersedes the exploratory notes in `serialization_options.md`. The transport
> mechanism (an *ephemeral* `options` carried on the packing context, never
> persisted) is taken from those notes unchanged. What this proposal adds is the
> piece the notes left open: options are **declared at registration**, so nqxpack
> owns a catalogue it can validate against, document, and warn on.

## Decisions

1. **Declared at registration.** A (de)serializer declares the options it accepts
   (name, type, default, one-line doc) when it is registered. nqxpack builds a
   catalogue from these declarations.
2. **Flat namespace, same-name = same knob.** Option names live in one flat
   namespace. Several types declaring `include_samples` with a *compatible*
   `(type, default)` are one logical knob honoured by all of them. An
   *incompatible* redeclaration (same name, different type/default/direction) is a
   registration-time error. Package prefixes (`netket.…`) are an optional
   convention to avoid accidental clashes, not a requirement.
3. **Core `options=` dict.** `save()`/`load()` take a single `options: dict`.
   Plugins may ship typed wrappers (`netket.save(vs, path, include_samples=True)`)
   that just build the dict — that ergonomic sugar lives on the plugin side.
4. **Validation.** An unknown key is an error (with a did-you-mean). A value whose
   type disagrees with the declaration is an error. A key declared but not read
   during the call warns *if it is a `SaveOption`* and is silent if it is a
   `LoadOption` (defensive load-time policy flags are expected to go unused).
5. **Direction.** Declarations are `SaveOption` or `LoadOption`. `save()` accepts
   only save options, `load()` only load options; misplacement is caught.
6. **Read API.** Inside a (de)serializer, `option('name')` returns the user value
   or the declared default. It is scoped to the *currently dispatching*
   serializer's declarations — reading an option this serializer did not declare
   raises (catches forgot-to-declare and keeps the catalogue authoritative).
7. **Self-describing load.** Save-time payload is recovered on load by probing for
   the asset (`AssetManager.has_asset`), not by reading back a flag. Only `save()`
   needs the toggle.

`options` is never written to the archive. The file format is unchanged; archives
written with no options are byte-identical to today's.

---

## Part 1 — Example usage

### 1a. End user

```python
import nqxpack

# Opt into heavy/optional payload at save time.
nqxpack.save(vs, "state.nk", options={"include_samples": True})

# Authorize a load-time policy (e.g. executing source bundled in the archive
# to reconstruct a class that moved). Off by default — safe by default.
model = nqxpack.load("state.nk", options={"allow_bundled_code": True})

# Discover what is tunable (built from the catalogue, no plugin docs needed).
nqxpack.list_options("save")   # -> include_samples (bool, default False): "Store ..."
nqxpack.list_options("load")   # -> allow_bundled_code (bool, default False): "Permit ..."
```

Validation the user actually feels:

```python
nqxpack.save(vs, "x.nk", options={"inculde_samples": True})
# KeyError: Unknown option 'inculde_samples'. Did you mean 'include_samples'?

nqxpack.save(vs, "x.nk", options={"allow_bundled_code": True})
# ValueError: 'allow_bundled_code' is a load-time option; it has no effect on save().

nqxpack.save(vs, "x.nk", options={"include_samples": "yes"})
# TypeError: option 'include_samples' expects bool, got str.

nqxpack.save(np.array([1, 2, 3]), "x.nk", options={"include_samples": True})
# UserWarning: option 'include_samples' was set but no serializer used it this
#              call (did you mean to save a variational state?).

nqxpack.load("state.nk", options={"allow_bundled_code": True})  # class resolved fine
# (no warning — defensive load-time flags are expected to often go unused)
```

Optional plugin sugar (lives in netket, not nqxpack):

```python
netket.save(vs, "state.nk", include_samples=True)   # -> nqxpack.save(..., options={...})
```

### 1b. API-extension developer

**Declaring a save toggle and acting on it.** The serializer reads the toggle; the
deserializer reads *nothing* — it branches on whether the asset is present.

```python
from nqxpack import register_serialization, SaveOption, option
from nqxpack._src.contextmgr import current_context

def serialize_mcstate(state):
    am = current_context().asset_manager
    state_dict = jax.tree.map(_replicate, serialization.to_state_dict(state))
    am.write_msgpack("state.msgpack", state_dict)

    if option("include_samples"):                 # default False, taken from the decl
        am.write_msgpack("samples.msgpack", {"samples": state.samples})

    return {
        "sampler": state.sampler,
        "model": _serialize_model_field(state),
        "variables_structure": jax.tree.structure(state.variables),
    }

def deserialize_vstate(cls, obj):
    am = current_context().asset_manager
    state_dict = am.read_msgpack("state.msgpack")
    variables = _unpack_variables(state_dict, obj)
    state = cls(**obj, variables=variables)
    state = serialization.from_state_dict(state, state_dict)

    if am.has_asset("samples.msgpack"):            # self-describing — no flag read
        state.samples = am.read_msgpack("samples.msgpack")["samples"]
    return state

register_serialization(
    MCState,
    serialize_mcstate,
    partial(deserialize_vstate, MCState),
    options=[
        SaveOption(
            "include_samples", bool, default=False,
            doc="Store the cached Monte-Carlo samples so they need not be "
                "regenerated after reload. Increases file size.",
        ),
    ],
)
```

**One knob shared by several types.** Declare the spec once and pass the same
object to each registration. The flat-namespace rule makes it a single
user-facing knob honoured by all three.

```python
SAMPLES = SaveOption("include_samples", bool, default=False, doc="...")

register_serialization(MCState,      serialize_mcstate,      ..., options=[SAMPLES])
register_serialization(MCMixedState, serialize_mcmixedstate, ..., options=[SAMPLES])
register_serialization(FullSumState, serialize_fullsumstate, ..., options=[SAMPLES])
# user sets include_samples once; all three serializers honour it.
```

**A load-time policy flag on a deserializer.**

```python
from nqxpack import LoadOption, option

def _import_model(obj):
    cls = _resolve_or_none(obj["_target_"])        # try the safe path first
    if cls is not None:
        return _reconstruct(cls, obj)
    if option("allow_bundled_code"):               # only then consult policy
        return _exec_bundled_and_reconstruct(obj)
    raise ClassNotFoundError(obj["_target_"])

register_serialization(
    MyRelocatableModel, serialize_model, _import_model,
    options=[
        LoadOption(
            "allow_bundled_code", bool, default=False,
            doc="Permit executing source bundled in the archive to reconstruct "
                "a class that can no longer be imported. Unsafe: trusted files only.",
        ),
    ],
)
```

**Mistakes the developer is protected from.** `option('foo')` inside
`serialize_mcstate` where `foo` was not declared by that registration raises
immediately — the declaration and the reads cannot silently drift apart.

---

## Part 2 — Internal design

### Option specs and the catalogue

```python
# nqxpack/_src/lib_v1/options.py
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class OptionSpec:
    name: str
    type: type
    default: Any
    doc: str
    direction: str            # "save" | "load"

def SaveOption(name, type, default, doc=""):
    return OptionSpec(name, type, default, doc, "save")

def LoadOption(name, type, default, doc=""):
    return OptionSpec(name, type, default, doc, "load")

# name -> merged spec. Built incrementally at registration time.
OPTION_CATALOGUE: dict[str, OptionSpec] = {}

def _register_option(spec: OptionSpec):
    existing = OPTION_CATALOGUE.get(spec.name)
    if existing is None:
        OPTION_CATALOGUE[spec.name] = spec
    elif (existing.type, existing.default, existing.direction) != (
        spec.type, spec.default, spec.direction
    ):
        raise ValueError(
            f"Incompatible redeclaration of option {spec.name!r}: "
            f"{existing} vs {spec}."
        )
    # compatible same-name declaration -> one logical knob, nothing else to do.
```

### Registration carries declared options

`register_serialization` gains `options: list[OptionSpec] | None = None`. It

1. validates each spec into `OPTION_CATALOGUE` via `_register_option`;
2. partitions the registration's names into a save-set and a load-set;
3. makes those sets available to the dispatch wrappers so they can push scope.

```python
def register_serialization(cls, serialization_fun, deserialization_fun=None,
                           *, options=None, ...):
    save_specs = {s.name: s for s in (options or []) if s.direction == "save"}
    load_specs = {s.name: s for s in (options or []) if s.direction == "load"}
    for s in (options or []):
        _register_option(s)

    # serialize wrapper pushes save scope around the user function
    @wraps(serialization_fun)
    def _serialize_fun(obj):
        with current_context().option_scope(save_specs):
            dict_data = serialization_fun(obj)
        ...   # existing _target_ handling unchanged
    TYPE_SERIALIZATION_REGISTRY[cls] = _serialize_fun

    if deserialization_fun is not None:
        @wraps(deserialization_fun)
        def _deser(obj):
            with current_context().option_scope(load_specs):
                return deserialization_fun(obj)
        register_deserialization(cls, _deser, ...)
```

Scope is pushed *around the user function body only*. Note that nested objects are
recursed over by `serialize_object` **after** `serialize_custom_object` returns
(see `lib.py`: `serialize_object(serialize_custom_object(obj))`), so a nested
serializer never sees an outer serializer's scope — the stack stays correct
without extra care.

### Context additions

```python
class PackingContext:
    def __init__(self, asset_manager=None, metadata=None,
                 options=None, mode="save"):
        ...
        self._options = options or {}     # validated user values (no defaults filled)
        self._mode = mode                 # "save" | "load"
        self._read_options: set[str] = set()
        self._option_scope: list[dict[str, OptionSpec]] = []

    @contextmanager
    def option_scope(self, specs: dict[str, OptionSpec]):
        self._option_scope.append(specs)
        try:
            yield
        finally:
            self._option_scope.pop()

    def _current_scope(self) -> dict[str, OptionSpec]:
        return self._option_scope[-1] if self._option_scope else {}
```

### The read helper

```python
# nqxpack/_src/lib_v1/options.py
def option(name: str):
    ctx = current_context()
    scope = ctx._current_scope()
    if name not in scope:
        raise LookupError(
            f"option({name!r}) was read but the current serializer did not "
            f"declare it. Add it to the `options=[...]` of its registration."
        )
    ctx._read_options.add(name)
    return ctx._options.get(name, scope[name].default)
```

### Entry-point validation (in `save`/`load`)

```python
import difflib, warnings

def _validate_options(options: dict, mode: str) -> dict:
    for key, value in options.items():
        spec = OPTION_CATALOGUE.get(key)
        if spec is None:
            hint = difflib.get_close_matches(key, OPTION_CATALOGUE, n=1)
            suffix = f" Did you mean {hint[0]!r}?" if hint else ""
            raise KeyError(f"Unknown option {key!r}.{suffix}")
        if spec.direction != mode:
            other = "load" if mode == "save" else "save"
            raise ValueError(
                f"{key!r} is a {other}-time option; it has no effect on {mode}()."
            )
        if not isinstance(value, spec.type):
            raise TypeError(
                f"option {key!r} expects {spec.type.__name__}, "
                f"got {type(value).__name__}."
            )
    return options

def _warn_unused(ctx):
    # only save options warn; load policy flags are expected to go unused.
    if ctx._mode != "save":
        return
    unused = set(ctx._options) - ctx._read_options
    for key in sorted(unused):
        warnings.warn(
            f"option {key!r} was set but no serializer used it this call.",
            UserWarning, stacklevel=3,
        )
```

`save()` / `load()` wiring:

```python
def save(object, path, *, zip=True, options=None):
    options = _validate_options(options or {}, "save")
    ...
    with PackingContext(asset_manager=..., options=options, mode="save") as ctx:
        object_json = serialize_object(object)
        ...
        _warn_unused(ctx)

def load(path, *, options=None):
    options = _validate_options(options or {}, "load")
    ...
    with PackingContext(asset_manager=..., metadata=metadata,
                        options=options, mode="load") as ctx:
        state = deserialize_object(state_obj_dict)
        # no unused-warning on load
```

### `AssetManager.has_asset` (small prerequisite)

```python
class AssetManager(ABC):
    @abstractmethod
    def _has(self, key: str) -> bool: ...

    def has_asset(self, asset_name, path=None) -> bool:
        if path is None:
            path = current_context().path
        return self._has(f"{path}/{asset_name}")

# InMemory:  return key in self._assets
# Folder:    return (self.folder / _resolved(key)).exists()
# Archive:   return _resolved(key) in self.archive.namelist()
```

### Discoverability

```python
def list_options(direction: str | None = None) -> list[OptionSpec]:
    return [s for s in OPTION_CATALOGUE.values()
            if direction is None or s.direction == direction]
```

Re-exported as `nqxpack.list_options`, `nqxpack.SaveOption`, `nqxpack.LoadOption`,
`nqxpack.option`.

---

## Backward compatibility

- `options=` defaults to `None` on `save()`/`load()`; `options=` defaults to
  `None` on `register_serialization`. Existing calls and existing (de)serializers
  are unchanged.
- The catalogue is empty for serializers that declare nothing; `option()` is never
  called by them.
- Nothing is persisted: the archive format does not change.

## Notes / follow-ups (not blocking)

- **Distributed.** `options` is plain Python passed identically to every process,
  so it needs no special handling. The *samples* payload interacts with sharding;
  `write_msgpack` already guards `process_index == 0` and the state path uses the
  existing `replicate_sharding` path — validate `include_samples` against a
  sharded state before shipping.
- **`min_version` interaction.** Versioned deserializers registered via
  `register_deserialization` should accept (and forward) the same `options=` so an
  old-format loader can still honour a `LoadOption`. Trivial: thread the same
  load-scope wrapper there.
