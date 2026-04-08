# Serializing Custom Classes

nqxpack handles plain Python types, NumPy arrays, and many JAX/Flax/NetKet objects out of
the box. This page explains what to do when you have a class that the library does not
know about yet — either your own or a third-party one.

## How objects are reconstructed

Every object is stored in `object.json` as a dict with a `_target_` key containing the
fully-qualified class path (e.g. `"mylib.models.MyModel"`). On load, nqxpack imports that
path and calls the class (or a registered factory) with the remaining fields as keyword
arguments.

Because reconstruction goes through a normal Python import, **objects defined inside
`__main__` (e.g. in a script or notebook) cannot be loaded back** — the class must be
importable from a proper module.

## Mechanisms at a glance

| Mechanism | Registration needed | Best for |
|---|---|---|
| `dataclasses.dataclass` | None | Your own simple data-holding classes |
| `__to_json__()` method | None | Your own classes with non-trivial construction |
| `register_automatic_serialization()` | Yes | Third-party classes whose attrs map directly to constructor args |
| `register_serialization()` | Yes | Full control: non-trivial construction, binary data |
| `register_deserialization()` | Yes | Backward compat: renamed/moved classes in old files |

---

## Option 1 — Dataclasses (zero setup)

Any standard Python dataclass is serialized automatically. All fields are stored and the
constructor is called with them on load.

```python
from dataclasses import dataclass
import nqxpack

@dataclass
class Config:
    learning_rate: float
    hidden_dim: int

nqxpack.save(Config(learning_rate=1e-3, hidden_dim=64), "config.nk")
cfg = nqxpack.load("config.nk")   # Config(learning_rate=0.001, hidden_dim=64)
```

This works as long as every field is itself serializable (plain types, arrays, other
registered types, or nested dataclasses).

---

## Option 2 — `__to_json__()` method (zero setup)

Add a `__to_json__()` instance method to your class. It should return a dict of
keyword arguments that will be passed back to the constructor on load. The `_target_`
key is added automatically.

```python
class Lattice:
    def __init__(self, shape: tuple[int, ...], periodic: bool = True):
        self.shape = shape
        self.periodic = periodic

    def __to_json__(self) -> dict:
        return {"shape": self.shape, "periodic": self.periodic}
```

On load, nqxpack calls `Lattice(shape=..., periodic=...)`.

This is the simplest option for classes you own, and requires no import of nqxpack in
your library code.

---

## Option 3 — `register_automatic_serialization()`

Use this for third-party classes (where you cannot add `__to_json__`) when the object's
constructor arguments are available as attributes.

```python
import nqxpack.registry as reg

class Optimizer:
    def __init__(self, lr: float, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum

# Serialize attrs 'lr' and 'momentum'; on load call Optimizer(lr=..., momentum=...)
reg.register_automatic_serialization(Optimizer, "lr", "momentum")
```

If the attribute name differs from the constructor argument name, use keyword syntax:

```python
# constructor takes 'size'; attribute is called 'N'
reg.register_automatic_serialization(MyHilbert, size="N")
```

Pass `array_to_list=True` if any attribute is a small NumPy/JAX array that should be
stored as a JSON list rather than a binary asset.

See {func}`nqxpack.registry.register_automatic_serialization` for the full signature.

---

## Option 4 — `register_serialization()` (full control)

When construction is non-trivial — e.g. the object holds state that cannot be recovered
from a few attributes, or you need to store binary data — provide explicit serializer
and deserializer functions.

```python
import nqxpack.registry as reg

class KMeans:
    def __init__(self, k: int):
        self.k = k
        self.centers = None   # set after fitting, a numpy array

def serialize_kmeans(model: KMeans) -> dict:
    return {"k": model.k, "centers": model.centers}

def deserialize_kmeans(data: dict) -> KMeans:
    m = KMeans(data["k"])
    m.centers = data["centers"]
    return m

reg.register_serialization(KMeans, serialize_kmeans, deserialize_kmeans)
```

The serializer returns a plain dict; nqxpack recursively serializes every value in it
(so `model.centers` — a NumPy array — is stored correctly as a binary asset
automatically).

### Storing large binary data explicitly

If you need direct control over binary storage (e.g. neural-network weights in
msgpack), access the {class}`nqxpack.registry.AssetManager` from the active context:

```python
from nqxpack._src.contextmgr import current_context

def serialize_model(model):
    am = current_context().asset_manager
    am.write_msgpack("weights.msgpack", model.state_dict())
    return {"architecture": model.arch}

def deserialize_model(data: dict):
    am = current_context().asset_manager
    state = am.read_msgpack("weights.msgpack")
    model = MyModel(data["architecture"])
    model.load_state_dict(state)
    return model
```

Asset names are automatically namespaced by the object's position in the JSON tree, so
two instances stored at different paths never collide.

---

## Option 5 — `register_deserialization()` (backward compatibility)

If a class was renamed or moved between library versions, old `.nk` files will contain
the old `_target_` path. Register a deserialization-only handler for the old path:

```python
from nqxpack._src.lib_v1.custom_types import register_deserialization

def load_old_optimizer(data: dict):
    from mylib.optim import NewOptimizer
    return NewOptimizer(**data)

register_deserialization(
    "mylib.old_module.OldOptimizer",
    load_old_optimizer,
    min_version=(0, 0, 0),   # apply for all file versions
)
```

The `min_version` tuple refers to the version of the package that **wrote** the file
(read from `metadata.json`). This lets you apply different deserializers depending on
when the file was saved.

---

## Closures

JAX-style initializers (functions returned by other functions) can be serialized with
{func}`nqxpack.registry.register_closure_simple_serialization`. This captures the
enclosing function's arguments and re-calls it on load. See the API reference for
details.
