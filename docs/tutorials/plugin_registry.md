# Supporting ``nqxpack`` in a package

If you want the types of your library to be supported by ``nqxpack``, you must write a registry file that declares how to serialize and deserialize your types as needed.

To ensure that loading files is always possible, even if your package is not loaded yet, nqxpack supports automatic
discovery via Python's standard **entry points** mechanism.


## Declaring an entry point

In your package's `pyproject.toml`, add:

```toml
[project.entry-points.nqxpack_registry]
mypackage = "mypackage._nqxpack_registry"
```

The **name** (`mypackage`) must be the importable name of the package whose types are being registered. 
The **value** is the module to import.

`mypackage/_nqxpack_registry.py` is a plain module with registration calls at the top
level:

```python
from nqxpack import register_serialization

def _serialize(obj): ...
def _deserialize(data): ...

register_serialization(MyType, _serialize, _deserialize)
```

After installation (e.g. `pip install -e .`), nqxpack picks this up automatically.

## Bridge-package pattern

If the serializers are shipped by a *different* package than the one that owns the
types (e.g. a `nqxpack-mypackage` bridge), use the type-owner's name as the entry
point name and the bridge module as the value:

```toml
# in mypackage's pyproject.toml — registers serializers for electroket types
[project.entry-points.nqxpack_registry]
electroket = "mypackage._src.nqxpack.electroket"
otherket = "mypackage._src.nqxpack.otherket"
```
