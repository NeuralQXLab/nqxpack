# Saving and Loading variational states, and other NetKet objects

A common paint point I've experienced and seen many people live through is how to save/load a variational state generated with NetKet.

A common issue is that with the current serialisation logic of NetKet, based on ``flax.serialisation``, we can serialise the data inside of a variational state but it is a requirement to separately store some way to regenerate the python object itself, such as the network class and hyperparameters, sampler settings, Hilbert space, simmetries and much more.

Moreover, a common issue is that when PhD students change something in their scripts, they might no longer be able to load old data files.
People coming from QuTiP often miss the beauty of ``Qobj.save('filename')/load('filename')``, which allows them to save/load any object with ease.

## Goal

Implement a mechanism to save/load Variational States, which store both the numerical data (weights and sampler state) as well as how to reconstruct the objects, such that it would be possible to save/load as follows:

```python
vs = MCState(....)
# train it...

vs.save("my_state.nk")

# -- kill python
import netket as nk
vs = nk.vqs.load("my_state.nk")
```

A goal would be to have the format as stable as possible, to make it easy to

## Inspiration and ideas

- ``Pickle`` : Built-in serialisation/deserialisation library of Python. Can serialise whole objects, but struggles with closures which are omnipresent in flax networks and must therefore be excluded.
- ``CloudPickle``: Improvement over pickle that can serialise basically everything you throw at it. It is extremely reliable because it is used by cloud computing solutions at Meta. Issues: 
  - It is not compatible between minor Python versions (3.11, 3.12, 3.13...) which somewhat defeats the purpose of a serialisation scheme.
  - Binary format that is not easily readable.
- ``Keras.saving`` : can serialise its own objects. Works by having a `_to_dict` and `_from_dict` method on every Keras object, and recursively calling those functions. Issues:
  - Cannot save at the same time both 'metadata' and 'numerical data', as this solution only stores the metadata.
  - Requires 'ownership' and control over the root classes, which we do not have because we do not own ``flax`` or ``equinox``.
  - Human readable format (json)

Pickling seems to be impossible to support properly, while Keras solution seems good enough are relatively stable.

## Proposed Solution

We can reimplement something like ``Keras`` serialisation system by leveraging the fact that most classes in NetKet are Jax PyTrees, so we already know how to pack/unpack them.

Subgoals:
    - Make the format human-readable, and possibly human-editable;
    - Keep the format easy;
    - Leverage the existing infrastructure to extract numerical data from netket classes;
    - Requires as little 'custom serialisation routines' as possible;
    - Compatible among Python versions;

To keep the format simple, we should not serialise code or binary objects, but instead only have a way to recreate them from scratch. This means that to be sure to get back the same file upon loading one must have the same dependencies he used when serialising the code, or at least 'sufficiently compatible versions' of packages. We could store a ``requirements.txt`` file in the save files if needed to allow easy reproducibility in the future.

Requirements:
    - 

## Description of the format

The file format will be a **standard zip archive** with file extension ``.nk`` which will contain the following files:

- ``metadata.json``: file containing general metadata such as format version, netket version, etc necessary to validate whether we can load the file, or to help the user understand why a file cannot be loaded.
- (maybe?) ``requirements.txt`` file with the specific versions used to save the file.
- ``config.json`` A human-readble json object that can be deserialised into the original file. This will follow a format inspired by Hydra configuration files, but customised to support serialising functions, closures and custom types.
- ``assets/**.mpack`` Some msgpack-encoded file with the standard flax-provided serialisation of the variational state, as well as other objects that might arise which are too heavy to be stored in a human readable json file.

### Json file format

The json file ``config.json`` should contain enough information to reconstruct the former objects. However, json naturally only supports **dictionaries, lists, strings, integers and floats** so we must convert everything to those formats. The idea is to encode objects in the following way:

- ``str``, ``ìnt``, ``float``, ``complex`` are natively supported
- ``dict``, ``list``: natively supported, and their content is recursively encoded
- ``tuples``, as custom types, see below. We cannot serialise them as lists as lists are not hashable, and it would break most of flax models.
- ``functions``: encoded as the special string ``"< module.my_fun >"``, for example ``jnp.zeroes`` would be encoded as ``"< jax.numpy.zeroes>"``. The ``<  >`` act as delimiters to identify those special strings which get deserialised to a function object.
- ``types`` (or classes, not class instances): encoded like a function above.
- ``np.dtype``: encoded as their string name ``dtype.name``. For example ``jnp.float64`` is encoded as ``"float64"``, which is supported by numpy/jax. They could also be encoded as a custom type, but we prefer to make them more readable. Note that this will break identity of serialised object with the original, but it's not a problem.

### Custom types

Custom types can be serialised according to one of the following 2 ways:

#### Simple format

A way to serialise a custom type is to encode the arguments necessary to reconstruct it. An example is a netket Spin hilbert space which could be encoded as follows

```python
hilb_dict = {
    "__target__": "netket.hilbert.Spin",
    "S": 0.5, 
    "N":10,
    "total_sz": 0.5
}
# translates to
hilb = nk.hilbert.Spin(S=0.5, N=10, total_sz=0.5)
```

It is also possible to pass positional arguments by specifying the key `__args__`. 
This is used for example to serialise tuples:

```python
my_tuple_dict = {
    "__target__": "builtin.tuple",
    "__args__": [
        [1,2,3],
        ]
}
# translates to
my_tuple = ([1,2,3])
```

### Dataclasses

Standard python dataclasses, which test true to ``dataclasses.is_dataclass(obj)``, are serialised by taking their fields implementation and recursively serialising the arguments.

```python
if dataclasses.is_dataclass(obj):
    # no: this recurses
    # dict_data = dataclasses.asdict(obj)
    dict_data = {"_target_": _qualname(obj)}
    for field in dataclasses.fields(obj):
        dict_data[field.name] = serialize_object(
            getattr(obj, field.name),
            path=path + (field.name,),
            asset_manager=asset_manager,
        )
    return dict_data
```

The deserialisation will simply call the class constructor ("_target_") directly with all serialized fields, as this is guaranteed to work by the dataclass protocol.

```python
constructor = resolve_qualname(dict_data.pop('_target_'))
constructor(**dict_data)
```

### Custom types

Custom types which are not dataclasses can be serialised by means of a custom function that returns a dictionary of objects that can be serialised and a specific target. 2 possibilities are given for the target:

- If the constructor of the type can be used to reconstruct the starting object, then the target should be the class name, and the dictionary should contain the objects that are passed to the constructor

```python 
class MyClass():
    def __init__(self, a, b):
        self.a = a
        self.b = b

def serialize_myclass(obj):
    return {
        "target": "MyClass"
        "a": obj.a
        "b": obj.b
    }
```
In this case no deserialization funciton is defined, as the class constructor is enough. Note that the same logic used for dataclasses can be used here, simplifying the deserialization, as we are just executing the given function/class.

- If the constructor of the type cannot be used to reconstruct the serialised object, the target will be a special sentinel value which signals that a custom deserialisation function will be called.
  - The sentinel value is ``"# qualified.class.name"``

```python
class MyClass():
    def __init__(self, a, seed):
        self.a = a
        self.b = np.random.rand(seed) 

def serialize_myclass(obj):
    return {
        # in this case the target is inserted automatically by our registry logic to avoid
        # the user having to deal with our sentinel values
        #"target": "# MyClass #",
        "a": obj.a
        "b": obj.b
    }

def deserialize_myclass(dict):
    obj = MyClass(dict['a'], 0)
    obj.b = dict['b']
    return obj
```

Some mechanism to register serialisation and deserialisation functions will be implemented.

- If the type cannot easily be serialized to a json dictionary because it contains large numeric arrays, it can write out some asset files. To this end, the serialize and deserialize classes can accept a `path` and `asset_manager` kwarg, which will be used to store or retrieve the content of a large binary blob.
  - The `path` is a tuple of strings object used to indicate the path of the object we are serialising in the stack. This is useful both to save the files to a path (imagine we will be saving two objects of the same type, but stored in different locations of the dictionary), but also to throw informative errors pointing to exactly where we found a type we could nto serialise.
  - The `assetmanager` is a special class type that can store the assets somewhere. Main implementations will be on folders, or in an archive.

```python
class MyFatClass:
    def __init__(self, a):
        self.a = a
        self.b = np.random.normal(shape=(100,100))

def serialize_fatclass(obj: MyFatClass, *, path: tuple[str, ...], asset_manager: "AssetManager"):

    state_dict = {"b": obj.b}
    asset_manager.write_msgpack("data.msgpack", state_dict, path=path)

    return {
        # in this case the target is inserted automatically by our registry logic to avoid
        # the user having to deal with our sentinel values
        #"target": "# MyFatClass #",
        'a':obj.a
    }

def deserialize_vstate(
    dict, *, path: tuple[str, ...], asset_manager: "AssetManager"
) -> MCState:
    from flax import serialization

    state_dict = asset_manager.read_msgpack("data.msgpack", path=path)

    obj = MyFatClass(**dict)
    obj.b = state_dict["b"]
    return obj
````

### Closures

Closures cannot be automatically serialised. For that reason, when a closure is encountered it will be serialised following the same procedure of custom types: a serialization function will extract all local variables captured in the closure and return a dictionary of quantities to be serialised, while a deserialization function will recomnstruct the starting closure.

A specific registry of serialisable closures, as well as a simple mechanism to write the serialisation/deserialisation function of closures will be implemented.

Example:

```python

obj = jax.initializers.normal(stddev=0.01, mean=0.5)

# this can be detected to be a closure by calling
# is_closure = isinstance(obj, FunctionType) and obj.__closure__ is not None

# the qualified name will be
# normal.<locals>.init

# closure captured vars are stored in __closure__ in the same order as co_freevars
closure_vars = obj.__closure__

# target is the original function that built the closure, if possible
res = {"_target_": parent_function_name(obj) } # "jax.initializers.normal"}
# this contains the names of the captured variables
for i, captured_var_name in enumerate(obj.__code__.co_freevars):
    constructor_var_name = vars_mapping.get(captured_var_name, captured_var_name)
    res[captured_var_name] = closure_vars[i].cell_contents

# results in 
res = {
    "_target_": "jax.initializers.normal"
    "stddev" : 0.01,
    "mean" : 0.5,
}

```

