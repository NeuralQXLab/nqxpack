# NQXPack

A library to save and load objects coming from Scientific Machine Learning libraries, with a special attention to Neural Quantum States from NetKet.

Goals:
- Simple format, possible to hand-edit and inspect manually;
- Compatibility among Python version;
- Allows to load Neural Networks with a single ``load`` command;

## Usage

Install with 

```bash
uv add git+https://github.com/NeuralQXLab/nqxpack.git
```
or (but seriously, stop using pip and start using uv)
```bash
pip install git+https://github.com/NeuralQXLab/nqxpack.git
```


### With `flax.linen`

Save a dictionary containing the model and the parameters.
Note that you cannot serialise jax arrays for the time-being but it could easily be added (I'd need to think about how to handle sharding...)

```python
import nqxpack
import jax
from flax import linen as nn
import numpy as np

model = nn.Sequential((
    nn.Dense(features=2),
    nn.gelu,
    nn.Dense(features=1),
    jax.numpy.squeeze,
))

variables = model.init(jax.random.key(1), jax.numpy.ones((2,4)))
variables_np = jax.tree.map(np.asarray, variables)

# for the moment cannot serialise jax arrays.
# Could easily be implemented
nqxpack.save({'model':model, 'variables':jax.tree.map(np.asarray, variables)}, "mymodel.nk")

loaded_dict = nqxpack.load("mymodel.nk")
loaded_model, loaded_variables = loaded_dict['model'], loaded_dict['variables']
```

### With `flax.nnx` (WIP, not working yet)


```python
import nqxpack
import jax
from flax import nnx
import numpy as np

rngs = nnx.Rngs(0)
model = nnx.Sequential(
  nnx.Linear(1, 4, rngs=rngs),  # data
  nnx.Linear(4, 2, rngs=rngs),  # data
)

# Contrary to flax.linen, we need to declare how to serialise/deserialise
# every type of nnx layer. So for now we only support Sequential and Linear
# see registry/flax.py to add support for more layers (its easy)
nqxpack.save({'graphdef':graphdef, 'variables':variables_np}, "mymodel.nk")

loaded_dict = nqxpack.load("mymodel.nk")

loaded_model = nnx.merge(loaded_graphdef, loaded_variables)
```


### With NetKet

```python
import nqxpack
import netket as nk

hi = nk.hilbert.Spin(0.5, 10)
operator = nk.operator.spin.sigmax(nqs_state.hilbert, 1)

nqs_state = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM(alpha=4))
# print expectation value:
nqs_state.expect(operator)

nqxpack.save(nqs_state, "nqs_state.nk")
nqs_state_loaded = nqxpack.load("nqs_state.nk")

nqs_state_loaded.expect(operator)
```

## The format

The format is a single zip file. You can decompress it yourself and look into it.

## Feedback required

If you use this library, please let us know of any issue you might find.
