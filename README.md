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

### With `flax.nnx`

Within your code:
```python
import nqxpack
import jax
from flax import nnx

model = nnx.Sequential(
    nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(1)),
    nnx.gelu,
    nnx.Linear(in_features=2, out_features=1, rngs=nnx.Rngs(1)),
    jax.numpy.squeeze,
)

# Train it... and then check the output:
model(jax.numpy.ones((2,4)))
# Array([0.16373987, 0.16373987], dtype=float64)

nqxpack.save(model, "mymodel.nk")
```

# To load it:
```python
import nqxpack

model = nqxpack.load("mymodel.nk")
model(jax.numpy.ones((2,4)))
# Array([0.16373987, 0.16373987], dtype=float64)
```

### With `flax.linen`

Save a dictionary containing the model and the parameters.

```python
import nqxpack
import jax
from flax import linen as nn

model = nn.Sequential(
    nn.Dense(features=2),
    nn.gelu,
    nn.Dense(features=1),
    jax.numpy.squeeze,
)

variables = model.init(jax.random.key(1), jax.numpy.ones((2,4)))

nqxpack.save({'model':model, 'variables':variables}, "mymodel.nk")

loaded_dict = nqxpack.load("mymodel.nk")
loaded_model, loaded_variables = loaded_dict['model'], loaded_dict['variables']
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


