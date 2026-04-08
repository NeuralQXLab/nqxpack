# Getting Started

## Installation

Install `nqxpack` with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add nqxpack
```

Or with pip:

```bash
pip install nqxpack
```

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/NeuralQXLab/nqxpack.git
```

### Requirements

- Python ≥ 3.11
- JAX ≥ 0.4.35
- Flax ≥ 0.10.2
- NetKet > 3.18

## Quick Example

The following saves and loads a Flax `linen` model together with its parameters:

```python
import nqxpack
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Build a small model and initialise it
model = nn.Sequential([nn.Dense(features=8), nn.gelu, nn.Dense(features=1)])
variables = model.init(jax.random.key(0), jnp.ones((1, 4)))

# nqxpack works with numpy arrays; convert from jax first
variables_np = jax.tree.map(np.asarray, variables)

# Save — the file gets a .nk extension automatically
nqxpack.save({"model": model, "variables": variables_np}, "checkpoint.nk")

# Load back — all types are reconstructed automatically
data = nqxpack.load("checkpoint.nk")
model_loaded = data["model"]
variables_loaded = data["variables"]
```

## NetKet Example

Save and restore a full `MCState` variational state:

```python
import nqxpack
import netket as nk

hi = nk.hilbert.Spin(0.5, N=10)
model = nk.models.RBM(alpha=2)
sampler = nk.sampler.MetropolisLocal(hi)
vqs = nk.vqs.MCState(sampler, model, n_samples=512)

# Compute the energy to verify we can compare before/after
ha = nk.operator.Ising(hi, nk.graph.Chain(10), h=1.0)
e_before = vqs.expect(ha).mean

nqxpack.save(vqs, "vqs.nk")

vqs_loaded = nqxpack.load("vqs.nk")
e_after = vqs_loaded.expect(ha).mean
```

## The File Format

An `.nk` file is a standard zip archive. You can inspect its contents directly:

```
unzip -l checkpoint.nk
```

Inside you will find:

| File | Contents |
|------|----------|
| `object.json` | JSON representation of the object tree (human-readable) |
| `metadata.json` | Format version and library versions at save time |
| `assets/` | Binary blobs for large arrays (NumPy `.npy` format) |

:::{note}
The format is forwards-incompatible: files saved with a *newer* version of nqxpack
cannot be loaded by an *older* version. An informative error is raised if this occurs.
:::

:::{warning}
Do not load `.nk` files from untrusted sources — the format can reconstruct arbitrary
Python objects.
:::

## Next Steps

- {doc}`tutorials/index` — detailed walkthroughs
- {doc}`api/index` — full API reference
