from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
)

# hydra
from omegaconf import ListConfig, DictConfig, OmegaConf

register_serialization(ListConfig, OmegaConf.to_object, reconstruct_type=False)
register_serialization(DictConfig, OmegaConf.to_object, reconstruct_type=False)
