from nqxpack._src.lib_v1 import (
    serialize_object,
    deserialize_object,
)

from omegaconf import DictConfig, ListConfig

from .. import common


@common.skipif_distributed
def test_list_and_dict_conversion():
    d = {"a": 1, "b": [1, 2, 3]}
    d_cfg = DictConfig(d)
    l = [1.0, 2.0, 3.0]
    l_cfg = ListConfig(l)

    assert serialize_object(l) == serialize_object(l_cfg)
    assert serialize_object(d) == serialize_object(d_cfg)
    assert deserialize_object(serialize_object(l)) == l
    assert deserialize_object(serialize_object(d)) == d
