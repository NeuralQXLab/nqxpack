from nqxpack._src.lib_v1.closure import register_closure_simple_serialization
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
    register_deserialization,
)

import jax

register_closure_simple_serialization(
    jax.nn.initializers.normal,
    "init",
    original_qualname="jax._src.nn.initializers.normal",
)
register_closure_simple_serialization(
    jax.nn.initializers.variance_scaling,
    "init",
    original_qualname="jax._src.nn.initializers.variance_scaling",
)


def serialize_PyTreeDef(obj):
    return {
        "node_data": obj.node_data(),
        "children": obj.children(),
    }


def deserialize_PyTreeDef(obj):
    # Fall back to old method if available (for older JAX versions)
    if hasattr(jax.tree_util.PyTreeDef, "make_from_node_data_and_children"):
        return jax.tree_util.PyTreeDef.make_from_node_data_and_children(
            jax.tree_util.default_registry, obj["node_data"], obj["children"]
        )

    # Workaround for new JAX versions with old serialized data
    # We need to reconstruct the PyTreeDef from node_data and children
    return _reconstruct_pytreedef_from_legacy(obj)


def _reconstruct_pytreedef_from_legacy(obj):
    """
    Reconstruct a PyTreeDef from the legacy serialization format.

    This function implements the logic from the old make_from_node_data_and_children
    method that was removed from JAX in https://github.com/jax-ml/jax/commit/0e7c96a54a98d0f2b7e1c29bd7cc61b1b9bcbf59
    """
    node_data, children = obj["node_data"], obj["children"]
    import jax.tree_util as tu

    # If we have a leaf node
    if node_data is None:
        # Create a PyTreeDef for a leaf
        _, leaf_treedef = tu.tree_flatten(object())
        return leaf_treedef

    # node_data is a tuple of (node_type, metadata)
    node_type, metadata = node_data

    # First, recursively reconstruct all children PyTreeDefs
    reconstructed_children = []
    for child in children:
        if isinstance(child, dict) and "node_data" in child and "children" in child:
            # This is a serialized PyTreeDef that needs reconstruction
            child_treedef = _reconstruct_pytreedef_from_legacy(
                child["node_data"], child["children"]
            )
            reconstructed_children.append(child_treedef)
        else:
            # This is already a PyTreeDef
            reconstructed_children.append(child)

    # Now we need to build a template structure and extract dummy leaf values
    # to create properly structured nested data

    # Create dummy data for each child based on their treedefs
    child_leaves_list = []
    for child_treedef in reconstructed_children:
        num_leaves = child_treedef.num_leaves  # This is a property, not a method
        # Create dummy leaves for this child
        dummy_leaves = [object() for _ in range(num_leaves)]
        # Unflatten to get the properly structured child
        child_structure = tu.tree_unflatten(child_treedef, dummy_leaves)
        child_leaves_list.append(child_structure)

    # Build a template structure based on the node type and metadata
    if node_type.__name__ == "tuple" and hasattr(node_type, "_fields"):
        # NamedTuple
        template = node_type(*child_leaves_list)
    elif node_type is tuple:
        # Regular tuple
        template = tuple(child_leaves_list)
    elif node_type is list:
        # List
        template = list(child_leaves_list)
    elif node_type is dict:
        # Dict - metadata contains the keys
        if metadata is not None:
            template = {key: child for key, child in zip(metadata, child_leaves_list)}
        else:
            template = {}
    else:
        # Custom pytree type
        # Most custom pytree types can be created from their children
        template = tuple(child_leaves_list)

    # Get the treedef from the template
    _, template_treedef = tu.tree_flatten(template)

    return template_treedef


register_serialization(
    jax.tree_util.PyTreeDef, serialize_PyTreeDef, deserialize_PyTreeDef
)

# Register versioned deserialization for the old jaxlib path
# In older JAX versions, PyTreeDef was exposed as jaxlib.xla_extension.pytree.PyTreeDef
# Now it's jax.tree_util.PyTreeDef
register_deserialization(
    "jaxlib.xla_extension.pytree.PyTreeDef",
    _reconstruct_pytreedef_from_legacy,
    min_version=(0, 0, 0),
)
