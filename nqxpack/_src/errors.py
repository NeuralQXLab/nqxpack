class MainScopeError(Exception):
    def __init__(self, name):
        super().__init__(
            f"""
            Impossible to serialize an object of type {name}.

            You cannot serialize objects defined in the main scope.
            Please define them in a module, and import the module.

            Of course, you will have to keep the module accessible to reload this file
            """
        )


class SerializationError(Exception):
    def __init__(self, typename, obj, path):
        super().__init__(
            f"""
            Impossible to serialize an object of type `{typename}` found at path
            `{path}`.

            You need to implement a custom serialization function for this object which
            converts in built in types or other serializable types, such as lists, integers
            dictionaries...

            To do it, you should add a new entry to the registry in the file
            in nqxpack._src.lib_v1.registry.

                from nqxpack._src.lib_v1 import (
                    register_serialization,
                    )

                def serialization_function(obj: typename) -> serializable_type:
                    ...

                register_serialization(
                    {typename}, serialization_function,
                    )

            """
        )


class JaxArraySerializationError(SerializationError):
    def __init__(self, obj, path):
        path = path or "<root>"
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)

        Exception.__init__(
            self,
            f"""
            Impossible to serialize a JAX array found at path
            `{path}`.

            nqxpack only supports NumPy arrays. Convert this array, or the pytree
            containing it, to NumPy before calling `nqxpack.save`.

            For example:
                array_np = np.asarray(array)
                pytree_np = jax.tree.map(np.asarray, pytree)

            Array details:
                shape: {shape}
                dtype: {dtype}
            """,
        )


class FutureVersionError(Exception):
    def __init__(self, file_version, max_version):
        super().__init__(
            f"""
            File was saved with a more recent version of nqxpack/netket_pro than your current installation:

                File version      : {file_version}
                Supported version : {max_version}

            Please update the library to a version that supports this file format.
            """
        )
