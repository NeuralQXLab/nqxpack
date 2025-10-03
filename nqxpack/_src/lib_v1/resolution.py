import re

from nqxpack._src.errors import (
    MainScopeError,
)
from nqxpack._src.contextmgr import current_context


def _register_package_version(module_name):
    import importlib.metadata

    # Extract the root package name (everything before the first dot)
    package_name = module_name.split(".")[0]

    PACKAGE_VERSIONS = current_context().package_versions

    if package_name in PACKAGE_VERSIONS:
        return PACKAGE_VERSIONS[package_name]

    # Look up the version of the root package
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        # Silently skip builtins and netket_pro since they don't have package metadata
        if package_name not in ("builtins", "netket_pro"):
            print(
                f"Package '{package_name}' is not installed, so you may not be able to reload this."
            )
        version = ""

    PACKAGE_VERSIONS[package_name] = version
    return version


def _qualname(obj, skip_register: bool = False):
    cls = obj if isinstance(obj, type) else type(obj)
    qname = cls.__module__ + "." + cls.__qualname__
    if cls.__module__.startswith("__main__"):
        raise MainScopeError(qname)

    if not skip_register:
        _register_package_version(cls.__module__)

    return qname


def _fname(fun, skip_register: bool = False):
    qname = fun.__module__ + "." + fun.__qualname__
    if fun.__module__.startswith("__main__"):
        raise MainScopeError(qname)

    if not skip_register:
        _register_package_version(fun.__module__)

    return qname


def _resolve_qualname(name: str):
    # this is to support parametric types
    match = re.match(r"([^\[]+)(?:\[(.+)\])?", name)
    if not match:
        raise ValueError(f"Invalid format: {name}")

    qualified_name = match.group(1)  # The part before the parametric brackets
    parametric_part = match.group(2)  # The content inside the brackets, if it exists

    if parametric_part is None:
        return _resolve_qualname_noparametric(qualified_name)
    else:
        qualified_name = _resolve_qualname_noparametric(qualified_name)
        parametric_part = _resolve_qualname_noparametric(parametric_part)
        return qualified_name[qualified_name]


def _resolve_qualname_noparametric(name: str):
    """
    Resolves a qualified name, guaranted not to have a parametric component.
    """

    # Split the qualified name into module and class
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    cls = parts[-1]

    module = None

    # Try to import it directly
    try:
        module = __import__(module_name, fromlist=[cls])
    except ImportError:
        pass
        # raise ImportError(
        #    f"Cannot import {module_name} necesary to resolve {name}. You need to install it to load this object."
        # )

    # it failed, maybe this is a path of the form `module.class.method`
    if module is None and len(parts) > 2:
        module_name = ".".join(parts[:-2])
        _cls = parts[-2]
        try:
            module = __import__(module_name, fromlist=[cls])
        except ImportError:
            raise ImportError(
                f"Cannot import {module_name} necesary to resolve {name}. You need to install it to load this object."
            )

        try:
            module = getattr(module, _cls)
        except AttributeError:
            raise AttributeError(
                f"Cannot resolve {name} from module {module}. Maybe it moved since you saved this object?"
            )

    try:
        obj = getattr(module, cls)
    except AttributeError:
        global_path = current_context().path
        raise AttributeError(
            f"Cannot resolve {name} from module {module}. Maybe it moved since you saved this object?\n"
            f"This error happened while processing object at path `{global_path}`."
        )
    return obj
