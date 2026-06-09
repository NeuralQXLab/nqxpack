"""Per-call serialization options.

Options are *ephemeral* knobs carried on the packing context from a single
``save()`` / ``load()`` call down to the individual (de)serializers that run
underneath it. They are never written to the archive.

Each option is *declared at registration* (:func:`SaveOption` / :func:`LoadOption`
passed to ``register_serialization``), so nqxpack owns a catalogue it can
validate user-supplied values against and document via :func:`list_options`.
"""

import difflib
import warnings

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptionSpec:
    """Declaration of a single (de)serialization option.

    Attributes:
        name: The option name, used as the key in the ``options=`` dict.
        type: The expected type of the value; checked with ``isinstance``.
        default: The value returned by :func:`option` when the user did not set it.
        doc: A one-line human description, surfaced by :func:`list_options`.
        direction: ``"save"`` or ``"load"`` -- which entry point accepts it.
    """

    name: str
    type: type
    default: Any
    doc: str
    direction: str  # "save" | "load"


def SaveOption(name: str, type: type, default: Any, doc: str = "") -> OptionSpec:
    """Declare an option honoured at ``save()`` time."""
    return OptionSpec(name, type, default, doc, "save")


def LoadOption(name: str, type: type, default: Any, doc: str = "") -> OptionSpec:
    """Declare an option honoured at ``load()`` time."""
    return OptionSpec(name, type, default, doc, "load")


# name -> merged spec. Built incrementally at registration time.
OPTION_CATALOGUE: dict[str, OptionSpec] = {}


def register_option(spec: OptionSpec):
    """Add ``spec`` to the catalogue, enforcing the flat-namespace rule.

    Several registrations may declare the same option name as long as the
    ``(type, default, direction)`` triple is identical -- they then describe a
    single logical knob honoured by all of them. An incompatible redeclaration
    is a registration-time error.
    """
    existing = OPTION_CATALOGUE.get(spec.name)
    if existing is None:
        OPTION_CATALOGUE[spec.name] = spec
    elif (existing.type, existing.default, existing.direction) != (
        spec.type,
        spec.default,
        spec.direction,
    ):
        raise ValueError(
            f"Incompatible redeclaration of option {spec.name!r}: "
            f"{existing} vs {spec}."
        )
    # compatible same-name declaration -> one logical knob, nothing else to do.


def list_options(direction: str | None = None) -> list[OptionSpec]:
    """List the options known to nqxpack.

    Args:
        direction: If ``"save"`` or ``"load"``, restrict to that direction.
            If ``None`` (default), return every declared option.
    """
    return [
        s
        for s in OPTION_CATALOGUE.values()
        if direction is None or s.direction == direction
    ]


def _validate_options(options: dict, mode: str) -> dict:
    """Validate user-supplied options against the catalogue for ``save``/``load``.

    Raises ``KeyError`` (unknown key, with a did-you-mean), ``ValueError``
    (wrong direction) or ``TypeError`` (wrong value type).
    """
    for key, value in options.items():
        spec = OPTION_CATALOGUE.get(key)
        if spec is None:
            hint = difflib.get_close_matches(key, OPTION_CATALOGUE, n=1)
            suffix = f" Did you mean {hint[0]!r}?" if hint else ""
            raise KeyError(f"Unknown option {key!r}.{suffix}")
        if spec.direction != mode:
            other = "load" if mode == "save" else "save"
            raise ValueError(
                f"{key!r} is a {other}-time option; it has no effect on {mode}()."
            )
        if not isinstance(value, spec.type):
            raise TypeError(
                f"option {key!r} expects {spec.type.__name__}, "
                f"got {type(value).__name__}."
            )
    return options


def _warn_unused(ctx):
    """Warn about save options that were set but never read this call.

    Only save options warn; load policy flags are expected to often go unused
    (the safe path resolved without consulting them), so the load side is silent.
    """
    if ctx._mode != "save":
        return
    unused = set(ctx._options) - ctx._read_options
    for key in sorted(unused):
        warnings.warn(
            f"option {key!r} was set but no serializer used it this call.",
            UserWarning,
            stacklevel=3,
        )
