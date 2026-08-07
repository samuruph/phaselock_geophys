"""Frozen external encoders -- the published GeoPhys path, kept as a reference baseline.

The project's question is whether GeoPhys geometry transfers to a generator's own
internals. Answering it requires knowing what the geometry scores on the representations
it was designed for, both to validate the implementation against published numbers and to
give the internal results something to be compared against.
"""

from typing import Any

from .dinov2 import DEFAULT_MODEL, DINOv2Encoder, select_layer

ENCODERS: dict[str, type] = {"dinov2": DINOv2Encoder}


def get_encoder(name: str, **kwargs: Any):
    """Instantiate a registered frozen encoder by name."""
    try:
        cls = ENCODERS[name]
    except KeyError:
        raise KeyError(
            f"unknown encoder {name!r}; registered encoders are {sorted(ENCODERS)}"
        ) from None
    return cls(**kwargs)


__all__ = ["DEFAULT_MODEL", "DINOv2Encoder", "ENCODERS", "get_encoder", "select_layer"]
