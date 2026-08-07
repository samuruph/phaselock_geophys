"""The internal signals that can be recorded, and what each one is.

Results are reported separately per source, never pooled across them: the whole question
is *which* internal representation carries the geometry, and averaging over sources would
destroy exactly that comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

HIDDEN_STATES = "hidden_states"
LATENT = "latent"
X0_HAT = "x0_hat"
VELOCITY = "velocity"
ATTENTION = "attention"


@dataclass(frozen=True)
class Source:
    """Description of one recordable internal signal."""

    name: str
    per_block: bool
    ode_state: bool
    """True if the signal is an affine function of the ODE state.

    Determines whether :mod:`phaselock.metrics.flow_geometry` can compute geometric
    drift exactly or has to fall back to finite differences across recorded steps.
    """

    description: str


SOURCES: dict[str, Source] = {
    HIDDEN_STATES: Source(
        name=HIDDEN_STATES,
        per_block=True,
        ode_state=False,
        description=(
            "Output of each DiT block, spatially mean-pooled within each latent frame. "
            "The signal 'The Invisible Hand of Physics' found most linearly decodable, "
            "peaking in the middle third of the network."
        ),
    ),
    LATENT: Source(
        name=LATENT,
        per_block=False,
        ode_state=True,
        description=(
            "The VAE latent x_t itself. The paper's negative control: linear probes on "
            "it sit at chance (48-53%). It is also the space PhaseLock's latent delta "
            "operates in, which is why the control matters here."
        ),
    ),
    X0_HAT: Source(
        name=X0_HAT,
        per_block=False,
        ode_state=True,
        description=(
            "The denoiser's running estimate of the clean latent. Free at every step, "
            "and closer to the eventual video than x_t at high noise levels."
        ),
    ),
    VELOCITY: Source(
        name=VELOCITY,
        per_block=False,
        ode_state=True,
        description=(
            "The probability-flow drift dz/dtau. Also the input to the flow-coupling "
            "metrics, where it plays the role of u in <grad phi, u>."
        ),
    ),
    ATTENTION: Source(
        name=ATTENTION,
        per_block=True,
        ode_state=False,
        description=(
            "Output of each block's self-attention sublayer, pooled per latent frame. "
            "Note this is the attention *output*, not the attention matrix: at native "
            "resolution a single block's video self-attention is 32760x32760, about 1e9 "
            "entries, and both backends route through fused SDPA which never "
            "materialises it. Capturing true attention maps would need a custom "
            "processor and a resolution reduction, which is deferred."
        ),
    ),
}

PER_BLOCK_SOURCES = frozenset(name for name, source in SOURCES.items() if source.per_block)
ODE_STATE_SOURCES = frozenset(name for name, source in SOURCES.items() if source.ode_state)


def validate_sources(sources: Sequence[str]) -> list[str]:
    """Check requested sources against the registry, preserving order."""
    if not sources:
        raise ValueError("at least one source must be requested")
    unknown = [name for name in sources if name not in SOURCES]
    if unknown:
        raise ValueError(f"unknown probe sources {unknown}; available: {sorted(SOURCES)}")
    seen: list[str] = []
    for name in sources:
        if name not in seen:
            seen.append(name)
    return seen


def supports_exact_drift(source: str) -> bool:
    """Whether geometric drift can be computed analytically for this source.

    False for hidden states and attention outputs, which are nonlinear functions of the
    ODE state and have no analytic ``d/dtau``.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown probe source {source!r}; available: {sorted(SOURCES)}")
    return SOURCES[source].ode_state
