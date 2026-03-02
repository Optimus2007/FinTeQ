from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CircuitConfig:
    n_modes: int = 8
    n_photons: int = 3
    amplitude_encoding: bool = False
    state_injection: bool = False


def get_default_circuit_config() -> CircuitConfig:
    return CircuitConfig()
