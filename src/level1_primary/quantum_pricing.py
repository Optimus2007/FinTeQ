from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .circuit_design import get_default_circuit_config


def level1_quantum_pricing_config() -> Dict[str, object]:
    cfg = get_default_circuit_config()
    return {
        "status": "level1 primary module scaffolded for submission structure",
        "circuit": asdict(cfg),
    }
