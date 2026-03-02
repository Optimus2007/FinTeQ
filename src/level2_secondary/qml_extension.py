from __future__ import annotations

from pathlib import Path
from typing import Dict

from .hybrid_model import fill_template


def run_level2_qml_extension(
    template_path: str,
    output_xlsx: str,
    output_report: str,
    min_quantum_weight: float = 0.50,
    cv_repeats: int = 3,
) -> Dict[str, object]:
    return fill_template(
        template_path=Path(template_path),
        out_xlsx=Path(output_xlsx),
        out_report=Path(output_report),
        cv_repeats=int(cv_repeats),
        min_w_q=float(min_quantum_weight),
    )
