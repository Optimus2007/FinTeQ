# quantum-option-pricing-qml

Hackathon-ready repository with structured Level-1 and Level-2 QML modules.

## Final Level-2 Model Location
- `src/level2_secondary/hybrid_model.py` (full final model code)
- `src/level2_secondary/qml_extension.py` (clean callable wrapper)

## Quick Run (Level-2)

1. Create env + install:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. Run on judge template:
```bash
PYTHONPATH=src .venv/bin/python -c "from level2_secondary.qml_extension import run_level2_qml_extension; run_level2_qml_extension('inputs/test_template.xlsx','results/test_template_filled_hybrid.xlsx','results/level2_validation_report.json',min_quantum_weight=0.50,cv_repeats=3)"
```

## Folder Tree
- `notebooks/` — exploration, classical baseline, quantum model notebooks
- `src/level1_primary/` — level-1 primary module files
- `src/level2_secondary/` — final level-2 hybrid model + extension
- `results/` — metrics and generated outputs
- `docs/` — methodology and references
