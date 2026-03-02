# Methodology

## Level-2 Hybrid Pipeline
- Data source: `Quandela/Challenge_Swaptions` with `level-2_Missing_data_prediction/train_level2.csv`.
- Features: mask-aware engineered features + photonic quantum features.
- Model: weighted blend of naive, classical ridge, and quantum-feature ridge.
- Selection: internal hidden-entry reconstruction objective with overfit-gap penalty.
- Validation: chronological expanding-window repeated CV, leakage-safe masking.

## Quantum Constraints
- Simulation limits enforced.
- QPU-safe settings enforced (`amplitude_encoding=False`, `state_injection=False`).
