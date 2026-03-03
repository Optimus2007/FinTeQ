# FinteQ — Photonic Quantum Kernel for Swaption IVS Forecasting

> Quandela Swaptions QML Hackathon 2026 · Perceval · MerLin · QPU-Compatible

## What We Did

Implemented a fidelity-based photonic quantum kernel ([Yin et al. 2024](https://arxiv.org/abs/2407.20364)) on Perceval's SLOS backend for swaption implied volatility surface forecasting (494 training days → 6 predictions, 224-dimensional surface).

Core contribution: **first indistinguishability sweep on financial time series** — varying photon indistinguishability 0.0→1.0 to measure Hong-Ou-Mandel interference effects on prediction. Result: strictly monotonic degradation. Quantum interference measurably alters kernel geometry (diff norm=4.97) but hurts on linear data.

## Results

| Model | R² |
|---|---|
| Rolling naive (true ceiling) | 0.9981 |
| Classical kernel (indist=0.0) | 0.9981 |
| **Quantum kernel (indist=1.0) ★** | **0.9960** |
| QFinger Hybrid L2 (CV) | 0.9997 |

## Why Quantum Doesn't Win Here

The surface lives on a near-linear 3D manifold (PCA(3)=99.96%, autocorr R=0.9999). Complex permanents obscure linear geometry rather than enriching it. 19 ablation experiments confirm — memory saturates at depth 1, chaos scale flat, residuals white noise.

**Pre-screening criterion:** if PCA(3) > 99% and residual autocorr < 0.05, quantum advantage is unlikely.

## Install

```bash
pip install perceval-quandela merlinquantum numpy pandas scikit-learn openpyxl pyarrow
```

## Run

```bash
# Indistinguishability sweep (core experiment, ~30-40 min)
python level1_kernel/indist_sweep.py

# Generate submission predictions
python level1_kernel/predict.py
```

## Structure

```
├── .gitignore
├── Final_Report.pdf
├── README.md
├── requirements.txt
├── docs/
│   └── references.md
├── graphs/
│   ├── experiment_findings_table.png
│   └── quantum_interference_linear_manifolds.png
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_level2_step_by_step.ipynb
├── predictions/
│   ├── level_1_prediction.xlsx
│   └── level_2_prediction.xlsx
└── src/
    ├── level1_primary/
    │   ├── level1_01_quantum_kernel_submission.py
    │   └── level1_02_baseline_validation.py
    └── level2_secondary/
        ├── hybrid_model.py
        └── qml_extension.py
```

## References

All references are listed in [docs/references.md](docs/references.md).

## Presentation Video

Watch our project presentation here: [Google Drive Video](https://drive.google.com/file/d/1xm2zRhzJcrVVupLNVXhDImOvF0_OzxR4/view?usp=sharing).

---
*Team FinteQ · Quandela 2026 · 19 experiments*
