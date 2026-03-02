# Sequential Photonic QRC for Swaption Surface Prediction

**Team:** FinteQ  
**Challenge:** Quandela Swaptions Surface Prediction  
**Level 1 R²:** 0.9967 | **Level 2 R²:** 0.9997

---

## Overview

Implementation of Sequential Photonic Quantum Reservoir Computing (QRC) for predicting interest rate swaption surfaces using MerLin's boson sampling simulation (SLOS).

**Method inspired by:** *Establishing Baselines for Photonic Quantum Machine Learning* (arXiv:2510.25839)

### Results
- **Level 1 (Future Prediction):** R² = 0.9967 using Sequential QRC QR2 ensemble
- **Level 2 (Missing Data):** R² = 0.9997 using QFinger quantum-classical hybrid
- Classical baseline (naive persistence) achieves R² = 0.9988 on Level 1

---

## Dataset

Swaption volatility surface with 494 time steps × 224 dimensions (tenor/maturity grid). High autocorrelation structure makes this a challenging but realistic financial forecasting task.

---

## Methodology

### Level 1: Sequential Photonic QRC

**Pipeline:**
```
224D Input → PCA(3) → Scale[-π,π] → Sequential Reservoir → Ridge → Predict
```

**Architecture:**
- 10-mode photonic circuit, 5 photons
- Hidden state feedback with memory depth of 3 steps
- QR2 ensemble: two reservoirs (different seeds) → 504D measurement space
- Chaos parameter π (tested 0.1→8.0, performance flat)

**Implementation:** MerLin SLOS (software photonic simulation)

### Level 2: QFinger Hybrid

Quantum-classical weighted ensemble (90% quantum / 10% classical) for missing data imputation.

---

## Results

### Level 1: 6-Step Future Prediction

| Model | Test R² |
|-------|---------|
| Naive Persistence | 0.9988 |
| Classical Ridge | 0.9988 |
| Linear Reservoir | 0.9970 |
| Sequential QRC QR2 | 0.9967 |
| Sequential QRC QR1 | 0.9959 |
| Simple QRC | 0.9873 |

*Note: Classical baselines achieve near-ceiling performance due to high autocorrelation in the data.*

### Level 2: Missing Data Imputation

QFinger Hybrid: R² = 0.9997

---

## Experiments

**Systematic ablation studies (19 total):**
- Memory depth (1-5), photon count (3-7), PCA components (2-10)
- Edge of chaos sweep (chaos_scale 0.1→8.0) → performance flat across all values
- Reservoir ensemble size (QR1 vs QR2)
- Residual prediction vs direct prediction

**Key finding:** Dataset geometry limits performance regardless of reservoir configuration. The near-random-walk structure leaves minimal exploitable nonlinear dynamics.

---

## Installation

```bash
pip install merlinquantum torch numpy pandas scikit-learn openpyxl
```

---

## Usage

Run the notebook `level1_sequential_qrc_QR2.ipynb`:
1. Cell 1: Load data & PCA preprocessing
2. Cell 2: Define SequentialPhotonicReservoir class  
3. Cell 3: Train QR2 ensemble & evaluate
4. Cell 4: Retrain on full 494 rows & generate predictions

Output: `submission_final.xlsx` with 6 future predictions

---

## Technical Details

**Sequential reservoir with hidden state feedback:**

```python
def _encode_sequential(self, x_sequence):
    hidden_state = np.zeros(3)
    for step in range(3):  # memory depth
        x_combined = np.tanh(x_sequence[step] + hidden_state)
        out = self.photonic_circuit.compute(x_combined)
        hidden_state = np.tanh(out[:3]) * np.pi  # phase feedback
    return out  # 252D measurement
```

**Pipeline:** 224D → PCA(3) → scale[-π,π] → sequences → QR2(504D) → ridge → inverse transform

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── docs/
│   ├── FinteQ_Final_Report.pdf
│   └── references.md
├── inputs/
│   └── PLACE_LEVEL2_TEMPLATE_HERE.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_level1_step_by_step.ipynb
│   └── 03_level2_step_by_step.ipynb
├── results/
│   ├── metrics.txt
│   ├── level2_hard_cv_q80_report.json
│   └── test_template_filled_hybrid_q80.xlsx
└── src/
    ├── level1_primary/
    │   ├── level1_a.py
    │   ├── level1_b.py
    │   ├── level1_c.py
    │   └── level1_d.py
    └── level2_secondary/
        ├── hybrid_model.py
        └── qml_extension.py
```

---

## References

All references are listed in [docs/references.md](docs/references.md).

---

**Team FinteQ** | Quandela Challenge 2026
