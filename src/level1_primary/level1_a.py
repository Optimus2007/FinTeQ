import numpy as np
import torch
import pandas as pd
import merlin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ── Load data ──
df = pd.read_parquet('/content/019c9f39-e697-7fa1-9725-d93bdd138124.parquet')
price_cols = [c for c in df.columns if c != 'Date']
X = df[price_cols].values  # (494, 224)

# ── Split ──
n_train, n_val = 345, 74
X_train = X[:n_train]
X_val   = X[n_train:n_train+n_val]
X_test  = X[n_train+n_val:]

# ── PCA ──
pca = PCA(n_components=3)
X_pca_train = pca.fit_transform(X_train)
X_pca_val   = pca.transform(X_val)
X_pca_test  = pca.transform(X_test)

# ── Scale ──
scaler_seq = MinMaxScaler(feature_range=(-np.pi, np.pi))
scaler_seq.fit(X_pca_train)
X_pca_train_s = scaler_seq.transform(X_pca_train)
X_pca_val_s   = scaler_seq.transform(X_pca_val)
X_pca_test_s  = scaler_seq.transform(X_pca_test)

# ── R2 function ──
def r2(y_true, y_pred):
    return 1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)

# ── Input state ──
def get_input_state_v2(n_modes=10, n_photons=3):
    state = [0] * n_modes
    positions = list(range(0, n_modes, 2))[:n_photons]
    if len(positions) < n_photons:
        positions += list(range(1, n_modes, 2))[:n_photons-len(positions)]
    for pos in positions:
        state[pos] = 1
    return state

# ── Sequences ──
def build_sequences(X_pca, memory):
    seqs, targets = [], []
    for i in range(memory, len(X_pca)):
        seqs.append(X_pca[i-memory:i])
        targets.append(X_pca[i])
    return np.array(seqs), np.array(targets)

memory = 3
X_seq_train, y_seq_train = build_sequences(X_pca_train_s, memory)
X_seq_val,   y_seq_val   = build_sequences(X_pca_val_s,   memory)
X_seq_test,  y_seq_test  = build_sequences(X_pca_test_s,  memory)

y_full_train = pca.inverse_transform(scaler_seq.inverse_transform(y_seq_train))
y_full_val   = pca.inverse_transform(scaler_seq.inverse_transform(y_seq_val))
y_full_test  = pca.inverse_transform(scaler_seq.inverse_transform(y_seq_test))

print("✓ Data ready")
print(f"Train: {X_seq_train.shape} | Val: {X_seq_val.shape} | Test: {X_seq_test.shape}")
print(f"Naive R2: {r2(y_full_test, pca.inverse_transform(scaler_seq.inverse_transform(X_seq_test[:,-1,:]))):.6f}")
