# ══════════════════════════════════════════════════════
# SEQUENTIAL PHOTONIC QRC WITH HIDDEN MODES
# Photonic implementation of the paper's architecture
# Input modes: 0,1,2 (3 PCA features)
# Hidden modes: 3-9 (7 memory modes)
# ══════════════════════════════════════════════════════

class SequentialPhotonicReservoir:
    """
    Implements sequential QRC from the volatility paper,
    adapted for photonic hardware using MerLin.

    At each timestep t:
    1. Encode input x_t into input modes
    2. Evolve full circuit (input + hidden modes interact)
    3. Read hidden mode measurements as new hidden state
    4. Feed hidden state to next timestep via phase modulation
    """

    def __init__(self, n_modes=10, n_input_features=3,
                 n_photons=5, memory_depth=3, seed=42):
        self.n_modes = n_modes
        self.n_input = n_input_features
        self.n_photons = n_photons
        self.memory_depth = memory_depth
        self.seed = seed
        self.proc = None
        self.LI_params = None
        self.RI_params = None
        self.W_out = None

    def build(self):
        print(f"\nBuilding SequentialPhotonicReservoir...")
        print(f"  Modes: {self.n_modes} ({self.n_input} input + "
              f"{self.n_modes-self.n_input} hidden)")
        print(f"  Memory depth: {self.memory_depth}")

        # FeatureMap: encodes n_input features into full n_modes circuit
        fm = merlin.FeatureMap.simple(
            input_size=self.n_input,
            n_modes=self.n_modes,
        )

        # Fixed random reservoir weights
        param_names = [p.name for p in
                      fm.experiment.unitary_circuit().get_parameters()]
        LI_count = sum(1 for n in param_names if n.startswith('LI'))
        RI_count = sum(1 for n in param_names if n.startswith('RI'))

        torch.manual_seed(self.seed)
        self.LI_params = torch.rand(LI_count) * 2 * np.pi
        self.RI_params = torch.rand(RI_count) * 2 * np.pi

        # ComputationProcess
        input_params = ['input'] if isinstance(
            fm.input_parameters, str) else list(fm.input_parameters)

        self.proc = merlin.ComputationProcess(
            circuit=fm.experiment.unitary_circuit(),
            input_state=get_input_state_v2(self.n_modes, self.n_photons),
            trainable_parameters=list(fm.trainable_parameters),
            input_parameters=input_params,
            n_photons=self.n_photons,
        )

        # Get output size
        test = self.proc.compute([
            self.LI_params, self.RI_params, torch.zeros(self.n_input)
        ]).real
        self.q_out_size = test.shape[1]
        print(f"  Output size: {self.q_out_size}")
        print(f"  Ready.")

    def _encode_sequential(self, x_sequence):
        """
        x_sequence: (memory_depth, n_input) — sequence of past inputs

        Mimics paper's hidden state propagation:
        - Step 1: encode t-k, get hidden state h1
        - Step 2: encode t-k+1 + h1 feedback, get h2
        - Step 3: encode t-1 + h2 feedback, measure all

        Photonic adaptation: hidden state fed back as
        additional phase shift on input encoding
        """
        hidden_state = np.zeros(self.n_input)  # init hidden feedback

        for step in range(self.memory_depth):
            x_t = x_sequence[step]  # (n_input,)

            # Combine input with hidden state feedback
            # This is the photonic analog of ρI ⊗ ρh
            x_combined = np.tanh(x_t + hidden_state)

            # Run through circuit
            out = self.proc.compute([
                self.LI_params,
                self.RI_params,
                torch.FloatTensor(x_combined)
            ]).real[0].detach().numpy()  # (q_out_size,)

            # Update hidden state: use first n_input outputs as feedback
            # Scale to [-pi, pi] range
            hidden_state = np.tanh(out[:self.n_input]) * np.pi

        # Final measurement after all timesteps
        return out  # (q_out_size,)

    def get_measurements(self, X):
        """
        X: (n_samples, memory_depth, n_input)
        Returns: (n_samples, q_out_size)
        """
        M = np.zeros((len(X), self.q_out_size))
        for i, seq in enumerate(X):
            M[i] = self._encode_sequential(seq)
        return M

    def fit(self, X, y, alpha=0.01):
        """X: (n_samples, memory_depth, n_input), y: targets"""
        print(f"\nFitting sequential reservoir...")
        M = self.get_measurements(X)
        print(f"  M: {M.shape}, range [{M.min():.3f}, {M.max():.3f}]")
        var_cols = (M.var(axis=0) > 1e-6).sum()
        print(f"  Non-zero variance cols: {var_cols}/{self.q_out_size}")
        self.W_out = np.linalg.solve(
            M.T @ M + alpha * np.eye(M.shape[1]),
            M.T @ y
        )
        pred = M @ self.W_out
        print(f"  Train R2: {r2(y, pred):.6f}")
        return self

    def predict(self, X):
        return self.get_measurements(X) @ self.W_out

print("✓ SequentialPhotonicReservoir defined")

# ── Build sequences with memory_depth=3 ──
memory = 3

def build_sequences(X_pca, memory):
    """Build (n_samples, memory, n_features) sequences."""
    seqs, targets = [], []
    for i in range(memory, len(X_pca)):
        seqs.append(X_pca[i-memory:i])   # past memory steps
        targets.append(X_pca[i])          # next step
    return np.array(seqs), np.array(targets)

# Scale PCA to [-pi, pi] first
scaler_seq = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_pca_all = np.vstack([X_pca_train, X_pca_val, X_pca_test])
scaler_seq.fit(X_pca_train)

X_pca_train_s = scaler_seq.transform(X_pca_train)
X_pca_val_s   = scaler_seq.transform(X_pca_val)
X_pca_test_s  = scaler_seq.transform(X_pca_test)

# Build sequences
X_seq_train, y_seq_train = build_sequences(X_pca_train_s, memory)
X_seq_val,   y_seq_val   = build_sequences(X_pca_val_s,   memory)
X_seq_test,  y_seq_test  = build_sequences(X_pca_test_s,  memory)

# Targets are PCA components — reconstruct to full surface
y_full_train = pca.inverse_transform(scaler_seq.inverse_transform(y_seq_train))
y_full_val   = pca.inverse_transform(scaler_seq.inverse_transform(y_seq_val))
y_full_test  = pca.inverse_transform(scaler_seq.inverse_transform(y_seq_test))

print(f"Sequences — Train: {X_seq_train.shape} | "
      f"Val: {X_seq_val.shape} | Test: {X_seq_test.shape}")
print(f"Targets   — Train: {y_seq_train.shape}")

# ── Build and train ──
seq_res = SequentialPhotonicReservoir(
    n_modes=10, n_input_features=3,
    n_photons=5, memory_depth=3, seed=42
)
seq_res.build()

# Tune alpha
print("\nTuning alpha...")
best_a, best_r2_v, best_W = None, -999, None

for alpha in [1e-4, 1e-2, 0.1, 1.0, 10.0]:
    seq_res.fit(X_seq_train, y_seq_train, alpha=alpha)
    M_val = seq_res.get_measurements(X_seq_val)
    pred_pca = scaler_seq.inverse_transform(M_val @ seq_res.W_out)
    pred_surf = pca.inverse_transform(pred_pca)
    rv = r2(y_full_val, pred_surf)
    print(f"  alpha={alpha:.0e} → Val R2: {rv:.8f}")
    if rv > best_r2_v:
        best_r2_v, best_a, best_W = rv, alpha, seq_res.W_out.copy()

# Final test
seq_res.W_out = best_W
M_test_seq = seq_res.get_measurements(X_seq_test)
pred_test_pca  = scaler_seq.inverse_transform(M_test_seq @ best_W)
pred_test_surf = pca.inverse_transform(pred_test_pca)

# Naive baseline on same test window
naive_surf = pca.inverse_transform(
    scaler_seq.inverse_transform(X_seq_test[:, -1, :]))

print(f"\n{'='*50}")
print(f"  SEQUENTIAL QRC RESULTS")
print(f"{'='*50}")
print(f"  Naive (last step):          {r2(y_full_test, naive_surf):.8f}")
print(f"  Simple QRC Reservoir:       0.98733525")
print(f"  Sequential QRC (paper):     {r2(y_full_test, pred_test_surf):.8f}")
print(f"  Best alpha: {best_a}")
