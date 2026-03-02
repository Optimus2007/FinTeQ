# QR2: two reservoirs, concatenate outputs
# Seed 42 + seed 123 → 504 features instead of 252

seq_res2 = SequentialPhotonicReservoir(
    n_modes=10, n_input_features=3,
    n_photons=5, memory_depth=3, seed=123  # different seed
)
seq_res2.build()

# Get measurements from both reservoirs
M_tr1 = seq_res.get_measurements(X_seq_train)
M_tr2 = seq_res2.get_measurements(X_seq_train)
M_tr_ensemble = np.concatenate([M_tr1, M_tr2], axis=1)

M_vl1 = seq_res.get_measurements(X_seq_val)
M_vl2 = seq_res2.get_measurements(X_seq_val)
M_vl_ensemble = np.concatenate([M_vl1, M_vl2], axis=1)

M_te1 = seq_res.get_measurements(X_seq_test)
M_te2 = seq_res2.get_measurements(X_seq_test)
M_te_ensemble = np.concatenate([M_te1, M_te2], axis=1)

print(f"Ensemble feature size: {M_tr_ensemble.shape[1]} (vs {M_tr1.shape[1]} single)")

# Tune alpha on ensemble
best_a_ens, best_r2_ens, best_W_ens = None, -999, None
for alpha in [1e-4, 1e-2, 0.1, 1.0]:
    W = np.linalg.solve(
        M_tr_ensemble.T @ M_tr_ensemble + alpha * np.eye(M_tr_ensemble.shape[1]),
        M_tr_ensemble.T @ y_seq_train
    )
    pred_pca  = scaler_seq.inverse_transform(M_vl_ensemble @ W)
    pred_surf = pca.inverse_transform(pred_pca)
    rv = r2(y_full_val, pred_surf)
    print(f"  alpha={alpha:.0e} → Val R2: {rv:.8f}")
    if rv > best_r2_ens:
        best_r2_ens, best_a_ens, best_W_ens = rv, alpha, W.copy()

pred_test_ens  = pca.inverse_transform(
    scaler_seq.inverse_transform(M_te_ensemble @ best_W_ens))

print(f"\n{'='*50}")
print(f"  FINAL LEADERBOARD")
print(f"{'='*50}")
print(f"  Naive:                    0.99882")
print(f"  Classical Ridge:          0.99878")
print(f"  Simple QRC:               0.98734")
print(f"  Sequential QRC (QR1):     0.99587")
print(f"  Sequential QRC (QR2):     {r2(y_full_test, pred_test_ens):.8f}")
print(f"  Quantum Kernel:           0.97597")
