# ══ EDGE OF CHAOS SWEEP ══
chaos_scales = [0.1, 0.5, 1.0, np.pi, 5.0, 8.0]
chaos_results = {}

print("Edge of Chaos Sweep")
print("="*55)
print(f"  scale  | Val R2    | Test R2   | regime")
print(f"  -------|-----------|-----------|--------")

for cs in chaos_scales:
    res1 = SequentialPhotonicReservoir(
        n_modes=10, n_input_features=3,
        n_photons=5, memory_depth=3,
        seed=42, chaos_scale=cs)
    res1.build()
    res2 = SequentialPhotonicReservoir(
        n_modes=10, n_input_features=3,
        n_photons=5, memory_depth=3,
        seed=123, chaos_scale=cs)
    res2.build()

    M_tr = np.concatenate([
        res1.get_measurements(X_seq_train),
        res2.get_measurements(X_seq_train)], axis=1)
    M_vl = np.concatenate([
        res1.get_measurements(X_seq_val),
        res2.get_measurements(X_seq_val)], axis=1)
    M_te = np.concatenate([
        res1.get_measurements(X_seq_test),
        res2.get_measurements(X_seq_test)], axis=1)

    best_r2_v, best_W = -999, None
    for alpha in [1e-4, 1e-2, 0.1, 1.0]:
        W = np.linalg.solve(
            M_tr.T @ M_tr + alpha*np.eye(M_tr.shape[1]),
            M_tr.T @ y_seq_train)
        pred = pca.inverse_transform(
            scaler_seq.inverse_transform(M_vl @ W))
        rv = r2(y_full_val, pred)
        if rv > best_r2_v:
            best_r2_v, best_W = rv, W.copy()

    pred_te = pca.inverse_transform(
        scaler_seq.inverse_transform(M_te @ best_W))
    test_r2 = r2(y_full_test, pred_te)

    if cs < 0.5:
        regime = "stable"
    elif cs < 2.0:
        regime = "edge <--"
    elif cs < 5.0:
        regime = "chaotic"
    else:
        regime = "unstable"

    chaos_results[cs] = {
        'val_r2': best_r2_v,
        'test_r2': test_r2,
        'W': best_W,
        'res1': res1, 'res2': res2
    }
    print(f"  {cs:<6.3f} | {best_r2_v:.6f} | {test_r2:.6f} | {regime}")

best_cs = max(chaos_results, key=lambda k: chaos_results[k]['val_r2'])
print(f"\n  Best chaos_scale:  {best_cs}")
print(f"  Best val R2:       {chaos_results[best_cs]['val_r2']:.6f}")
print(f"  Best test R2:      {chaos_results[best_cs]['test_r2']:.6f}")
print(f"  Previous best:     0.996700 (fixed scale=pi)")
print(f"  Improvement:       {chaos_results[best_cs]['test_r2'] - 0.996700:+.6f}")
