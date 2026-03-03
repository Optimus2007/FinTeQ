import pandas as pd
import numpy as np


df_train = pd.read_parquet('/019c9f39-e697-7fa1-9725-d93bdd138124.parquet')
df_test = pd.read_excel('/test.xlsx')

price_cols = [c for c in df_train.columns if c != 'Date']
price_cols_test = [c for c in df_test.columns if c != 'Date']
X_train = df_train[price_cols].values
y_true = df_test[price_cols_test].values


def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


naive_v1 = np.tile(df_train[price_cols_test].values[-1], (6, 1))
print(f'Naive v1 (last train row): {r2(y_true, naive_v1):.8f}')

all_rows = np.vstack([df_train[price_cols_test].values, y_true])
naive_v2 = all_rows[493:499]
print(f'Naive v2 (rolling):        {r2(y_true, naive_v2):.8f}')

print()
print('=== FINAL HONEST SCOREBOARD ===')
print(f'Naive v1 (train-only ceiling):     {r2(y_true, naive_v1):.8f}')
print(f'Naive v2 (rolling, true ceiling):  {r2(y_true, naive_v2):.8f}')
print(f'QRC Sequential (train-only):       0.89847522  vs naive_v1: {0.89847522 - r2(y_true, naive_v1):+.5f}')
print(f'Quantum Kernel (uses test context): 0.99601068  vs naive_v2: {0.99601068 - r2(y_true, naive_v2):+.5f}')
print(f'Classical Kernel (uses test ctx):  0.99807561  vs naive_v2: {0.99807561 - r2(y_true, naive_v2):+.5f}')