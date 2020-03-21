import numpy as np
import pandas as pd


def load_data():
    dtype = {}

    for col in ['TransactionID', 'TransactionDT']:
        dtype[col] = np.uint32

    for col in ['dist1', 'dist2']:
        dtype[col] = np.float16

    for col in ['TransactionAmt']:
        dtype[col] = np.float16

    for col in ['isFraud']:
        dtype[col] = np.bool

    for v in range(1, 339):
        dtype[f'V{v}'] = np.float32

    for d in range(1, 15):
        dtype[f'D{d}'] = np.float16

    for c in range(1, 14):
        dtype[f'C{c}'] = np.float16

    for i in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11']:
        dtype[f'id_{i}'] = np.float32

    for c in range(1, 6):
        dtype[f'card{c}'] = np.object

    for i in range(12, 38):
        dtype[f'id_{i}'] = np.object

    for a in range(1, 2):
        dtype[f'addr{a}'] = np.object

    return pd.read_csv('../data/dataset.csv', dtype=dtype)
