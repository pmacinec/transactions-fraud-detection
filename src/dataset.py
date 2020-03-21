import numpy as np
import pandas as pd


def load_data():
    dtype = {}

    for col in ['TransactionID', 'TransactionDT']:
        dtype[col] = np.uint32

    for col in ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2']:
        dtype[col] = np.float16

    for col in ['TransactionAmt']:
        dtype[col] = np.float16;

    for col in ['isFraud']:
        dtype[col] = np.bool

    for v in range(1, 339):
        dtype[f'V{v}'] = np.float32;

    for d in range(1, 15):
        dtype[f'D{d}'] = np.float16;

    for c in range(1, 14):
        dtype[f'C{c}'] = np.float16;

    for id in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11',
               'id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24', 'id_25', 'id_26',
               'id_32']:
        dtype[f'id_{id}'] = np.float32;

    return pd.read_csv('../data/dataset.csv', dtype=dtype)
