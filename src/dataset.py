import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def load_data():
    """
    Load data with custom types.

    :return: loaded dataframe.
    """
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

    for i in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',
              'id_07', 'id_08', 'id_09', 'id_10', 'id_11']:
        dtype[f'id_{i}'] = np.float32

    for c in range(1, 6):
        dtype[f'card{c}'] = np.object

    for i in range(12, 38):
        dtype[f'id_{i}'] = np.object

    for a in range(1, 2):
        dtype[f'addr{a}'] = np.object

    return pd.read_csv('../data/dataset.csv', dtype=dtype)


def split_and_save_processed_data(new_df_transactions, **kwargs):
    """
    Split and store dataframe into train-test dataframes.

    :param new_df_transactions: dataframe to be splitted.
    :param kwargs: additional arguments:
        test_size (float): test data ratio (default 0.2).
    """
    print('Splitting the data...')
    x = new_df_transactions.drop('isFraud', axis=1)
    y = new_df_transactions[['isFraud']]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=kwargs.get('test_size', 0.2), random_state=42
    )

    print('Saving data...')
    x_train.to_csv('../data/x_train.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    x_test.to_csv('../data/x_test.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)


def load_processed_data(**kwargs):
    """
    Load preprocessed train-test splitted data.

    :param frac: fraction of axis items to return.

    :return: x_train, y_train, x_test, y_test data.
    """
    x_train = pd.read_csv('../data/x_train.csv')
    y_train = pd.read_csv('../data/y_train.csv')
    x_test = pd.read_csv('../data/x_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    
    frac = kwargs.get('frac', 1)
    
    if frac < 1:
        x_train.sample(frac=frac, replace=True, random_state=420)
        y_train.sample(frac=frac, replace=True, random_state=420)
        x_test.sample(frac=frac, replace=True, random_state=420)
        y_test.sample(frac=frac, replace=True, random_state=420)        

    y_train = y_train['isFraud'].tolist()
    y_test = y_test['isFraud'].tolist()

    return x_train, y_train, x_test, y_test
