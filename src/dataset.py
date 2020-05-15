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

    for v in range(1, 340):
        dtype[f'V{v}'] = np.float32

    for d in range(1, 16):
        dtype[f'D{d}'] = np.float16

    for c in range(1, 15):
        dtype[f'C{c}'] = np.float16

    for i in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',
              'id_07', 'id_08', 'id_09', 'id_10', 'id_11']:
        dtype[i] = np.float32

    for c in range(1, 7):
        dtype[f'card{c}'] = np.object

    for i in range(12, 39):
        dtype[f'id_{i}'] = np.object

    for a in range(1, 3):
        dtype[f'addr{a}'] = np.object

    return pd.read_csv('../data/dataset.csv', dtype=dtype)


def split_and_save_processed_data(df, **kwargs):
    """
    Split and store dataframe into train-test dataframes.

    :param df: dataframe to be splitted.
    :param kwargs: additional arguments:
        test_size (float): test data ratio (default 0.2).
    """
    print('Splitting the data...')
    x = df.drop('isFraud', axis=1)
    y = df[['isFraud']]
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
        n_train = int(len(x_train) * frac)
        n_test = int(len(x_test) * frac)
        
        x_train = x_train.sample(n=n_train, replace=False, random_state=420)
        y_train = y_train.sample(n=n_train, replace=False, random_state=420)
        x_test = x_test.sample(n=n_test, replace=False, random_state=420)
        y_test = y_test.sample(n=n_test, replace=False, random_state=420)        

    y_train = y_train['isFraud'].tolist()
    y_test = y_test['isFraud'].tolist()
    
    print(f'Number of records:')
    print(f'  x_train - {len(x_train)}')
    print(f'  y_train - {len(y_train)}')
    print(f'  x_test - {len(x_test)}')
    print(f'  y_test - {len(y_test)}')

    return x_train, y_train, x_test, y_test


def shrink_data(data, n, **kwargs):
    """
    Shrinks the data and returns it.

    :param data: dataframe or list to be shrinked.
    :param n: the new data size.
    :param kwargs: additional arguments:
        random_state (int): random state for sampling from the data
    """
    
    random_state = kwargs.get('random_state', None)
    
    if isinstance(data, list):
        df_data = pd.DataFrame(data).sample(n=n, replace=False, random_state=random_state)
        return df_data[0].tolist()
    
    return data.sample(n=n, replace=False, random_state=random_state)