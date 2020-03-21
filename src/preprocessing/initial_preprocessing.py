from os.path import join, dirname
import pandas as pd


def merge_datasets():
    """Merge `transactions.csv` and `identities.csv` files."""
    print('Datasets merging started.')

    dir_name = dirname(__file__)

    df_identities = pd.read_csv(join(dir_name, '../../data/identities.csv'))
    print('File identities.csv read.')

    df_transactions = pd.read_csv(
        join(dir_name, '../../data/transactions.csv')
    )
    print('File transactions.csv read.')

    df = pd.merge(
        df_transactions,
        df_identities,
        on='TransactionID',
        how='left'
    )

    print('Saving merged dataset...')
    df.to_csv(join(dir_name, '../../data/dataset.csv'), index=False)
    print('Dataset successfully stored as dataset.csv.')


if __name__ == '__main__':
    merge_datasets()
