import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class SelectFeatures(BaseEstimator, TransformerMixin):
    """
    Select subset of features from dataframe.
    
    :param columns: columns to choose from dataframe.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        assert isinstance(df, pd.DataFrame)

        return df[self.columns]


class FilterColumnsByCountOfMissingValues(BaseEstimator, TransformerMixin):
    """
    Filter columns by count of missing values.
    
    :param threshold: threshold ratio of missing values in column.
    """

    def __init__(self, threshold, **kwargs):
        self.threshold = threshold

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        assert isinstance(df, pd.DataFrame)

        to_drop = df.columns[1 - df.isnull().mean() < self.threshold].tolist()

        return df.drop(columns=to_drop)


class KeepOnlyMostCommonValues(BaseEstimator, TransformerMixin):
    def __init__(self, n, **kwargs):
        self.n = n
        self.columns = kwargs.get('columns', None)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        columns = self.columns if self.columns is not None else x.columns

        for column in columns:
            most_common = x[column].value_counts().nlargest(n=self.n).index.tolist()
            x.loc[~x[column].isin(most_common), column] = 'Other'

        return x[columns]


class EmailProviderTransform(TransformerMixin, BaseEstimator):
    """
    Transform email addresses into domains only.

    :param columns: columns containing email addresses.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None, copy=None):
        def transform_email(value):
            """
            Transform email domains into domain names only.

            We have found in the data, that some email domains are
            repeated (e.g. `gmail` and `gmail.com`). Those domains
            should be aggregated into simply `gmail`.

            :param value: value to be transformed.
            :return: transformed email domain.
            """
            if value is np.nan:
                return value
            return value.split('.')[0]

        for col in self.columns:
            df[col] = df[col].astype('str').apply(transform_email).astype('str')

        return df


class Normalizer(TransformerMixin):
    """Normalize numerical attributes of dataframe."""

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = (df - df.mean()) / df.std()
        return df
