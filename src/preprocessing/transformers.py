import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        return x[self.columns]


class FilterColumnsByCountOfMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        columns_to_drop = x.columns[1 - x.isnull().mean() < self.threshold].tolist()

        return x.drop(columns=columns_to_drop)


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
