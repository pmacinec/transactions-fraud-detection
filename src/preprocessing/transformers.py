import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize


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


class EmailProviderTransform(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=None):
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
            x[col] = x[col].astype('str').apply(transform_email).astype('str')

        return x


class Normalizer(TransformerMixin):
    """Normalize numerical attributes of dataframe."""

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = (df - df.mean()) / df.std()
        return df


class OutliersFilter(TransformerMixin):
    """Remove outliers using quartiles."""

    def __init__(self):
        self.Q1 = {}
        self.Q3 = {}

    def fit(self, df, y=None, **fit_params):
        for col in df.columns:
            self.Q1[col] = df[col].quantile(.25)
            self.Q3[col] = df[col].quantile(.75)
        return self

    def transform(self, df, **transform_params):
        for col in df.columns:
            Q1 = self.Q1[col]
            Q3 = self.Q3[col]
            lower_outlier = Q1 - (Q3 - Q1) * 1.5
            upper_outlier = Q3 + (Q3 - Q1) * 1.5

            df = df[(df[col] >= lower_outlier) & (df[col] <= upper_outlier)]
        return df
