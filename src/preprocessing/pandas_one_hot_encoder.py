import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df):
        assert isinstance(df, pd.DataFrame)

        columns = list(filter(lambda x: '_missing' not in x, df.columns))
        return pd.get_dummies(df, columns=columns)
