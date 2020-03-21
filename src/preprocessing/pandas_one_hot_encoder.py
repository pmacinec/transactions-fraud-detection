import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        return pd.get_dummies(x, dummy_na=True, drop_first=True)
