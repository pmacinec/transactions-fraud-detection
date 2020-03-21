import pandas as pd

from sklearn.impute import SimpleImputer


class PandasSimpleImputer(SimpleImputer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y=None):
        return super().fit(x, y)

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        matrix = super().transform(x)

        for index, col in enumerate(x.columns):
            x.loc[:, col] = matrix[:, index]

        return x
