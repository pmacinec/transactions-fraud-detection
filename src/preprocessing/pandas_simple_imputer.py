import pandas as pd

from sklearn.impute import SimpleImputer


class PandasSimpleImputer(SimpleImputer):
    """Simple imputer transformer for pandas."""
    
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
            if '_missing' in col:
                continue
            x.loc[:, col] = matrix[:, index]

        return x
