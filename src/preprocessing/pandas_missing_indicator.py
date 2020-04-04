import pandas as pd
import numpy as np

from sklearn.impute import MissingIndicator


class PandasMissingIndicator(MissingIndicator):
    """Missing indicator transformer for pandas."""

    def __init__(self, **kwargs):
        super().__init__(features='all', **kwargs)

        self._suffix = kwargs.get('suffix', '_missing')

    def fit(self, x, y=None):
        super().fit(x, y)
        return self

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        matrix = super().transform(x)

        for index, col in enumerate(x.columns):
            row = matrix[:, index]
            col_name = col + self._suffix

            if np.any(row == True):
                x.loc[:, col_name] = row

                x[col_name] = x[col_name].astype(int)

        return x
