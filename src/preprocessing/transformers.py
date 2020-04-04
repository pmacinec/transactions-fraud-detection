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


class MergeSmallCategories(TransformerMixin):
    """
    Merge too small classes in categorical attribute into 'other' class.

    Taken from other author's project:
    https://github.com/pmacinec/diabetes-patients-readmissions-prediction/

    :param threshold: threshold of percentage frequency under which 
        classes will be merged into 'other' class.
    """

    def __init__(self, threshold=0.05, **kwargs):
        self.threshold = threshold
        self.mapping = {}

    def fit(self, df, y=None):
        for col in df.columns:
            if '_missing' in col:
                continue
            values = df[col].value_counts(normalize=True)
            for name, value in values.iteritems():
                if value < self.threshold:
                    self.mapping[name] = 'other'
        return self

    def transform(self, df, **transform_params):
        for col in df.columns:
            if '_missing' in col:
                continue
            df[col] = df[col].apply(
                lambda x: self.get_value(x)
            )
        return df

    def get_value(self, value):
        """
        Get value from mapping with handling special cases.

        :param value: value to be found in mapping.
        :return: mapped value or None, if value is unknown or NaN.
        """
        if pd.isna(value):
            return None
        if value not in self.mapping.keys():
            return value
        return self.mapping[value]


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
            if col not in df.columns:
                continue
            df[col] = df[col].astype('str').apply(transform_email).astype('str')

        return df


class Normalizer(TransformerMixin):
    """Normalize numerical attributes of dataframe."""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, df, y=None):
        for col in df.columns:
            if '_missing' in col:
                continue
            self.means[col] = df[col].astype(float).mean()
            self.stds[col] = df[col].astype(float).std()
        return self

    def transform(self, df):
        for col in df.columns:
            if '_missing' in col:
                continue
            df[col] = (df[col] - self.means[col]) / self.stds[col]
        return df
