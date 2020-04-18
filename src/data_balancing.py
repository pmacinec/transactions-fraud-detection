from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def random_undersample(x, y):
    """
    Undersample data by random choosing samples from majority class.

    :param x: train data features.
    :param y: train data labels.
    :return: x, y after undersampling.
    """
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(x, y)


def smote_oversample(x, y):
    """
    Oversample data using SMOTE method.

    :param x: train data features.
    :param y: train data labels.
    :return: x, y after oversampling.
    """
    smote = SMOTE(random_state=42)
    return smote.fit_resample(x, y)
