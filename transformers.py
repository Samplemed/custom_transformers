"""
Custom scikit-learn transformers.
"""

from sklearn.base import BaseEstimator, TransformerMixin


# alternative:
# https://scikit-learn.org/stable/modules/generated/
# sklearn.compose.make_column_selector.html
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for selecting columns in specified order
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """
        fit
        """
        return self

    def transform(self, X, y=None):
        """
        transform
        """
        return X[self.columns]
