"""
Custom scikit-learn transformers.
"""

import pandas as pd

# from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

# NOTE: about the warning
# If you receive the following warning:
#
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
#
# The pandas package will send you to the following docs, that
# states that starting in pandas version 3.0, the copy-on-write will
# be default, so there's no need to worry about this except in extreme
# performance scenarios.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.copy_on_write = True


# alternative:
# https://scikit-learn.org/stable/modules/generated/
# sklearn.compose.make_column_selector.html
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for selecting columns in specified order
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """
        fit
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        transform
        """
        return X[self.columns]


class YearExtractor(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for selecting columns in specified order
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """
        fit
        """
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        """
        transform
        """
        # yeah the ideal is to optimally copy the df
        for col in self.columns:
            X[f"{col}_year"] = pd.to_datetime(X[col])
            X[f"{col}_year"] = X[f"{col}_year"].dt.year

        return X


if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = {
        "birthday": [
            "1981-06-24",
            "1981-06-24",
            "1981-06-24",
            "1981-06-24",
            "1981-06-24",
        ],
        # "age": [25, 30, 35, 40, 45],
        "height": [5.5, 5.8, 5.7, 6.0, 5.9],
        "weight": [150, 160, 170, 180, 190],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "purchased": [0, 1, 0, 1, 1],
    }

    df = pd.DataFrame(data)

    X = df.drop("purchased", axis=1)
    y = df["purchased"]

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_test = X.iloc[-1:]
    y_test = y.iloc[-1:]

    pipeline = Pipeline(
        [
            ("year_extractor", YearExtractor(columns=["birthday"])),
            (
                "feature_selector",
                FeatureSelector(columns=["salary", "weight", "birthday_year"]),
            ),
            ("scaler", StandardScaler()),
            # ("classifier", RandomForestClassifier()),  # Classifier
        ]
    )
    print(pipeline.fit_transform(X_train))

    # pipeline.fit(X_train, y_train)
    #
    # y_pred = pipeline.predict(X_test)
    # print(f"Predicted: {y_pred[0]}, Actual: {y_test.values[0]}")
