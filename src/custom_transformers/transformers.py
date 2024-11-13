"""
Custom scikit-learn transformers.
"""

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


# alternative:
# https://scikit-learn.org/stable/modules/generated/
# sklearn.compose.make_column_selector.html
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for selecting columns in specified order
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: DataFrame, y=None):
        """
        fit
        """
        return self

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """
        transform
        """
        return X[self.columns]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = {
        "age": [25, 30, 35, 40, 45],
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
            (
                "feature_selector",
                FeatureSelector(columns=["age", "weight"]),
            ),
            ("scaler", StandardScaler()),  # Scale the selected features
            ("classifier", RandomForestClassifier()),  # Classifier
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Predicted: {y_pred[0]}, Actual: {y_test.values[0]}")
