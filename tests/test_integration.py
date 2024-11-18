import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from custom_transformers.transformers import AgeExtractor, FeatureSelector, YMDExtractor


@pytest.fixture
def sample_data():
    data = {
        "birthday": [
            "1981-06-24",
            "1981-06-24",
            "1981-06-24",
            "1981-06-24",
            "1981-06-24",
        ],
        "height": [5.5, 5.8, 5.7, 6.0, 5.9],
        "weight": [150, 160, 170, 180, 190],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "purchased": [0, 1, 0, 1, 1],
    }
    df = pd.DataFrame(data)
    X = df.drop("purchased", axis=1)
    y = df["purchased"]
    return X, y


def test_pipeline_with_model(sample_data):
    X, y = sample_data

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_test = X.iloc[-1:]
    y_test = y.iloc[-1:]

    pipeline = Pipeline(
        [
            (
                "year_extractor",
                YMDExtractor(
                    columns=["birthday"],
                    drop_original_cols=False,
                ),
            ),
            (
                "age_extractor",
                AgeExtractor(
                    birthdate_column="birthday",
                ),
            ),
            (
                "feature_selector",
                FeatureSelector(columns=["salary", "weight", "birthday_year", "age"]),
            ),
            ("classifier", RandomForestClassifier()),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"Predicted: {y_pred[0]}, Actual: {y_test.values[0]}")

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1,)
    assert y_pred[0] in [0, 1]


def test_pipeline_without_model(sample_data):
    X, y = sample_data

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_test = X.iloc[-1:]

    pipeline = Pipeline(
        [
            (
                "year_extractor",
                YMDExtractor(
                    columns=["birthday"],
                    drop_original_cols=False,
                ),
            ),
            (
                "age_extractor",
                AgeExtractor(
                    birthdate_column="birthday",
                ),
            ),
            (
                "feature_selector",
                FeatureSelector(columns=["salary", "weight", "birthday_year", "age"]),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    X_transformed = pipeline.transform(X_test)

    expected_columns = ["salary", "weight", "birthday_year", "age"]
    assert all(col in X_transformed.columns for col in expected_columns)
    assert X_transformed.shape[1] == len(expected_columns)
    assert X_transformed.shape[0] == 1