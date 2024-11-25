import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from custom_transformers.multi_predictor import JoinTransformer, MultiPredictor


@pytest.fixture
def mock_classifiers() -> tuple[dict[str, Pipeline], np.ndarray]:
    """
    Provides mock classifiers and a feature matrix for testing.
    ----------
    Outputs:
    - A dictionary with string keys and Pipeline values.
    - A feature matrix X as a numpy array.
    """
    clf1 = Pipeline([("classifier", DecisionTreeClassifier(random_state=0))])
    clf2 = Pipeline([("classifier", DecisionTreeClassifier(random_state=0))])

    iris = load_iris()
    X, y = iris.data, iris.target
    clf1.fit(X, y)
    clf2.fit(X, y)

    return {"clf1": clf1, "clf2": clf2}, X


def test_multi_predictor(mock_classifiers):
    vars_and_pipe_dict, X = mock_classifiers
    multi_predictor = MultiPredictor(vars_and_pipe_dict)

    predictions = multi_predictor.predict(X)
    assert isinstance(predictions, dict)
    assert "clf1" in predictions
    assert "clf2" in predictions
    assert len(predictions["clf1"]) == len(X)
    assert len(predictions["clf2"]) == len(X)

    probas = multi_predictor.predict_proba(X)
    assert isinstance(probas, dict)
    assert "clf1" in probas
    assert "clf2" in probas
    assert len(probas["clf1"]) == 3 * len(X)
    assert len(probas["clf2"]) == 3 * len(X)

    probas_class_0 = multi_predictor.predict_proba(X, bin_classif_return_classes_idx=0)
    assert isinstance(probas_class_0, dict)
    assert "clf1" in probas_class_0
    assert "clf2" in probas_class_0
    assert len(probas_class_0["clf1"]) == len(X)
    assert len(probas_class_0["clf2"]) == len(X)

    probas_class_1 = multi_predictor.predict_proba(X, bin_classif_return_classes_idx=1)
    assert isinstance(probas_class_1, dict)
    assert "clf1" in probas_class_1
    assert "clf2" in probas_class_1
    assert len(probas_class_1["clf1"]) == len(X)
    assert len(probas_class_1["clf2"]) == len(X)


def test_multi_predictor_invalid_classifiers():
    with pytest.raises(ValueError):
        multi_predictor = MultiPredictor(vars_and_pipe_dict={})
        multi_predictor.predict([[1, 2, 3, 4]])


def test_join_transformer():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature1": [10, 20, 30],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 4],
            "feature2": [100, 200, 400],
        }
    )

    join_transformer = JoinTransformer(
        on="id", other_df=df2, how="inner", drop_key=True
    )
    result = join_transformer.transform(df1)

    assert "id" not in result.columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns
    assert len(result) == 2


def test_join_transformer_with_drop_key_false():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature1": [10, 20, 30],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 4],
            "feature2": [100, 200, 400],
        }
    )

    join_transformer = JoinTransformer(
        on="id", other_df=df2, how="inner", drop_key=False
    )
    result = join_transformer.transform(df1)

    assert "id" in result.columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns
    assert len(result) == 2


def test_join_transformer_invalid_input():
    with pytest.raises(AssertionError):
        join_transformer = JoinTransformer(on="id", other_df="invalid_df", how="inner")
        join_transformer.transform(pd.DataFrame())
