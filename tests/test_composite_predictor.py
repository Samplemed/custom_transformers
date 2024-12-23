import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from custom_transformers.multi_predictor import CompositePredictor


@pytest.fixture(scope="class")
def setup_models():
    """
    Setup a simple dataset and example models for testing.
    This fixture creates a synthetic dataset and a models dictionary with
    outer and inner combinations as keys.
    """
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_1 = Pipeline([("clf", RandomForestClassifier(n_estimators=10))])
    model_2 = Pipeline([("clf", RandomForestClassifier(n_estimators=10))])

    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    models_dict = {
        "outer_1": {"inner_1": model_1, "inner_2": model_2},
        "outer_2": {"inner_1": model_1, "inner_2": model_2},
    }

    return X_test, models_dict


def test_initialization_empty_dict():
    """Test that ValueError is raised when models_dict is empty"""
    with pytest.raises(ValueError):
        CompositePredictor(models_dict={})


def test_predict_structure(setup_models):
    """Test the predict method of CompositePredictor"""
    X_test, models_dict = setup_models

    transformer = CompositePredictor(models_dict=models_dict)
    predictions = transformer.predict(X_test)

    assert isinstance(predictions, dict)
    assert "outer_1" in predictions
    assert "outer_2" in predictions
    assert "inner_1" in predictions["outer_1"]
    assert "inner_2" in predictions["outer_1"]
    assert isinstance(predictions["outer_1"]["inner_1"], list)
    assert isinstance(predictions["outer_1"]["inner_2"], list)


def test_predict_values(setup_models):
    """Test the predict method values for outer/inner combinations"""
    X_test, models_dict = setup_models

    transformer = CompositePredictor(models_dict=models_dict)
    predictions = transformer.predict(X_test)

    for _, inners in predictions.items():
        for _, prediction in inners.items():
            assert isinstance(prediction, list)
            assert len(prediction) == len(X_test)


def test_predict_proba_structure(setup_models):
    """Test the predict_proba method of CompositePredictor"""
    X_test, models_dict = setup_models

    transformer = CompositePredictor(models_dict=models_dict)
    probas = transformer.predict_proba(X_test)

    assert isinstance(probas, dict)
    assert "outer_1" in probas
    assert "outer_2" in probas
    assert "inner_1" in probas["outer_1"]
    assert "inner_2" in probas["outer_1"]
    assert isinstance(probas["outer_1"]["inner_1"], list)
    assert isinstance(probas["outer_1"]["inner_2"], list)


def test_predict_proba_values(setup_models):
    """Test the predict_proba method values for outer/inner combinations"""
    X_test, models_dict = setup_models

    transformer = CompositePredictor(models_dict=models_dict)
    probas = transformer.predict_proba(X_test)

    for _, inners in probas.items():
        for _, proba in inners.items():
            assert isinstance(proba, list)
            assert (len(proba) / 2) == len(X_test)
            # For binary classification, the proba will be 2 values per sample (for class 0 and 1)
            # Thus, ensure it is a list of probabilities per sample
            assert all(0 <= p <= 1 for p in proba)


def test_predict_proba_binary_classification(setup_models):
    """Test the predict_proba method with a binary classification problem (class 0)"""
    X_test, models_dict = setup_models

    transformer = CompositePredictor(models_dict=models_dict)
    probas = transformer.predict_proba(X_test, bin_classif_return_classes_idx=0)

    for _, inners in probas.items():
        for _, proba in inners.items():
            assert isinstance(proba, list)
            assert len(proba) == len(X_test)
            assert all(0 <= p <= 1 for p in proba)


def test_predict_proba_binary_classification_class_1(setup_models):
    """Test the predict_proba method with a binary classification problem (class 1)"""
    X_test, models_dict = setup_models

    transformer = CompositePredictor(models_dict=models_dict)
    probas = transformer.predict_proba(X_test, bin_classif_return_classes_idx=1)

    for _, inners in probas.items():
        for _, proba in inners.items():
            assert isinstance(proba, list)
            assert len(proba) == len(X_test)
            assert all(0 <= p <= 1 for p in proba)
