from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline


class MultiPredictor(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn transformer for aggregating all coverages predictions.
    -----------------
    **Inputs**:
        - `vars_and_pipe_dict`: dict that contains the `client_coverage` as
            keys and the associated classifier pipeline as value
    **Output**:
        Pandas DataFrame aggregation of predictions
    """

    def __init__(
        self,
        vars_and_pipe_dict: dict[str, Pipeline],
    ):
        self.vars_and_pipe_dict = vars_and_pipe_dict

    def predict(self, X) -> dict[str, list[int]]:
        """
        predicts
        """

        predictions_dict = {}
        for client_coverage in self.vars_and_pipe_dict:
            predictions_dict[client_coverage] = (
                self.vars_and_pipe_dict[client_coverage].predict(X).ravel().tolist()
            )

        return predictions_dict

    def predict_proba(self, X) -> dict[str, list[float]]:
        """
        predicts proba
        """

        predictions_dict = {}
        for client_coverage in self.vars_and_pipe_dict:
            predictions_dict[client_coverage] = (
                self.vars_and_pipe_dict[client_coverage]
                .predict_proba(X)
                .ravel()
                .tolist()
            )

        return predictions_dict


if __name__ == "__main__":
    from sklearn import datasets, tree

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(X, y)

    multi_predictor = MultiPredictor(
        {
            "clf1": clf,
            "clf2": clf2,
        }
    )

    print(multi_predictor.predict_proba(X))
