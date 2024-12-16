from typing import Literal, Optional

from pandas import DataFrame, merge
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
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
        if not vars_and_pipe_dict:
            raise ValueError("The 'vars_and_pipe_dict' cannot be empty.")
        self.vars_and_pipe_dict = vars_and_pipe_dict

    def fit(self, X, y=None):
        return self

    def predict(self, X: DataFrame) -> dict[str, list[int]]:
        """
        predicts
        """

        predictions_dict = {}
        for client_coverage in self.vars_and_pipe_dict:
            predictions_dict[client_coverage] = (
                self.vars_and_pipe_dict[client_coverage].predict(X).ravel().tolist()
            )

        return predictions_dict

    def predict_proba(
        self,
        X: DataFrame,
        bin_classif_return_classes_idx: Optional[Literal[0, 1]] = None,
    ) -> dict[str, list[float]]:
        """
        predicts proba

        Args:

            - `bin_classif_return_classes_idx`: mostly usefull for binary classification,
            returns either the 0 (P(X=0)) probability class, 1 probability class (P(X=1))
            or all probability classes predictions.
        """

        assert (
            bin_classif_return_classes_idx == 0
            or bin_classif_return_classes_idx == 1
            or bin_classif_return_classes_idx is None
        ), f"`return_classes_idx` should either 0, 1 or None, not {bin_classif_return_classes_idx}"

        predictions_dict = {}
        for client_coverage in self.vars_and_pipe_dict:
            probas = self.vars_and_pipe_dict[client_coverage].predict_proba(X)
            if bin_classif_return_classes_idx == 0:
                predictions_dict[client_coverage] = probas[:, 0].ravel().tolist()
            elif bin_classif_return_classes_idx == 1:
                predictions_dict[client_coverage] = probas[:, 1].ravel().tolist()
            else:
                predictions_dict[client_coverage] = probas.ravel().tolist()

        return predictions_dict


class JoinTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""

    def __init__(
        self,
        *,
        on: str,
        other_df: DataFrame,
        how: Literal["inner", "left", "right"],
        drop_key: bool = True,
    ):
        """todo"""
        self.on = on
        self.other_df = other_df
        self.how = how
        self.drop_key = drop_key

    def fit(self, x, y=None):
        return self

    def transform(self, x: DataFrame):
        self._assert_df(x)
        self._assert_df(self.other_df)

        if self.drop_key:
            return merge(x, self.other_df, how=self.how, on=self.on).drop(
                columns=[self.on]
            )
        else:
            return merge(x, self.other_df, how=self.how, on=self.on)

        # return self.transform_func(x) if callable(self.transform_func) else x

    def _health_checks(self, df) -> None:
        """todo"""
        self._assert_df(df)
        self._check_feature_names(df)

    def _assert_df(self, df) -> None:
        """todo"""
        assert isinstance(df, DataFrame)

    def _check_key_inside_cols(self, df) -> None:
        """todo"""
        assert self.key in df.cols

    def get_feature_names(self):
        return self.cols


class CompositePredictor(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn transformer for aggregating all combinations of
    outer/inner predictions.
    -----------------
    **Inputs**:
        - `models_dict`: dict that contains the outer as keys and
          for each outer, another dictionary of inner as keys
          with the associated classifier model (or pipeline).
    **Output**:
        Pandas DataFrame aggregation of predictions
    """

    def __init__(self, models_dict: dict):
        if not models_dict:
            raise ValueError("The 'models_dict' cannot be empty.")
        self.models_dict = models_dict

    def fit(self, X, y=None):
        return self

    def predict(self, X) -> dict[str, list[int]]:
        """
        predicts the labels for each outer/inner combination
        """

        predictions_dict = {}

        for outer, inners in self.models_dict.items():
            predictions_dict[outer] = {}

            for inner, model in inners.items():
                predictions_dict[outer][inner] = model.predict(X).ravel().tolist()

        return predictions_dict

    def predict_proba(
        self, X, bin_classif_return_classes_idx: Optional[Literal[0, 1]] = None
    ) -> dict[str, list[float]]:
        """
        predicts the probabilities for each outer/inner combination.

        Args:
            - `bin_classif_return_classes_idx`: mostly useful for binary classification,
              returns either the 0 (P(X=0)) or 1 (P(X=1)) probability class or all classes.
        """

        assert (
            bin_classif_return_classes_idx == 0
            or bin_classif_return_classes_idx == 1
            or bin_classif_return_classes_idx is None
        ), f"`bin_classif_return_classes_idx` should either 0, 1 or None, not {bin_classif_return_classes_idx}"

        predictions_dict = {}

        for outer, inners in self.models_dict.items():
            predictions_dict[outer] = {}

            for inner, model in inners.items():
                probas = model.predict_proba(X)

                if bin_classif_return_classes_idx == 0:
                    predictions_dict[outer][inner] = probas[:, 0].ravel().tolist()
                elif bin_classif_return_classes_idx == 1:
                    predictions_dict[outer][inner] = probas[:, 1].ravel().tolist()
                else:
                    predictions_dict[outer][inner] = probas.ravel().tolist()

        return predictions_dict


if __name__ == "__main__":
    from sklearn import datasets, tree

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

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

    single_row = X[[50]]

    print(
        f"""
        Probability prediction for all classes:
        { multi_predictor.predict_proba(single_row) }
        """
    )
    print(
        f"""
        Probability prediction for 0 class:
        {multi_predictor.predict_proba(single_row, bin_classif_return_classes_idx=0)}
        """
    )
    print(
        f"""
        Probability prediction for 1 class:
        {multi_predictor.predict_proba(single_row, bin_classif_return_classes_idx=1)}
        """
    )
    print(
        f"""
        Class predictions:
        {multi_predictor.predict(single_row)}
        """
    )
