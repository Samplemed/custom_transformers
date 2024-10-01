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
        self.vars_and_pipe_dict = vars_and_pipe_dict

    def fit(self, x, y=None):
        return self

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

    def predict_proba(
        self, X, return_all_classes: Optional[bool] = False
    ) -> dict[str, list[float]]:
        """
        predicts proba

        Args:

            - return_all_classes: mostly usefull for binary classification,
            if false returns the class in index 0.
        """
        predictions_dict = {}
        for client_coverage in self.vars_and_pipe_dict:
            probas = self.vars_and_pipe_dict[client_coverage].predict_proba(X)
            if not return_all_classes:
                predictions_dict[client_coverage] = probas[:, 0].ravel().tolist()
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

    single_row = X[[100]]

    print(
        f"""
        Probability prediction for principal class:
        { multi_predictor.predict_proba(single_row) }
        """
    )
    print(
        f"""
        Probability prediction for all classes:
        {multi_predictor.predict_proba(single_row, return_all_classes=True)}
        """
    )
    print(
        f"""
        Class predictions:
        {multi_predictor.predict(single_row)}
        """
    )
