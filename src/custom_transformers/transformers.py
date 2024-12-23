"""
Custom scikit-learn transformers.
"""

from collections import Counter
from datetime import datetime
from typing import Literal

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

    def __init__(self, columns: list[str], drop_cols: bool = False):
        self.columns = columns
        self.drop_cols = drop_cols

        column_counts = Counter(self.columns)
        duplicates = [col for col, count in column_counts.items() if count > 1]

        if duplicates:
            raise ValueError(
                f"Duplicate columns found in the columns list: {', '.join(duplicates)}."
            )

    def fit(self, X: pd.DataFrame, y=None):
        """
        fit
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        transform
        """
        if self.drop_cols:
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise KeyError(f"{missing_cols} not found in axis")

            return X.drop(columns=self.columns)
        else:
            return X[self.columns]


# TODO: optimize this to avoid copying, etc
class YMDExtractor(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for selecting columns in specified order.
    NOTE: a copy of the dataframe will be made!
    """

    def __init__(
        self,
        columns: list[str],
        ymd_to_extract: tuple[
            Literal["year"], Literal["month"], Literal["day"], Literal["weekday"]
        ] = (
            "year",
            "month",
            "day",
            "weekday",
        ),
        drop_original_cols: bool = False,
    ):
        self.columns = columns
        self.ymd_to_extract = ymd_to_extract
        self.drop_original_cols = drop_original_cols

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
        X_copy = X.copy()
        if "year" in self.ymd_to_extract:
            for col in self.columns:
                X_copy[f"{col}_year"] = pd.to_datetime(X_copy[col])
                X_copy[f"{col}_year"] = X_copy[f"{col}_year"].dt.year
        if "month" in self.ymd_to_extract:
            for col in self.columns:
                X_copy[f"{col}_month"] = pd.to_datetime(X_copy[col])
                X_copy[f"{col}_month"] = X_copy[f"{col}_month"].dt.month
        if "day" in self.ymd_to_extract:
            for col in self.columns:
                X_copy[f"{col}_day"] = pd.to_datetime(X_copy[col])
                X_copy[f"{col}_day"] = X_copy[f"{col}_day"].dt.day
        if "weekday" in self.ymd_to_extract:
            for col in self.columns:
                X_copy[f"{col}_weekday"] = pd.to_datetime(X_copy[col])
                X_copy[f"{col}_weekday"] = X_copy[f"{col}_weekday"].dt.weekday

        if self.drop_original_cols:
            X_copy.drop(col, axis=1, inplace=True)

        return X_copy


class AgeExtractor(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for calculating the age of someone.
    Inputs:
        - `birthdate_column`: name of the birthdate column in the dataframe
        - `base_year`: the year to be considered. If set to `None`, current year will be used.

    Outputs:
        A copy of the dataframe with an extra column, named `age`.

    NOTE: a copy of the dataframe will be made!
    """

    def __init__(
        self,
        birthdate_column: str,
        base_year: int | None = None,
        drop_original_cols: bool = False,
    ):
        self.birthdate_column = birthdate_column
        self.base_year = base_year if base_year is not None else datetime.now().year
        self.drop_original_cols = drop_original_cols

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

        X_copy = X.copy()
        X_copy["age"] = pd.to_datetime(X_copy[self.birthdate_column])
        X_copy["age"] = self.base_year - X_copy["age"].dt.year
        if self.drop_original_cols:
            X_copy.drop(self.birthdate_column, axis=1, inplace=True)

        return X_copy


class BmiCalculator(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer calculate bmi, based on weight (kilograms)
    and height (meters).
    """

    def __init__(
        self,
        height_col: str,
        weight_col: str,
        drop_original_cols: bool,
        trefethen: bool = False,
    ):
        self.height_col = height_col
        self.weight_col = weight_col
        self.drop_original_cols = drop_original_cols
        self.trefethen = trefethen

    def calculate_bmi(self, pd_row) -> float:
        if pd_row[self.height_col] == 0:
            raise ZeroDivisionError("Height cannot be zero for BMI calculation.")

        if self.trefethen:
            return 1.3 * (pd_row[self.weight_col] / (pd_row[self.height_col] ** 2.5))
        else:
            return pd_row[self.weight_col] / (pd_row[self.height_col] ** 2)

    def fit(self, X: pd.DataFrame, y=None):
        """
        fit
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        transform
        """
        X_copy = X.copy()
        X_copy["bmi"] = X_copy.apply(self.calculate_bmi, axis=1)

        if self.drop_original_cols:
            X_copy.drop([self.height_col, self.weight_col], axis=1, inplace=True)

        return X_copy


if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

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
                AgeExtractor(birthdate_column="birthday", drop_original_cols=True),
            ),
            (
                "bmi_calculator",
                BmiCalculator(
                    height_col="height",
                    weight_col="weight",
                    drop_original_cols=True,
                    trefethen=False,
                ),
            ),
            ("classifier", RandomForestClassifier()),
        ]
    )

    print(pipeline[0:3].fit_transform(X_train))

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Predicted: {y_pred[0]}, Actual: {y_test.values[0]}")
