import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from custom_transformers.transformers import BmiCalculator


@pytest.fixture
def example_dataframe():
    data = {"weight": [70, 80, 60, 90, 75], "height": [1.75, 1.80, 1.65, 1.85, 1.70]}
    return pd.DataFrame(data)


def test_bmi_trefethen(example_dataframe):
    bmi_transformer_trefethen = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=False,
        trefethen=True,
    )

    df_transformed_trefethen = bmi_transformer_trefethen.fit_transform(
        example_dataframe
    )

    expected_bmi = [
        1.3 * (70 / (1.75**2.5)),
        1.3 * (80 / (1.80**2.5)),
        1.3 * (60 / (1.65**2.5)),
        1.3 * (90 / (1.85**2.5)),
        1.3 * (75 / (1.70**2.5)),
    ]

    for i, row in df_transformed_trefethen.iterrows():
        assert pytest.approx(row["bmi"], rel=1e-5) == expected_bmi[i]


def test_bmi_standard(example_dataframe):
    bmi_transformer_standard = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=False,
        trefethen=False,
    )

    df_transformed_standard = bmi_transformer_standard.fit_transform(example_dataframe)

    expected_bmi = [
        70 / (1.75**2),
        80 / (1.80**2),
        60 / (1.65**2),
        90 / (1.85**2),
        75 / (1.70**2),
    ]

    for i, row in df_transformed_standard.iterrows():
        assert pytest.approx(row["bmi"], rel=1e-5) == expected_bmi[i]


def test_drop_original_columns(example_dataframe):
    bmi_transformer_drop_cols = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=True,
        trefethen=False,
    )

    df_transformed_drop_cols = bmi_transformer_drop_cols.fit_transform(
        example_dataframe
    )

    assert "weight" not in df_transformed_drop_cols.columns
    assert "height" not in df_transformed_drop_cols.columns
    assert "bmi" in df_transformed_drop_cols.columns


def test_dont_drop_original_columns(example_dataframe):
    bmi_transformer_dont_drop_cols = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=False,
        trefethen=False,
    )

    df_transformed_dont_drop_cols = bmi_transformer_dont_drop_cols.fit_transform(
        example_dataframe
    )

    assert "weight" in df_transformed_dont_drop_cols.columns
    assert "height" in df_transformed_dont_drop_cols.columns
    assert "bmi" in df_transformed_dont_drop_cols.columns


def test_zero_height():
    data = {"weight": [70], "height": [0]}
    df = pd.DataFrame(data)

    bmi_transformer = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=False,
        trefethen=False,
    )

    with pytest.raises(ZeroDivisionError):
        bmi_transformer.fit_transform(df)


def test_nan_values():
    data = {"weight": [70, 80, None, 90, 75], "height": [1.75, None, 1.65, 1.85, 1.70]}
    df = pd.DataFrame(data)

    bmi_transformer = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=False,
        trefethen=False,
    )

    df_transformed = bmi_transformer.fit_transform(df)
    assert df_transformed["bmi"].isna().sum() == 2


def test_bmi_in_pipeline(example_dataframe):
    bmi_transformer = BmiCalculator(
        height_col="height",
        weight_col="weight",
        drop_original_cols=False,
        trefethen=False,
    )

    pipeline = Pipeline([("bmi_calculator", bmi_transformer)])

    df_transformed_pipeline = pipeline.fit_transform(example_dataframe)

    assert "bmi" in df_transformed_pipeline.columns
    assert df_transformed_pipeline.shape[0] == example_dataframe.shape[0]
