import pandas as pd
import pytest

from custom_transformers.transformers import YMDExtractor


@pytest.fixture
def date_data():
    return pd.DataFrame({"date_col": ["2020-01-01", "2021-02-02", "2022-03-03"]})


def test_ymd_extractor_extract_year_month_day_weekday(date_data: pd.DataFrame):
    transformer = YMDExtractor(
        columns=["date_col"], ymd_to_extract=("year", "month", "day", "weekday")
    )

    transformer.fit(date_data)
    transformed = transformer.transform(date_data)

    assert "date_col_year" in transformed.columns
    assert "date_col_month" in transformed.columns
    assert "date_col_day" in transformed.columns
    assert "date_col_weekday" in transformed.columns

    assert transformed["date_col_year"].iloc[0] == 2020
    assert transformed["date_col_month"].iloc[0] == 1
    assert transformed["date_col_day"].iloc[0] == 1
    assert transformed["date_col_weekday"].iloc[0] == 2


def test_ymd_extractor_drop_original_columns(date_data: pd.DataFrame):
    transformer = YMDExtractor(
        columns=["date_col"],
        ymd_to_extract=("year", "month", "day", "weekday"),
        drop_original_cols=True,
    )

    _ = transformer.fit(date_data)
    transformed = transformer.transform(date_data)

    assert "date_col" not in transformed.columns


def test_ymd_extractor_partial_fields(date_data: pd.DataFrame):
    transformer = YMDExtractor(columns=["date_col"], ymd_to_extract=("year", "month"))

    transformer.fit(date_data)
    transformed = transformer.transform(date_data)

    assert "date_col_year" in transformed.columns
    assert "date_col_month" in transformed.columns
    assert "date_col_day" not in transformed.columns
    assert "date_col_weekday" not in transformed.columns
