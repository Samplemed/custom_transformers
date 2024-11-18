import pandas as pd
import pytest

from custom_transformers.transformers import AgeExtractor


@pytest.fixture
def age_data():
    return pd.DataFrame({"birthdate": ["2000-01-01", "1990-05-15", "1985-12-25"]})


def test_age_extractor_computation(age_data: pd.DataFrame):
    transformer = AgeExtractor(birthdate_column="birthdate", base_year=2024)

    transformer.fit(age_data)
    transformed = transformer.transform(age_data)

    # Check the age column is added
    assert "age" in transformed.columns
    assert transformed["age"].iloc[0] == 24  # 2000-01-01
    assert transformed["age"].iloc[1] == 34  # 1990-05-15
    assert transformed["age"].iloc[2] == 39  # 1985-12-25


def test_age_extractor_default_base_year(age_data: pd.DataFrame):
    transformer = AgeExtractor(birthdate_column="birthdate")

    transformer.fit(age_data)
    transformed = transformer.transform(age_data)

    assert "age" in transformed.columns
    assert transformed["age"].iloc[0] == 24  # 2000-01-01


def test_age_extractor_missing_value(age_data: pd.DataFrame):
    # Adds missing value in the birthdate column
    age_data_with_na = age_data.copy()
    age_data_with_na.loc[1, "birthdate"] = None

    transformer = AgeExtractor(birthdate_column="birthdate", base_year=2024)

    transformer.fit(age_data_with_na)
    transformed = transformer.transform(age_data_with_na)

    assert pd.isna(transformed["age"].iloc[1])
