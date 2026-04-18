"""
preprocessing.py

Defines the preprocessing pipeline used by model scripts.

Expected usage in model scripts:
    from preprocessing import build_preprocessing_pipeline

    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
"""


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Feature Columns

CATEGORICAL_COLUMNS = [
    "borough",
    "zip_code",
    "on_street_name",
    "cross_street_name",
    "off_street_name",
    "contributing_factor_vehicle_1",
    "contributing_factor_vehicle_2",
    "vehicle_type_code_1",
    "vehicle_type_code_2",
]

NUMERICAL_COLUMNS = [
    "latitude",
    "longitude",
]

DATE_TIME_COLUMNS = [
    "crash_date",
    "crash_time",
]

TARGET_COLUMN = "severity_class"
ID_COLUMN = "collision_id"


class DateTimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Converts crash_date and crash_time into a few simple numeric features.

    Output columns:
        crash_year
        crash_month
        crash_dayofweek
        crash_hour
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # ColumnTransformer may pass a 2D array
            df = pd.DataFrame(X, columns=DATE_TIME_COLUMNS)

        date_series = pd.to_datetime(df["crash_date"], errors="coerce")

        # Parse times like "13:45" or "13:45:00"
        time_series = pd.to_datetime(df["crash_time"], format="%H:%M", errors="coerce")
        if time_series.isna().all():
            time_series = pd.to_datetime(df["crash_time"], errors="coerce")

        out = pd.DataFrame(index=df.index)
        out["crash_year"] = date_series.dt.year
        out["crash_month"] = date_series.dt.month
        out["crash_dayofweek"] = date_series.dt.dayofweek
        out["crash_hour"] = time_series.dt.hour

        return out
    
    def get_feature_names_out(self, input_features=None):
        return [
            "crash_year",
            "crash_month",
            "crash_dayofweek",
            "crash_hour",
        ]


def build_preprocessing_pipeline():
    """
    Returns a preprocessing pipeline for raw feature data Xraw.

    - categorical: most frequent imputation + one-hot encoding
    - numerical: median imputation + scaling
    - datetime: extract simple features + median imputation + scaling
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    datetime_pipeline = Pipeline(
        steps=[
            ("datetime_features", DateTimeFeaturesExtractor()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("num", numerical_pipeline, NUMERICAL_COLUMNS),
            ("dt", datetime_pipeline, DATE_TIME_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessor