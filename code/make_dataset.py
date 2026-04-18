"""
make_dataset.py

Loads the sampled collisions CSV, cleans it, builds a simple raw feature matrix Xraw
and target vector y, and saves:

    - cleaned dataset as parquet
    - Xraw as parquet
    - y as .npy
    - text report describing the cleaning process

Run:
    python code/make_dataset.py
"""

import os
import re

import numpy as np
import pandas as pd

from preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    DATE_TIME_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
)


RAW_DATA_PATH = "data/motor_vehicle_collisions_sample.csv"
CLEANED_DATA_PATH = "data/processed/motor_vehicle_collisions_cleaned.parquet"
XRAW_PATH = "data/processed/Xraw.parquet"
Y_PATH = "data/processed/y.npy"
REPORT_PATH = "data/processed/make_dataset_report.txt"


NUMBER_PREFIX = "number_of_"

PERSONS_INJURED_COL = "number_of_persons_injured"
PERSONS_KILLED_COL = "number_of_persons_killed"


def standardize_column_name(name):
    """
    Convert column names to lowercase snake_case-like format.
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_empty_strings_to_nan(df):
    """
    Replace empty strings / whitespace-only strings with NaN.
    Returns updated df and count of cells changed.
    """
    before_missing = df.isna().sum().sum()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    after_missing = df.isna().sum().sum()
    changed = int(after_missing - before_missing)
    return df, changed


def enforce_nonnegative_number_columns(df):
    """
    Convert all number_of_* columns to numeric and clip values at 0.
    Returns updated df, list of touched columns, and count of negative values clipped.
    """
    number_cols = [col for col in df.columns if col.startswith(NUMBER_PREFIX)]
    clipped_count = 0

    for col in number_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        negative_mask = df[col] < 0
        clipped_count += int(negative_mask.sum())
        df.loc[negative_mask, col] = 0

    return df, number_cols, clipped_count


def drop_duplicate_collision_ids(df):
    """
    Drop duplicate collision_id rows, keeping the last occurrence.
    """
    if ID_COLUMN not in df.columns:
        return df, 0

    before = len(df)
    df = df.drop_duplicates(subset=[ID_COLUMN], keep="last")
    removed = before - len(df)
    return df, removed


def compute_severity_class(df):
    """
    Severity class:
        None   -> no injuries and no deaths
        Injury -> injuries but no deaths
        Fatal  -> at least one death

    Uses:
        number_of_persons_injured
        number_of_persons_killed
    """
    if PERSONS_INJURED_COL not in df.columns:
        df[PERSONS_INJURED_COL] = 0
    if PERSONS_KILLED_COL not in df.columns:
        df[PERSONS_KILLED_COL] = 0

    injured = pd.to_numeric(df[PERSONS_INJURED_COL], errors="coerce").fillna(0)
    killed = pd.to_numeric(df[PERSONS_KILLED_COL], errors="coerce").fillna(0)

    df[TARGET_COLUMN] = np.where(
        killed >= 1,
        "Fatal",
        np.where(injured >= 1, "Injury", "None"),
    )

    return df


def ensure_feature_columns_exist(df, feature_columns):
    """
    Add any missing feature columns as NaN so downstream code does not break.
    """
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_report(
    rows_before,
    rows_after,
    empty_strings_changed,
    duplicate_rows_removed,
    clipped_negative_values,
    number_cols,
    class_counts,
    cleaned_df,
    xraw_df,
):
    lines = []
    lines.append("MAKE DATASET REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Cleaning summary")
    lines.append(f"- Rows before cleaning: {rows_before}")
    lines.append(f"- Rows after cleaning:  {rows_after}")
    lines.append(f"- Empty/blank string cells converted to NaN: {empty_strings_changed}")
    lines.append(f"- Duplicate collision_id rows removed: {duplicate_rows_removed}")
    lines.append(f"- Negative values clipped to 0 in number_of_* columns: {clipped_negative_values}")
    lines.append("")

    lines.append("Target distribution")
    for cls in ["None", "Injury", "Fatal"]:
        lines.append(f"- {cls}: {class_counts.get(cls, 0)}")
    lines.append("")

    lines.append("Feature column groups")
    lines.append(f"- CATEGORICAL_COLUMNS ({len(CATEGORICAL_COLUMNS)}): {CATEGORICAL_COLUMNS}")
    lines.append(f"- NUMERICAL_COLUMNS ({len(NUMERICAL_COLUMNS)}): {NUMERICAL_COLUMNS}")
    lines.append(f"- DATE_TIME_COLUMNS ({len(DATE_TIME_COLUMNS)}): {DATE_TIME_COLUMNS}")
    lines.append("")

    lines.append("number_of_* columns found")
    lines.append(f"- Count: {len(number_cols)}")
    lines.append(f"- Columns: {number_cols}")
    lines.append("")

    lines.append("Saved Xraw info")
    lines.append(f"- Xraw shape: {xraw_df.shape}")
    lines.append(f"- Xraw columns: {list(xraw_df.columns)}")
    lines.append("")

    lines.append("Missing values in Xraw")
    missing = xraw_df.isna().sum().sort_values(ascending=False)
    for col, val in missing.items():
        lines.append(f"- {col}: {int(val)}")
    lines.append("")

    lines.append("Cleaned dataset columns")
    lines.append(f"- Total columns in cleaned dataset: {len(cleaned_df.columns)}")
    lines.append(f"- Column names: {list(cleaned_df.columns)}")
    lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    rows_before = len(df)

    df.columns = [standardize_column_name(col) for col in df.columns]

    df, empty_strings_changed = normalize_empty_strings_to_nan(df)
    df, number_cols, clipped_negative_values = enforce_nonnegative_number_columns(df)
    df, duplicate_rows_removed = drop_duplicate_collision_ids(df)
    df = compute_severity_class(df)

    feature_columns = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + DATE_TIME_COLUMNS
    df = ensure_feature_columns_exist(df, feature_columns)

    xraw_df = df[feature_columns].copy()
    y = df[TARGET_COLUMN].to_numpy()

    df.to_parquet(CLEANED_DATA_PATH, index=False)
    xraw_df.to_parquet(XRAW_PATH, index=False)
    np.save(Y_PATH, y)

    rows_after = len(df)
    class_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()

    report_text = build_report(
        rows_before=rows_before,
        rows_after=rows_after,
        empty_strings_changed=empty_strings_changed,
        duplicate_rows_removed=duplicate_rows_removed,
        clipped_negative_values=clipped_negative_values,
        number_cols=number_cols,
        class_counts=class_counts,
        cleaned_df=df,
        xraw_df=xraw_df,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Saved cleaned dataset to:", CLEANED_DATA_PATH)
    print("Saved Xraw to:", XRAW_PATH)
    print("Saved y to:", Y_PATH)
    print("Saved report to:", REPORT_PATH)
    print()
    print("Class counts:")
    print(pd.Series(class_counts))


if __name__ == "__main__":
    main()