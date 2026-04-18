"""
logistic_regression.py

Train and evaluate a Logistic Regression classifier for crash severity prediction.

Pipeline:
    1. Load Xraw and y
    2. Train/test split (80/20, stratified)
    3. Fit preprocessing pipeline on training data only
    4. Transform train/test
    5. Apply SMOTE on training data only
    6. Train Logistic Regression
    7. Evaluate and save outputs

Run:
    python code/logistic_regression.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessing_pipeline


XRAW_PATH = "data/processed/Xraw.parquet"
Y_PATH = "data/processed/y.npy"

OUTPUT_DIR = Path("figures/model_reports")
REPORT_PATH = OUTPUT_DIR / "logistic_regression_report.txt"
METRICS_JSON_PATH = OUTPUT_DIR / "logistic_regression_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "logistic_regression_confusion_matrix.npy"
Y_TEST_PATH = OUTPUT_DIR / "logistic_regression_y_test.npy"
Y_PRED_PATH = OUTPUT_DIR / "logistic_regression_y_pred.npy"
COEFFICIENTS_PATH = OUTPUT_DIR / "logistic_regression_coefficients.csv"


RANDOM_STATE = 35
TEST_SIZE = 0.20

LR_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}


def save_text_report(report_text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(XRAW_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    y = np.asarray(y).astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessing_pipeline()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_processed)

    labels = ["None", "Injury", "Fatal"]

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    clf_report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    clf_report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0,
    )

    np.save(CONFUSION_MATRIX_PATH, cm)
    np.save(Y_TEST_PATH, y_test)
    np.save(Y_PRED_PATH, y_pred)

    # Save coefficients in a long/simple format
    coef_rows = []
    for class_index, class_name in enumerate(model.classes_):
        for feature_name, coef_value in zip(feature_names, model.coef_[class_index]):
            coef_rows.append(
                {
                    "class": class_name,
                    "feature": feature_name,
                    "coefficient": float(coef_value),
                    "abs_coefficient": float(abs(coef_value)),
                }
            )

    coef_df = pd.DataFrame(coef_rows).sort_values(
        ["class", "abs_coefficient"], ascending=[True, False]
    )
    coef_df.to_csv(COEFFICIENTS_PATH, index=False)

    metrics_payload = {
        "model": "LogisticRegression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "smote_used": True,
        "train_rows_before_smote": int(len(y_train)),
        "train_rows_after_smote": int(len(y_train_resampled)),
        "test_rows": int(len(y_test)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report_dict,
        "lr_params": LR_PARAMS,
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    y_train_counts = pd.Series(y_train).value_counts().to_dict()
    y_train_resampled_counts = pd.Series(y_train_resampled).value_counts().to_dict()
    y_test_counts = pd.Series(y_test).value_counts().to_dict()

    report_lines = []
    report_lines.append("LOGISTIC REGRESSION MODEL REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Data")
    report_lines.append(f"- X shape before split: {X.shape}")
    report_lines.append(f"- y shape before split: {y.shape}")
    report_lines.append(f"- Train size: {X_train.shape[0]}")
    report_lines.append(f"- Test size:  {X_test.shape[0]}")
    report_lines.append(f"- Train/test split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    report_lines.append("")

    report_lines.append("Class distribution")
    report_lines.append(f"- Training set before SMOTE: {y_train_counts}")
    report_lines.append(f"- Training set after SMOTE:  {y_train_resampled_counts}")
    report_lines.append(f"- Test set:                 {y_test_counts}")
    report_lines.append("")

    report_lines.append("Preprocessing")
    report_lines.append("- Preprocessing pipeline was fit on the training split only.")
    report_lines.append("- Categorical features: impute most frequent + one-hot encode")
    report_lines.append("- Numerical features: impute median + standardize")
    report_lines.append("- Date/time features: extract simple date/time features + impute + standardize")
    report_lines.append("")

    report_lines.append("SMOTE")
    report_lines.append("- Applied only to the transformed training set.")
    report_lines.append(f"- Training rows before SMOTE: {len(y_train)}")
    report_lines.append(f"- Training rows after SMOTE:  {len(y_train_resampled)}")
    report_lines.append("")

    report_lines.append("Logistic Regression parameters")
    for k, v in LR_PARAMS.items():
        report_lines.append(f"- {k}: {v}")
    report_lines.append("")

    report_lines.append("Evaluation metrics")
    report_lines.append(f"- Accuracy:    {acc:.4f}")
    report_lines.append(f"- Macro F1:    {macro_f1:.4f}")
    report_lines.append(f"- Weighted F1: {weighted_f1:.4f}")
    report_lines.append("")

    report_lines.append("Confusion matrix")
    report_lines.append(f"- Label order: {labels}")
    report_lines.append(str(cm))
    report_lines.append("")

    report_lines.append("Classification report")
    report_lines.append(clf_report_text)
    report_lines.append("")

    report_lines.append("Top 10 coefficients by class (absolute value)")
    for class_name in model.classes_:
        report_lines.append(f"- Class: {class_name}")
        top_class_rows = coef_df[coef_df["class"] == class_name].head(10)
        for _, row in top_class_rows.iterrows():
            report_lines.append(
                f"    - {row['feature']}: coef={row['coefficient']:.6f}"
            )
    report_lines.append("")

    report_lines.append("Saved outputs")
    report_lines.append(f"- Report: {REPORT_PATH}")
    report_lines.append(f"- Metrics JSON: {METRICS_JSON_PATH}")
    report_lines.append(f"- Confusion matrix: {CONFUSION_MATRIX_PATH}")
    report_lines.append(f"- y_test: {Y_TEST_PATH}")
    report_lines.append(f"- y_pred: {Y_PRED_PATH}")
    report_lines.append(f"- Coefficients CSV: {COEFFICIENTS_PATH}")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    save_text_report(report_text, REPORT_PATH)

    print("Logistic Regression training complete.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()