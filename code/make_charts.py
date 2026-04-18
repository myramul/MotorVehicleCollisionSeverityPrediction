"""
make_charts.py

Rebuild charts from saved dataset/model artifacts.

Charts saved to figures/:
    - class_distr.png
    - bar_most_common_contr_factors.png
    - model_metric_comparison.png
    - rf_per_class_metrics.png
    - lr_per_class_metrics.png
    - rf_confusion_matrix.png
    - lr_confusion_matrix.png

Run:
    python code/make_charts.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Paths
CLEANED_DATA_PATH = "data/processed/motor_vehicle_collisions_cleaned.parquet"

RF_METRICS_PATH = "figures/model_reports/random_forest_metrics.json"
LR_METRICS_PATH = "figures/model_reports/logistic_regression_metrics.json"

RF_CM_PATH = "figures/model_reports/random_forest_confusion_matrix.npy"
LR_CM_PATH = "figures/model_reports/logistic_regression_confusion_matrix.npy"

OUTPUT_DIR = "figures"



# Custom color scheme
COLORS = [
    "#2E0014",
    "#442220",
    "#809848",
    "#B0CA87",
    "#B5DEAD",
]

DPI = 300 # quality of images
CLASS_ORDER = ["None", "Injury", "Fatal"]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_class_distribution(df):
    counts = df["severity_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values, color=[COLORS[0], COLORS[2], COLORS[4]])
    plt.title("Class Distribution")
    plt.xlabel("Severity Class")
    plt.ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distr.png"), dpi=DPI)
    plt.close()


def plot_most_common_contributing_factors(df, top_n=10):
    if "contributing_factor_vehicle_1" not in df.columns:
        print("Skipping contributing factor chart: column not found.")
        return

    counts = (
        df["contributing_factor_vehicle_1"]
        .fillna("Unknown")
        .replace("", "Unknown")
        .value_counts()
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    bars = plt.barh(counts.index[::-1], counts.values[::-1], color=COLORS[3])
    plt.title("Most Common Contributing Factors")
    plt.xlabel("Count")
    plt.ylabel("Contributing Factor")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            ha="left",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bar_most_common_contr_factors.png"), dpi=DPI)
    plt.close()


def plot_model_metric_comparison(rf_metrics, lr_metrics):
    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]
    rf_values = [
        rf_metrics["accuracy"],
        rf_metrics["macro_f1"],
        rf_metrics["weighted_f1"],
    ]
    lr_values = [
        lr_metrics["accuracy"],
        lr_metrics["macro_f1"],
        lr_metrics["weighted_f1"],
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width / 2, rf_values, width, label="Random Forest", color=COLORS[2])
    bars2 = plt.bar(x + width / 2, lr_values, width, label="Logistic Regression", color=COLORS[0])

    plt.xticks(x, metric_names)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Model Metric Comparison")
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_metric_comparison.png"), dpi=DPI)
    plt.close()


def plot_per_class_metrics(metrics_json, out_name, title, color):
    report = metrics_json["classification_report"]

    precision_vals = [report[label]["precision"] for label in CLASS_ORDER]
    recall_vals = [report[label]["recall"] for label in CLASS_ORDER]
    f1_vals = [report[label]["f1-score"] for label in CLASS_ORDER]

    x = np.arange(len(CLASS_ORDER))
    width = 0.25

    plt.figure(figsize=(9, 5))
    bars1 = plt.bar(x - width, precision_vals, width, label="Precision", color=COLORS[0])
    bars2 = plt.bar(x, recall_vals, width, label="Recall", color=COLORS[2])
    bars3 = plt.bar(x + width, f1_vals, width, label="F1", color=color)

    plt.xticks(x, CLASS_ORDER)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=DPI)
    plt.close()


def plot_confusion_matrix(cm, labels, out_name, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Greens")
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=DPI)
    plt.close()


def main():
    ensure_output_dir()

    df = pd.read_parquet(CLEANED_DATA_PATH)

    rf_metrics = load_json(RF_METRICS_PATH)
    lr_metrics = load_json(LR_METRICS_PATH)

    rf_cm = np.load(RF_CM_PATH)
    lr_cm = np.load(LR_CM_PATH)

    plot_class_distribution(df)
    plot_most_common_contributing_factors(df)
    plot_model_metric_comparison(rf_metrics, lr_metrics)

    plot_per_class_metrics(
        rf_metrics,
        out_name="rf_per_class_metrics.png",
        title="Random Forest Per-Class Metrics",
        color=COLORS[4],
    )

    plot_per_class_metrics(
        lr_metrics,
        out_name="lr_per_class_metrics.png",
        title="Logistic Regression Per-Class Metrics",
        color=COLORS[3],
    )

    plot_confusion_matrix(
        rf_cm,
        labels=CLASS_ORDER,
        out_name="rf_confusion_matrix.png",
        title="Random Forest Confusion Matrix",
    )

    plot_confusion_matrix(
        lr_cm,
        labels=CLASS_ORDER,
        out_name="lr_confusion_matrix.png",
        title="Logistic Regression Confusion Matrix",
    )

    print("Saved charts:")
    print("- figures/class_distr.png")
    print("- figures/bar_most_common_contr_factors.png")
    print("- figures/model_metric_comparison.png")
    print("- figures/rf_per_class_metrics.png")
    print("- figures/lr_per_class_metrics.png")
    print("- figures/rf_confusion_matrix.png")
    print("- figures/lr_confusion_matrix.png")


if __name__ == "__main__":
    main()