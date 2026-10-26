"""Train an untuned SVM classifier for wine quality.

SVC is constructed with no arguments, so every hyperparameter stays at the
scikit-learn default: RBF kernel, C=1, gamma="scale", no class weights.

Features are standardized with StandardScaler at its own defaults. The RBF
kernel is not scale-invariant. The scaler is fit on training rows only.
Cross-validation uses a fresh pipeline so each fold refits its own scaler.

The split matches the other models: 80/20, stratified, random_state=42.
Five-fold cross-validation stays on the training rows. The held-out 20% is
used once, for the reported test metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"
MODEL_PATH = ROOT / "models" / "svm.joblib"
METRICS_PATH = ROOT / "reports" / "svm_metrics.json"

LABELS = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def round_metric(value: float) -> float:
    return round(float(value), 4)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC()),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df["quality"]
    X = df.drop(columns=["quality"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline()
    model.fit(X_train, y_train)
    classifier: SVC = model.named_steps["classifier"]
    scaler: StandardScaler = model.named_steps["scaler"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        make_pipeline(),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        n_jobs=-1,
    )

    y_pred = model.predict(X_test)
    majority = y_train.mode().iloc[0]
    y_baseline = pd.Series(majority, index=y_test.index)

    report = classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    per_class = []
    for label in LABELS:
        stats = report[label]
        per_class.append(
            {
                "label": label,
                "precision": round_metric(stats["precision"]),
                "recall": round_metric(stats["recall"]),
                "f1": round_metric(stats["f1-score"]),
                "support": int(stats["support"]),
            }
        )

    class_index = {label: i for i, label in enumerate(classifier.classes_)}
    importances = sorted(
        (
            {
                "feature": feature,
                "permutation_importance_mean": round_metric(mean),
                "permutation_importance_std": round_metric(std),
            }
            for feature, mean, std in zip(
                X.columns,
                perm.importances_mean,
                perm.importances_std,
            )
        ),
        key=lambda row: row["permutation_importance_mean"],
        reverse=True,
    )

    params = classifier.get_params()
    metrics = {
        "model": "SVC",
        "estimator": "Pipeline(StandardScaler, SVC)",
        "source": DATA_PATH.name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": LABELS,
        "preprocessing": {
            "StandardScaler": {
                "with_mean": bool(scaler.with_mean),
                "with_std": bool(scaler.with_std),
                "fit_on": "training fold only",
            }
        },
        "hyperparameters": {
            "C": params["C"],
            "kernel": params["kernel"],
            "degree": params["degree"],
            "gamma": params["gamma"],
            "coef0": params["coef0"],
            "shrinking": params["shrinking"],
            "tol": params["tol"],
            "max_iter": params["max_iter"],
            "decision_function_shape": params["decision_function_shape"],
            "break_ties": params["break_ties"],
            "class_weight": params["class_weight"],
            "random_state": params["random_state"],
            "n_support": {
                label: int(classifier.n_support_[class_index[label]]) for label in LABELS
            },
            "n_iter": [int(n) for n in classifier.n_iter_],
        },
        "majority_class_baseline": {
            "class": str(majority),
            "test_accuracy": round_metric(accuracy_score(y_test, y_baseline)),
            "test_balanced_accuracy": round_metric(
                balanced_accuracy_score(y_test, y_baseline)
            ),
            "test_f1_macro": round_metric(
                f1_score(y_test, y_baseline, labels=LABELS, average="macro", zero_division=0)
            ),
        },
        "cross_validation_on_train": {
            "folds": 5,
            "accuracy_mean": round_metric(cv_scores["test_accuracy"].mean()),
            "accuracy_std": round_metric(cv_scores["test_accuracy"].std()),
            "balanced_accuracy_mean": round_metric(
                cv_scores["test_balanced_accuracy"].mean()
            ),
            "balanced_accuracy_std": round_metric(
                cv_scores["test_balanced_accuracy"].std()
            ),
            "f1_macro_mean": round_metric(cv_scores["test_f1_macro"].mean()),
            "f1_macro_std": round_metric(cv_scores["test_f1_macro"].std()),
        },
        "test": {
            "accuracy": round_metric(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": round_metric(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": round_metric(report["macro avg"]["f1-score"]),
            "f1_weighted": round_metric(report["weighted avg"]["f1-score"]),
            "per_class": per_class,
            "confusion_matrix": {
                "rows": "true label",
                "columns": "predicted label",
                "labels": LABELS,
                "values": matrix.tolist(),
            },
        },
        "feature_importance": importances,
        "train_class_counts": {label: int((y_train == label).sum()) for label in LABELS},
        "test_class_counts": {label: int((y_test == label).sum()) for label in LABELS},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Train rows: {metrics['n_train']}  Test rows: {metrics['n_test']}")
    print(
        f"Kernel: {params['kernel']}  C: {params['C']}  gamma: {params['gamma']}  "
        f"class_weight: {params['class_weight']}"
    )
    print(
        f"Majority baseline ({majority}) accuracy: "
        f"{metrics['majority_class_baseline']['test_accuracy']}"
    )
    print(f"Test accuracy: {metrics['test']['accuracy']}")
    print(f"Test balanced accuracy: {metrics['test']['balanced_accuracy']}")
    print(f"Test macro F1: {metrics['test']['f1_macro']}")
    print(
        "5-fold train macro F1: "
        f"{metrics['cross_validation_on_train']['f1_macro_mean']} ± "
        f"{metrics['cross_validation_on_train']['f1_macro_std']}"
    )
    print("Confusion matrix (rows true, columns predicted):")
    print(LABELS)
    for row in matrix.tolist():
        print(row)
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
