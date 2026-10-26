"""Train an untuned XGBoost classifier for wine quality.

XGBClassifier is constructed with no arguments. In XGBoost 3.4 that leaves
the booster parameters unset, and the library fills its own defaults
(100 trees, learning rate 0.3, max depth 6, histogram splits). Because the
target has five classes, the classifier selects multi:softprob during fit.
The constructor default objective is binary:logistic.

XGBClassifier requires integer class codes. A LabelEncoder is fit on the
training labels only, and predictions are mapped back to the original names.
No feature scaling is applied; tree splits are scale-invariant.

The split matches the other models: 80/20, stratified, random_state=42.
Five-fold cross-validation stays on the training rows. The held-out 20% is
used once, for the reported test metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"
MODEL_PATH = ROOT / "models" / "xgboost.joblib"
METRICS_PATH = ROOT / "reports" / "xgboost_metrics.json"

LABELS = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


class DefaultXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier at library defaults, accepting the original string labels."""

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(np.asarray(y))
        self.model_ = XGBClassifier()
        self.model_.fit(X, y_encoded)
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        encoded = np.asarray(self.model_.predict(X)).astype(int)
        return self.encoder_.inverse_transform(encoded)


def round_metric(value: float) -> float:
    return round(float(value), 4)


def resolved_hyperparameters(model: XGBClassifier) -> dict:
    """Read the booster values XGBoost filled in after an argument-free fit."""
    config = json.loads(model.get_booster().save_config())
    tree = config["learner"]["gradient_booster"]["tree_train_param"]
    train = config["learner"]["learner_train_param"]
    gbtree = config["learner"]["gradient_booster"]["gbtree_train_param"]

    def as_number(value: str) -> float | int:
        number = float(value)
        if number.is_integer():
            return int(number)
        return round_metric(number)

    params = model.get_params()
    return {
        "n_estimators": int(model.get_booster().num_boosted_rounds()),
        "objective": train["objective"],
        "constructor_objective": params["objective"],
        "booster": train["booster"],
        "multi_strategy": train["multi_strategy"],
        "tree_method": gbtree["tree_method"],
        "updater": gbtree["updater"],
        "learning_rate": as_number(tree["learning_rate"]),
        "max_depth": as_number(tree["max_depth"]),
        "min_child_weight": as_number(tree["min_child_weight"]),
        "gamma": as_number(tree["gamma"]),
        "subsample": as_number(tree["subsample"]),
        "colsample_bytree": as_number(tree["colsample_bytree"]),
        "reg_alpha": as_number(tree["reg_alpha"]),
        "reg_lambda": as_number(tree["reg_lambda"]),
        "max_bin": as_number(tree["max_bin"]),
        "grow_policy": tree["grow_policy"],
        "sampling_method": tree["sampling_method"],
        "random_state": params["random_state"],
        "enable_categorical": params["enable_categorical"],
        "importance_type": params["importance_type"] or "gain",
    }


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

    model = DefaultXGBClassifier()
    model.fit(X_train, y_train)
    classifier: XGBClassifier = model.model_

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        DefaultXGBClassifier(),
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

    importances = sorted(
        (
            {
                "feature": feature,
                "gain_importance": round_metric(gain),
                "permutation_importance_mean": round_metric(mean),
                "permutation_importance_std": round_metric(std),
            }
            for feature, gain, mean, std in zip(
                X.columns,
                classifier.feature_importances_,
                perm.importances_mean,
                perm.importances_std,
            )
        ),
        key=lambda row: row["permutation_importance_mean"],
        reverse=True,
    )

    metrics = {
        "model": "XGBClassifier",
        "estimator": "XGBClassifier() with LabelEncoder on the target",
        "source": DATA_PATH.name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": LABELS,
        "label_encoding": {
            "method": "LabelEncoder",
            "fit_on": "training rows only",
            "classes_in_encoder_order": [str(label) for label in model.encoder_.classes_],
        },
        "hyperparameters": resolved_hyperparameters(classifier),
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
        "Defaults: "
        f"{metrics['hyperparameters']['n_estimators']} trees, "
        f"learning_rate={metrics['hyperparameters']['learning_rate']}, "
        f"max_depth={metrics['hyperparameters']['max_depth']}, "
        f"objective={metrics['hyperparameters']['objective']}"
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
