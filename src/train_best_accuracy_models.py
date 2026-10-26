"""Train the highest-accuracy random forest, XGBoost, and SVM.

Each specification is the setting with the highest 5-fold accuracy on the
stratified training split. The held-out 20% is not used to choose them.

Random forest, from src/tune_random_forest.py and
src/tune_random_forest_n_estimators.py: 0.439 at 100 trees, leaf size 2,
depth 20, 2 features per split, and balanced subsample weights.

XGBoost, from src/tune_xgboost.py: 0.434 at learning rate 0.05, 100 trees,
depth 4, and min_child_weight 10. The other booster settings stay at the
values held fixed during that search.

SVM, from src/tune_svm.py: 0.430, tied across 26 converged settings. This
file uses the simplest one, a linear kernel with C=0.01. Features are
standardized inside the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        max_depth=20,
        max_features=2,
        class_weight="balanced_subsample",
        criterion="gini",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(
        learning_rate=0.05,
        n_estimators=100,
        max_depth=4,
        min_child_weight=10,
        objective="multi:softprob",
        tree_method="hist",
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


def make_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    C=0.01,
                    class_weight=None,
                    decision_function_shape="ovr",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_xgboost(model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y_train))
    model.fit(X_train, y_encoded)
    return encoder


def predict_xgboost(model: XGBClassifier, encoder: LabelEncoder, X: pd.DataFrame):
    encoded = np.asarray(model.predict(X)).astype(int)
    return encoder.inverse_transform(encoded)


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

    forest = make_random_forest()
    forest.fit(X_train, y_train)

    xgboost_model = make_xgboost()
    encoder = fit_xgboost(xgboost_model, X_train, y_train)

    svm = make_svm()
    svm.fit(X_train, y_train)

    scores = {
        "random forest": accuracy_score(y_test, forest.predict(X_test)),
        "xgboost": accuracy_score(y_test, predict_xgboost(xgboost_model, encoder, X_test)),
        "svm": accuracy_score(y_test, svm.predict(X_test)),
    }
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    for name, score in scores.items():
        print(f"{name} holdout accuracy: {score:.4f}")


if __name__ == "__main__":
    main()
