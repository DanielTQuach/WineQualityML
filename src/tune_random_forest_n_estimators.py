"""Sweep the tree count for the accuracy-best random forest.

The other settings stay at the highest-accuracy cell from the 240-combination
grid: min_samples_leaf=2, max_depth=20, max_features=2, and
class_weight="balanced_subsample". Scoring is 5-fold accuracy on the
stratified training split. The held-out 20% is not used.

The figure is written to reports/random_forest_n_estimators_accuracy.png.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"
FIGURE_PATH = ROOT / "reports" / "random_forest_n_estimators_accuracy.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TREE_COUNTS = [10, 25, 50, 100, 200, 300]

MIN_SAMPLES_LEAF = 2
MAX_DEPTH = 20
MAX_FEATURES = 2
CLASS_WEIGHT = "balanced_subsample"


def round_metric(value: float) -> float:
    return round(float(value), 4)


def evaluate(n_estimators: int, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=MAX_DEPTH,
        max_features=MAX_FEATURES,
        class_weight=CLASS_WEIGHT,
        criterion="gini",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    accuracy = scores["test_score"]
    return {
        "n_estimators": n_estimators,
        "accuracy_mean": round_metric(accuracy.mean()),
        "accuracy_std": round_metric(accuracy.std()),
    }


def save_figure(results: list[dict]) -> None:
    counts = [str(row["n_estimators"]) for row in results]
    means = [row["accuracy_mean"] for row in results]
    stds = [row["accuracy_std"] for row in results]
    best = max(results, key=lambda row: row["accuracy_mean"])
    colors = [
        "#111827" if row["n_estimators"] == best["n_estimators"] else "#1d4e89"
        for row in results
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    positions = list(range(len(results)))
    ax.bar(
        positions,
        means,
        yerr=stds,
        color=colors,
        ecolor="#94a3b8",
        capsize=3,
        width=0.72,
        zorder=2,
    )
    for position, mean, std in zip(positions, means, stds):
        ax.text(
            position,
            mean + std + 0.004,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    low = min(mean - std for mean, std in zip(means, stds))
    high = max(mean + std for mean, std in zip(means, stds))
    ax.set_ylim(max(0, low - 0.02), high + 0.03)
    ax.set_xticks(positions, counts)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("5-fold mean accuracy")
    ax.set_title(
        "Accuracy by tree count, with the other settings held at the best cell\n"
        "Leaf 2, depth 20, 2 features per split, balanced subsample weights\n"
        "5-fold mean on the 4,000 training rows. Gray lines are the fold standard deviation.\n"
        f"The dark bar is the highest mean: {best['n_estimators']} trees, "
        f"{best['accuracy_mean']:.3f}."
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    y = df["quality"]
    X = df.drop(columns=["quality"])
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = [evaluate(count, X_train, y_train) for count in TREE_COUNTS]
    best = max(results, key=lambda row: row["accuracy_mean"])
    save_figure(results)

    print(f"Train rows: {len(X_train)}")
    for row in results:
        print(
            f"{row['n_estimators']:>4} trees  "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        )
    print(
        "Highest accuracy: "
        f"{best['accuracy_mean']} ± {best['accuracy_std']} "
        f"at {best['n_estimators']} trees"
    )
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
