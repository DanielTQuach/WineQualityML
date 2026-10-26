"""Score every combination of the four main XGBoost settings.

The search stays on the stratified training split. The held-out 20% is not
used. subsample, colsample_bytree, gamma, reg_alpha, and reg_lambda stay at
the untuned defaults, and every model uses histogram splits, so the 256
classifiers differ only in learning_rate, n_estimators, max_depth, and
min_child_weight.

Primary score is 5-fold mean accuracy. The figure is written to
reports/xgboost_grid_accuracy.png. No JSON file is written.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"
FIGURE_PATH = ROOT / "reports" / "xgboost_grid_accuracy.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

LEARNING_RATES = [0.05, 0.1, 0.2, 0.3]
TREE_COUNTS = [50, 100, 200, 400]
DEPTH_VALUES = [3, 4, 6, 8]
CHILD_VALUES = [1, 3, 5, 10]

GRID = {
    "learning_rate": LEARNING_RATES,
    "n_estimators": TREE_COUNTS,
    "max_depth": DEPTH_VALUES,
    "min_child_weight": CHILD_VALUES,
}

RATE_LABELS = ["0.05", "0.1", "0.2", "0.3"]
TREE_LABELS = ["50", "100", "200", "400"]
DEPTH_LABELS = ["3", "4", "6", "8"]
CHILD_LABELS = ["1", "3", "5", "10"]


def round_metric(value: float) -> float:
    return round(float(value), 4)


def rate_label(value: float) -> str:
    return f"{value:g}"


def is_default(params: dict) -> bool:
    return (
        params["learning_rate"] == 0.3
        and params["n_estimators"] == 100
        and params["max_depth"] == 6
        and params["min_child_weight"] == 1
    )


def evaluate(params: dict, X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    model = XGBClassifier(
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
        **params,
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
    return {
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_child_weight": params["min_child_weight"],
        "learning_rate_label": rate_label(params["learning_rate"]),
        "n_estimators_label": str(params["n_estimators"]),
        "max_depth_label": str(params["max_depth"]),
        "min_child_weight_label": str(params["min_child_weight"]),
        "is_default": is_default(params),
        "accuracy_mean": round_metric(scores["test_score"].mean()),
        "accuracy_std": round_metric(scores["test_score"].std()),
    }


def save_figure(results: list[dict]) -> None:
    lookup = {
        (
            row["learning_rate_label"],
            row["n_estimators_label"],
            row["max_depth_label"],
            row["min_child_weight_label"],
        ): row
        for row in results
    }
    values = np.array([row["accuracy_mean"] for row in results], dtype=float)
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.cm.Blues

    fig, axes = plt.subplots(
        nrows=len(RATE_LABELS),
        ncols=len(TREE_LABELS),
        figsize=(13.2, 11.4),
        constrained_layout=True,
    )

    for row_index, rate in enumerate(RATE_LABELS):
        for col_index, trees in enumerate(TREE_LABELS):
            ax = axes[row_index, col_index]
            grid = np.array(
                [
                    [
                        lookup[(rate, trees, depth, child)]["accuracy_mean"]
                        for child in CHILD_LABELS
                    ]
                    for depth in DEPTH_LABELS
                ],
                dtype=float,
            )
            ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(len(CHILD_LABELS)), CHILD_LABELS)
            ax.set_yticks(range(len(DEPTH_LABELS)), DEPTH_LABELS)
            if row_index == len(RATE_LABELS) - 1:
                ax.set_xlabel("min_child_weight")
            if col_index == 0:
                ax.set_ylabel("max_depth")
            ax.set_title(f"learning_rate: {rate}\nn_estimators: {trees}", fontsize=9)

            for depth_index, depth in enumerate(DEPTH_LABELS):
                for child_index, child in enumerate(CHILD_LABELS):
                    cell = lookup[(rate, trees, depth, child)]
                    score = cell["accuracy_mean"]
                    text_color = "white" if norm(score) > 0.62 else "black"
                    ax.text(
                        child_index,
                        depth_index,
                        f"{score:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )
                    if cell["is_default"]:
                        ax.add_patch(
                            Rectangle(
                                (child_index - 0.5, depth_index - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="#b45309",
                                linewidth=1.8,
                                linestyle="--",
                            )
                        )

    best = max(results, key=lambda row: row["accuracy_mean"])
    best_ax = axes[
        RATE_LABELS.index(best["learning_rate_label"]),
        TREE_LABELS.index(best["n_estimators_label"]),
    ]
    best_ax.add_patch(
        Rectangle(
            (
                CHILD_LABELS.index(best["min_child_weight_label"]) - 0.5,
                DEPTH_LABELS.index(best["max_depth_label"]) - 0.5,
            ),
            1,
            1,
            fill=False,
            edgecolor="#111827",
            linewidth=2.0,
        )
    )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.03,
        pad=0.02,
    )
    colorbar.set_label("5-fold mean accuracy")
    fig.suptitle(
        "XGBoost grid on the training split: 5-fold mean accuracy\n"
        "256 combinations · histogram trees · wine_quality.csv · train rows only · "
        "dashed box is the untuned default · solid box is the highest accuracy",
        fontsize=12,
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
    y_encoded = LabelEncoder().fit_transform(y_train)

    candidates = list(ParameterGrid(GRID))
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate)(params, X_train, y_encoded) for params in candidates
    )
    results = sorted(results, key=lambda row: row["accuracy_mean"], reverse=True)
    best = results[0]
    default = next(row for row in results if row["is_default"])
    default_rank = next(
        rank for rank, row in enumerate(results, start=1) if row["is_default"]
    )
    tied = sum(row["accuracy_mean"] == best["accuracy_mean"] for row in results)

    save_figure(results)

    print(f"Combinations: {len(results)}")
    print(
        "Highest accuracy: "
        f"{best['accuracy_mean']} ± {best['accuracy_std']} "
        f"(learning_rate={best['learning_rate_label']}, "
        f"trees={best['n_estimators']}, "
        f"depth={best['max_depth_label']}, "
        f"min_child_weight={best['min_child_weight_label']})"
    )
    if tied > 1:
        print(f"Combinations tied at that accuracy: {tied}")
    print(
        "Default accuracy: "
        f"{default['accuracy_mean']} ± {default['accuracy_std']} "
        f"(rank {default_rank} of {len(results)})"
    )
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
