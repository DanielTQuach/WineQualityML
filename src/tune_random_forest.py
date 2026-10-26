"""Score every combination of the four interacting random-forest settings.

The search stays on the stratified training split. The held-out 20% is not
used. n_estimators, criterion, bootstrap, and random_state stay fixed so the
240 forests differ only in min_samples_leaf, max_depth, max_features, and
class_weight.

Primary score is 5-fold mean accuracy. The figure is written to
reports/random_forest_grid_accuracy.png. No JSON file is written.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"
FIGURE_PATH = ROOT / "reports" / "random_forest_grid_accuracy.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
N_SPLITS = 5

LEAF_VALUES = [1, 2, 5, 10, 20]
DEPTH_VALUES = [None, 8, 12, 20]
FEATURE_VALUES = [2, "sqrt", 5, None]
WEIGHT_VALUES = [None, "balanced", "balanced_subsample"]

GRID = {
    "min_samples_leaf": LEAF_VALUES,
    "max_depth": DEPTH_VALUES,
    "max_features": FEATURE_VALUES,
    "class_weight": WEIGHT_VALUES,
}

DEPTH_LABELS = ["unlimited", "8", "12", "20"]
FEATURE_LABELS = ["2", "sqrt", "5", "all 11"]
WEIGHT_LABELS = ["none", "balanced", "balanced subsample"]


def round_metric(value: float) -> float:
    return round(float(value), 4)


def depth_label(value: int | None) -> str:
    return "unlimited" if value is None else str(value)


def feature_label(value: int | str | None) -> str:
    if value is None:
        return "all 11"
    return str(value)


def weight_label(value: str | None) -> str:
    if value is None:
        return "none"
    if value == "balanced_subsample":
        return "balanced subsample"
    return value


def is_default(params: dict) -> bool:
    return (
        params["min_samples_leaf"] == 1
        and params["max_depth"] is None
        and params["max_features"] == "sqrt"
        and params["class_weight"] is None
    )


def evaluate(params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        criterion="gini",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
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
        "min_samples_leaf": params["min_samples_leaf"],
        "max_depth": params["max_depth"],
        "max_features": params["max_features"],
        "class_weight": params["class_weight"],
        "max_depth_label": depth_label(params["max_depth"]),
        "max_features_label": feature_label(params["max_features"]),
        "class_weight_label": weight_label(params["class_weight"]),
        "is_default": is_default(params),
        "accuracy_mean": round_metric(scores["test_score"].mean()),
        "accuracy_std": round_metric(scores["test_score"].std()),
    }


def save_figure(results: list[dict]) -> None:
    lookup = {
        (
            row["class_weight_label"],
            row["max_features_label"],
            row["max_depth_label"],
            row["min_samples_leaf"],
        ): row
        for row in results
    }
    values = np.array([row["accuracy_mean"] for row in results], dtype=float)
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.cm.Blues

    fig, axes = plt.subplots(
        nrows=len(WEIGHT_LABELS),
        ncols=len(FEATURE_LABELS),
        figsize=(12.5, 8.2),
        constrained_layout=True,
    )

    for row_index, weight in enumerate(WEIGHT_LABELS):
        for col_index, features in enumerate(FEATURE_LABELS):
            ax = axes[row_index, col_index]
            grid = np.array(
                [
                    [
                        lookup[(weight, features, depth, leaf)]["accuracy_mean"]
                        for leaf in LEAF_VALUES
                    ]
                    for depth in DEPTH_LABELS
                ],
                dtype=float,
            )
            ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(len(LEAF_VALUES)), [str(leaf) for leaf in LEAF_VALUES])
            ax.set_yticks(range(len(DEPTH_LABELS)), DEPTH_LABELS)
            if row_index == len(WEIGHT_LABELS) - 1:
                ax.set_xlabel("min_samples_leaf")
            if col_index == 0:
                ax.set_ylabel("max_depth")
            ax.set_title(f"class_weight: {weight}\nmax_features: {features}", fontsize=9)

            for depth_index, depth in enumerate(DEPTH_LABELS):
                for leaf_index, leaf in enumerate(LEAF_VALUES):
                    cell = lookup[(weight, features, depth, leaf)]
                    score = cell["accuracy_mean"]
                    text_color = "white" if norm(score) > 0.62 else "black"
                    ax.text(
                        leaf_index,
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
                                (leaf_index - 0.5, depth_index - 0.5),
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
        WEIGHT_LABELS.index(best["class_weight_label"]),
        FEATURE_LABELS.index(best["max_features_label"]),
    ]
    best_ax.add_patch(
        Rectangle(
            (
                LEAF_VALUES.index(best["min_samples_leaf"]) - 0.5,
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
        "Random forest grid on the training split: 5-fold mean accuracy\n"
        "240 combinations · 100 trees · wine_quality.csv · train rows only · "
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

    candidates = list(ParameterGrid(GRID))
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate)(params, X_train, y_train) for params in candidates
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
        f"(leaf={best['min_samples_leaf']}, depth={best['max_depth_label']}, "
        f"features={best['max_features_label']}, weight={best['class_weight_label']})"
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
