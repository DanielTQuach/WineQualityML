"""Tune an SVM classifier for wine quality and plot every compared setting.

The target is the five-class quality column. Rows are split 80/20 with
stratification, matching the other models. Every comparison uses 5-fold
stratified cross-validation on the training split only. The held-out 20%
is scored once, after the winning setting is chosen. The winner is the
converged setting with the highest mean training accuracy.

Two kinds of hyperparameters are separated on purpose.

Standalone knobs apply for every kernel and do not turn other knobs on or
off. They are class_weight (None vs balanced) and decision_function_shape
(ovr vs ovo). Each is scored by changing only that knob, first on the
default RBF model (C=1, gamma=scale) and then again at the best joint setting.

Kernel-conditional knobs are scored together, because the useful ones depend
on the kernel and C trades off against gamma:

- linear: C only (gamma, degree, and coef0 are ignored)
- rbf: C x gamma
- poly: C x gamma x degree x coef0

gamma="scale" on standardized features is 1 / n_features. With 11 inputs that
value is 1/11, so the grids use that number directly. Features are standardized
inside the pipeline, and each fold fits its own scaler. The estimator is always
sklearn.svm.SVC, including the linear kernel, so C stays on the same scale.

Polynomial gamma=1 is omitted. On standardized features that kernel overflows
the solver and was still running after several minutes. The polynomial grid
uses gamma=0.3 instead, which sits between scale and 1. Every fit has an
iteration cap so a hard cell cannot block the search. A cell that hits the
cap is plotted with a star and cannot be selected.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
import threading
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.patches import Rectangle
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "wine_quality.csv"
MODEL_PATH = ROOT / "models" / "svm_tuned.joblib"
RESULTS_PATH = ROOT / "reports" / "svm_tuning.json"
FIGURE_PATH = ROOT / "reports" / "svm_tuning.png"
CACHE_PATH = ROOT / "reports" / "svm_cv_cache.jsonl"

LABELS = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
N_FEATURES = 11
# sklearn's gamma="scale" after StandardScaler, whose columns have variance 1.
GAMMA_SCALE = 1 / N_FEATURES

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMA_VALUES = [0.001, 0.01, 0.03, GAMMA_SCALE, 0.1, 0.3, 1.0, 10.0]
POLY_C_VALUES = [0.1, 1.0, 10.0, 100.0]
POLY_GAMMA_VALUES = [0.01, GAMMA_SCALE, 0.3]
MAX_ITER = 500_000
POLY_DEGREES = [2, 3]
POLY_COEF0_VALUES = [0.0, 1.0]

METRIC_COLORS = {"accuracy": "#4C78A8"}
METRIC_LABELS = {"accuracy": "Accuracy"}

DATA: dict = {}
PROGRESS = {"done": 0, "total": 0}
PROGRESS_LOCK = threading.Lock()


def round_metric(value: float) -> float:
    return round(float(value), 4)


def values_match(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    return left == right


def param_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def blank_params() -> dict:
    return {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": GAMMA_SCALE,
        "degree": None,
        "coef0": None,
        "class_weight": None,
        "decision_function_shape": "ovr",
    }


def with_params(**overrides: object) -> dict:
    params = blank_params()
    params.update(overrides)
    return params


def svc_kwargs(params: dict) -> dict:
    kwargs = {
        "kernel": params["kernel"],
        "C": params["C"],
        "class_weight": params["class_weight"],
        "decision_function_shape": params["decision_function_shape"],
        "cache_size": 500,
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
    }
    if params["kernel"] in {"rbf", "poly"}:
        kwargs["gamma"] = params["gamma"]
    if params["kernel"] == "poly":
        kwargs["degree"] = params["degree"]
        kwargs["coef0"] = params["coef0"]
    return kwargs


def make_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(**svc_kwargs(params))),
        ]
    )


def describe(params: dict) -> str:
    parts = [str(params["kernel"]), f"C={params['C']:g}"]
    gamma = params["gamma"]
    if gamma is not None:
        if values_match(gamma, GAMMA_SCALE):
            parts.append("gamma=scale")
        else:
            parts.append(f"gamma={gamma:g}")
    if params["kernel"] == "poly":
        parts.append(f"degree={params['degree']}")
        parts.append(f"coef0={params['coef0']:g}")
    if params["class_weight"] is not None:
        parts.append(f"class_weight={params['class_weight']}")
    if params["decision_function_shape"] != "ovr":
        parts.append(str(params["decision_function_shape"]))
    return ", ".join(parts)


def rank_key(row: dict) -> tuple:
    return (
        row["accuracy_mean"],
        row["balanced_accuracy_mean"],
        row["f1_macro_mean"],
        -row["accuracy_std"],
    )


def add_task(tasks: dict, params: dict, stage: str) -> None:
    key = param_key(params)
    if key in tasks:
        if stage not in tasks[key]["stages"]:
            tasks[key]["stages"].append(stage)
        return
    tasks[key] = {"params": params, "stages": [stage]}


def build_joint_tasks() -> dict:
    tasks: dict = {}
    add_task(tasks, with_params(), "independent")
    add_task(tasks, with_params(class_weight="balanced"), "independent")
    add_task(tasks, with_params(decision_function_shape="ovo"), "independent")

    for c_value in C_VALUES:
        add_task(
            tasks,
            with_params(kernel="linear", C=c_value, gamma=None),
            "joint_linear",
        )
        for gamma in GAMMA_VALUES:
            add_task(
                tasks,
                with_params(kernel="rbf", C=c_value, gamma=gamma),
                "joint_rbf",
            )

    for c_value in POLY_C_VALUES:
        for gamma in POLY_GAMMA_VALUES:
            for degree in POLY_DEGREES:
                for coef0 in POLY_COEF0_VALUES:
                    add_task(
                        tasks,
                        with_params(
                            kernel="poly",
                            C=c_value,
                            gamma=gamma,
                            degree=degree,
                            coef0=coef0,
                        ),
                        "joint_poly",
                    )
    return tasks


def evaluate(task: dict) -> dict:
    params = task["params"]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            labels=LABELS,
            zero_division=0,
        ),
    }
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            scores = cross_validate(
                make_pipeline(params),
                DATA["X"],
                DATA["y"],
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )
    row = {
        "params": params,
        "stages": list(task["stages"]),
        "iteration_capped": any(issubclass(item.category, ConvergenceWarning) for item in caught),
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(scores["test_balanced_accuracy"].std()),
        "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "f1_macro_std": float(scores["test_f1_macro"].std()),
    }
    with PROGRESS_LOCK:
        PROGRESS["done"] += 1
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": param_key(params), "row": row}) + "\n")
        cap_note = "  iteration cap" if row["iteration_capped"] else ""
        print(
            f"[{PROGRESS['done']}/{PROGRESS['total']}] "
            f"{describe(params)}  macro F1 {row['f1_macro_mean']:.4f}{cap_note}",
            flush=True,
        )
    return row


def load_cache() -> dict[str, dict]:
    cached = {}
    if not CACHE_PATH.exists():
        return cached
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            cached[item["key"]] = item["row"]
    return cached


def run_tasks(tasks: dict) -> list[dict]:
    cached = load_cache()
    pending = []
    ready = []
    for key, task in tasks.items():
        if key in cached:
            row = cached[key]
            row["stages"] = list(task["stages"])
            ready.append(row)
        else:
            pending.append(task)
    if ready:
        print(f"Reused {len(ready)} cached settings", flush=True)
    if not pending:
        return ready
    PROGRESS["done"] = 0
    PROGRESS["total"] = len(pending)
    workers = max(1, min(8, os.cpu_count() or 1))
    print(f"Scoring {len(pending)} settings on {workers} workers", flush=True)
    scored = Parallel(n_jobs=workers, backend="threading")(
        delayed(evaluate)(task) for task in pending
    )
    return ready + scored


def find_row(results: list[dict], **expected: object) -> dict:
    matches = []
    for row in results:
        params = row["params"]
        if all(values_match(params[key], value) for key, value in expected.items()):
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {expected}, found {len(matches)}")
    return matches[0]


def is_joint(row: dict) -> bool:
    return any(stage.startswith("joint_") for stage in row["stages"])


def score_matrix(
    results: list[dict],
    c_values: list[float],
    gamma_values: list[float],
    metric: str,
    **filters: object,
) -> np.ndarray:
    matrix = np.zeros((len(gamma_values), len(c_values)))
    for row_index, gamma in enumerate(gamma_values):
        for column_index, c_value in enumerate(c_values):
            row = find_row(results, C=c_value, gamma=gamma, **filters)
            matrix[row_index, column_index] = row[metric]
    return matrix


def cap_matrix(
    results: list[dict],
    c_values: list[float],
    gamma_values: list[float],
    **filters: object,
) -> np.ndarray:
    matrix = np.zeros((len(gamma_values), len(c_values)), dtype=bool)
    for row_index, gamma in enumerate(gamma_values):
        for column_index, c_value in enumerate(c_values):
            row = find_row(results, C=c_value, gamma=gamma, **filters)
            matrix[row_index, column_index] = bool(row["iteration_capped"])
    return matrix


def format_gamma(value: float) -> str:
    if values_match(value, GAMMA_SCALE):
        return "1/11"
    return f"{value:g}"


def annotation_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "black"
    position = (value - vmin) / (vmax - vmin)
    return "black" if position >= 0.62 else "white"


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    c_values: list[float],
    gamma_values: list[float],
    vmin: float,
    vmax: float,
    title: str,
    show_ylabel: bool,
    capped: np.ndarray | None = None,
) -> plt.cm.ScalarMappable:
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels([f"{value:g}" for value in c_values], fontsize=8)
    ax.set_yticks(range(len(gamma_values)))
    ax.set_yticklabels([format_gamma(value) for value in gamma_values], fontsize=8)
    ax.set_xlabel("C")
    if show_ylabel:
        ax.set_ylabel("gamma")
    ax.set_title(title, fontsize=10)
    eligible = np.where(capped, -np.inf, matrix) if capped is not None else matrix
    best_value = float(np.max(eligible)) if np.isfinite(eligible).any() else float("-inf")
    font_size = 6 if matrix.shape[1] >= 6 else 7
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            star = "*" if capped is not None and bool(capped[row_index, column_index]) else ""
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}{star}",
                ha="center",
                va="center",
                fontsize=font_size,
                color=annotation_color(float(value), vmin, vmax),
            )
    if np.isfinite(best_value):
        tie_rows, tie_columns = np.where(np.isfinite(eligible) & np.isclose(eligible, best_value, atol=1e-4))
        for best_row, best_column in zip(tie_rows, tie_columns):
            ax.add_patch(
                Rectangle(
                    (best_column - 0.5, best_row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#E45756",
                    linewidth=2,
                )
            )
    return image


def style_comparison_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("5-fold CV mean")


def plot_results(
    results: list[dict],
    _best_joint: dict,
    selected: dict,
    holdout: dict,
) -> None:
    joint_filter = {"class_weight": None, "decision_function_shape": "ovr"}
    rbf_f1 = score_matrix(
        results,
        C_VALUES,
        GAMMA_VALUES,
        "f1_macro_mean",
        kernel="rbf",
        degree=None,
        coef0=None,
        **joint_filter,
    )
    rbf_balanced = score_matrix(
        results,
        C_VALUES,
        GAMMA_VALUES,
        "balanced_accuracy_mean",
        kernel="rbf",
        degree=None,
        coef0=None,
        **joint_filter,
    )
    rbf_accuracy = score_matrix(
        results,
        C_VALUES,
        GAMMA_VALUES,
        "accuracy_mean",
        kernel="rbf",
        degree=None,
        coef0=None,
        **joint_filter,
    )
    rbf_capped = cap_matrix(
        results,
        C_VALUES,
        GAMMA_VALUES,
        kernel="rbf",
        degree=None,
        coef0=None,
        **joint_filter,
    )
    poly_matrices = []
    for degree in POLY_DEGREES:
        for coef0 in POLY_COEF0_VALUES:
            poly_filters = {
                "kernel": "poly",
                "degree": degree,
                "coef0": coef0,
                **joint_filter,
            }
            poly_matrices.append(
                (
                    degree,
                    coef0,
                    score_matrix(
                        results,
                        POLY_C_VALUES,
                        POLY_GAMMA_VALUES,
                        "accuracy_mean",
                        **poly_filters,
                    ),
                    cap_matrix(results, POLY_C_VALUES, POLY_GAMMA_VALUES, **poly_filters),
                )
            )
    accuracy_values = [rbf_accuracy, *[matrix for _, _, matrix, _ in poly_matrices]]
    accuracy_min = min(float(matrix.min()) for matrix in accuracy_values)
    accuracy_max = max(float(matrix.max()) for matrix in accuracy_values)

    fig = plt.figure(figsize=(18, 12.5), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.35, 1.05, 1.15])

    ax_f1 = fig.add_subplot(grid[0, 0:2])
    ax_balanced = fig.add_subplot(grid[0, 2])
    ax_accuracy = fig.add_subplot(grid[0, 3])
    accuracy_image = draw_heatmap(
        ax_f1,
        rbf_accuracy,
        C_VALUES,
        GAMMA_VALUES,
        accuracy_min,
        accuracy_max,
        "RBF joint grid — accuracy",
        show_ylabel=True,
        capped=rbf_capped,
    )
    balanced_image = draw_heatmap(
        ax_balanced,
        rbf_balanced,
        C_VALUES,
        GAMMA_VALUES,
        float(rbf_balanced.min()),
        float(rbf_balanced.max()),
        "RBF — balanced accuracy",
        show_ylabel=False,
        capped=rbf_capped,
    )
    f1_image = draw_heatmap(
        ax_accuracy,
        rbf_f1,
        C_VALUES,
        GAMMA_VALUES,
        float(rbf_f1.min()),
        float(rbf_f1.max()),
        "RBF — macro F1",
        show_ylabel=False,
        capped=rbf_capped,
    )
    fig.colorbar(accuracy_image, ax=ax_f1, fraction=0.046, pad=0.04, label="Mean CV accuracy")
    fig.colorbar(balanced_image, ax=ax_balanced, fraction=0.046, pad=0.04)
    fig.colorbar(f1_image, ax=ax_accuracy, fraction=0.046, pad=0.04)

    ax_linear = fig.add_subplot(grid[1, 0:4])
    linear_rows = [
        find_row(
            results,
            kernel="linear",
            C=c_value,
            gamma=None,
            degree=None,
            coef0=None,
            **joint_filter,
        )
        for c_value in C_VALUES
    ]
    linear_accuracy = [row["accuracy_mean"] for row in linear_rows]
    linear_accuracy_std = [row["accuracy_std"] for row in linear_rows]
    ax_linear.errorbar(
        C_VALUES,
        linear_accuracy,
        yerr=linear_accuracy_std,
        marker="o",
        capsize=3,
        color=METRIC_COLORS["accuracy"],
        label=METRIC_LABELS["accuracy"],
    )
    best_linear = max((row for row in linear_rows if not row["iteration_capped"]), key=rank_key)
    ax_linear.axvline(best_linear["params"]["C"], color="#E45756", linestyle="--", linewidth=1)
    capped_c = [row["params"]["C"] for row in linear_rows if row["iteration_capped"]]
    if capped_c:
        capped_accuracy = [
            row["accuracy_mean"] for row in linear_rows if row["iteration_capped"]
        ]
        ax_linear.scatter(
            capped_c,
            capped_accuracy,
            marker="*",
            s=90,
            color="black",
            zorder=4,
            label="Hit iteration cap",
        )
    ax_linear.set_xscale("log")
    ax_linear.set_xticks(C_VALUES)
    ax_linear.set_xticklabels([f"{value:g}" for value in C_VALUES])
    ax_linear.set_xlabel("C")
    ax_linear.set_title("Linear kernel — C only (gamma is unused)", fontsize=10)
    ax_linear.legend(frameon=False, fontsize=8)
    style_comparison_axis(ax_linear)
    ax_linear.set_ylabel("5-fold CV accuracy")

    poly_axes = []
    for panel_index, (degree, coef0, matrix, capped) in enumerate(poly_matrices):
        ax_poly = fig.add_subplot(grid[2, panel_index])
        poly_axes.append(ax_poly)
        draw_heatmap(
            ax_poly,
            matrix,
            POLY_C_VALUES,
            POLY_GAMMA_VALUES,
            accuracy_min,
            accuracy_max,
            f"Polynomial degree {degree}, coef0 {coef0:g}",
            show_ylabel=panel_index == 0,
            capped=capped,
        )
    fig.colorbar(accuracy_image, ax=poly_axes, fraction=0.046, pad=0.04, label="Mean CV accuracy")

    fig.suptitle("SVM hyperparameter search for wine quality", fontsize=16)
    tied = [
        row
        for row in results
        if not row["iteration_capped"]
        and math.isclose(row["accuracy_mean"], selected["accuracy_mean"], abs_tol=1e-4)
    ]
    footer = (
        "Heatmap numbers on the large RBF panel and the polynomial panels are mean CV accuracy. "
        "Outlines mark every converged cell tied for the best accuracy in that panel. "
        "A star means the solver hit the iteration cap. "
        "Polynomial panels share the RBF accuracy color scale. "
        "1/11 is gamma='scale' after standardization. "
        "The dashed line marks the best converged linear C by accuracy. "
        f"Highest training CV accuracy is {selected['accuracy_mean']:.3f}, "
        f"tied across {len(tied)} converged settings, including {describe(selected['params'])}. "
        f"Holdout for that setting, scored once: accuracy {holdout['accuracy']:.3f}, "
        f"balanced accuracy {holdout['balanced_accuracy']:.3f}, "
        f"macro F1 {holdout['f1_macro']:.3f}."
    )
    fig.supxlabel("\n".join(textwrap.wrap(footer, width=170)), fontsize=8)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def boundary_notes(best_joint: dict) -> list[str]:
    params = best_joint["params"]
    notes = []
    kernel = params["kernel"]
    if kernel == "linear":
        grid = C_VALUES
        if values_match(params["C"], grid[0]) or values_match(params["C"], grid[-1]):
            notes.append(f"linear C={params['C']:g} is on the edge of {grid}")
    elif kernel == "rbf":
        if values_match(params["C"], C_VALUES[0]) or values_match(params["C"], C_VALUES[-1]):
            notes.append(f"rbf C={params['C']:g} is on the edge of {C_VALUES}")
        if values_match(params["gamma"], GAMMA_VALUES[0]) or values_match(params["gamma"], GAMMA_VALUES[-1]):
            notes.append(f"rbf gamma={params['gamma']:g} is on the edge of the gamma grid")
    elif kernel == "poly":
        if values_match(params["C"], POLY_C_VALUES[0]) or values_match(params["C"], POLY_C_VALUES[-1]):
            notes.append(f"poly C={params['C']:g} is on the edge of {POLY_C_VALUES}")
        if values_match(params["gamma"], POLY_GAMMA_VALUES[0]) or values_match(
            params["gamma"], POLY_GAMMA_VALUES[-1]
        ):
            notes.append("poly gamma is on the edge of the polynomial gamma grid")
    return notes


def public_params(params: dict) -> dict:
    gamma = params["gamma"]
    return {
        "kernel": params["kernel"],
        "C": params["C"],
        "gamma": None if gamma is None else float(gamma),
        "gamma_is_scale": gamma is not None and values_match(gamma, GAMMA_SCALE),
        "degree": params["degree"],
        "coef0": params["coef0"],
        "class_weight": params["class_weight"],
        "decision_function_shape": params["decision_function_shape"],
    }


def public_row(row: dict) -> dict:
    return {
        "stages": row["stages"],
        "params": public_params(row["params"]),
        "cv": {
            "accuracy_mean": round_metric(row["accuracy_mean"]),
            "accuracy_std": round_metric(row["accuracy_std"]),
            "balanced_accuracy_mean": round_metric(row["balanced_accuracy_mean"]),
            "balanced_accuracy_std": round_metric(row["balanced_accuracy_std"]),
            "f1_macro_mean": round_metric(row["f1_macro_mean"]),
            "f1_macro_std": round_metric(row["f1_macro_std"]),
        },
        "label": describe(row["params"]),
        "iteration_capped": bool(row["iteration_capped"]),
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
    DATA["X"] = X_train.to_numpy()
    DATA["y"] = y_train.to_numpy()

    results = run_tasks(build_joint_tasks())
    joint_rows = [row for row in results if is_joint(row) and not row["iteration_capped"]]
    if not joint_rows:
        raise RuntimeError("Every joint setting hit the iteration cap")
    best_joint = max(joint_rows, key=rank_key)
    print(f"Best joint setting: {describe(best_joint['params'])}", flush=True)
    for note in boundary_notes(best_joint):
        print(f"Boundary: {note}", flush=True)

    confirm_tasks: dict = {}
    for class_weight in (None, "balanced"):
        for decision_shape in ("ovr", "ovo"):
            params = dict(best_joint["params"])
            params["class_weight"] = class_weight
            params["decision_function_shape"] = decision_shape
            key = param_key(params)
            existing = next((row for row in results if param_key(row["params"]) == key), None)
            if existing is None:
                add_task(confirm_tasks, params, "confirm_at_best_joint")
            elif "confirm_at_best_joint" not in existing["stages"]:
                existing["stages"].append("confirm_at_best_joint")
    if confirm_tasks:
        results.extend(run_tasks(confirm_tasks))

    eligible = [row for row in results if not row["iteration_capped"]]
    selected = max(eligible, key=rank_key)
    print(f"Selected: {describe(selected['params'])}", flush=True)

    model = make_pipeline(selected["params"])
    model.fit(X_train, y_train)
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
    holdout = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
    }
    plot_results(results, best_joint, selected, holdout)

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=make_scorer(f1_score, average="macro", labels=LABELS, zero_division=0),
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier: SVC = model.named_steps["classifier"]
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
        key=lambda item: item["permutation_importance_mean"],
        reverse=True,
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

    payload = {
        "model": "SVC",
        "estimator": "Pipeline(StandardScaler, SVC)",
        "source": DATA_PATH.name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": LABELS,
        "selection_metric": "mean stratified 5-fold accuracy on the training split",
        "search": {
            "standalone_at_default_rbf": {
                "class_weight": [None, "balanced"],
                "decision_function_shape": ["ovr", "ovo"],
                "held_fixed": {
                    "kernel": "rbf",
                    "C": 1.0,
                    "gamma": GAMMA_SCALE,
                    "gamma_note": "1/11, equal to gamma='scale' after StandardScaler",
                },
            },
            "joint": {
                "linear": {"C": C_VALUES},
                "rbf": {"C": C_VALUES, "gamma": GAMMA_VALUES},
                "poly": {
                    "C": POLY_C_VALUES,
                    "gamma": POLY_GAMMA_VALUES,
                    "degree": POLY_DEGREES,
                    "coef0": POLY_COEF0_VALUES,
                },
            },
            "confirmation": (
                "class_weight x decision_function_shape at the best joint setting, "
                "so the final combination is one that was actually scored"
            ),
            "fixed": {
                "tol": 0.001,
                "shrinking": True,
                "probability": False,
                "max_iter": MAX_ITER,
            },
            "boundary_notes": boundary_notes(best_joint),
        },
        "results": [public_row(row) for row in sorted(results, key=rank_key, reverse=True)],
        "best_joint_setting": public_row(best_joint),
        "selected_model": {
            "params": public_params(selected["params"]),
            "label": describe(selected["params"]),
            "n_support_vectors": int(classifier.n_support_.sum()),
            "support_vectors_per_class": {
                label: int(count)
                for label, count in (
                    dict(zip(classifier.classes_, classifier.n_support_)).items()
                )
                if label in LABELS
            },
            "cross_validation_on_train": public_row(selected)["cv"] | {"folds": N_FOLDS},
            "majority_class_baseline": {
                "class": str(majority),
                "test_accuracy": round_metric(accuracy_score(y_test, y_baseline)),
                "test_balanced_accuracy": round_metric(
                    balanced_accuracy_score(y_test, y_baseline)
                ),
                "test_f1_macro": round_metric(
                    f1_score(
                        y_test,
                        y_baseline,
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
            },
            "test": {
                "accuracy": round_metric(holdout["accuracy"]),
                "balanced_accuracy": round_metric(holdout["balanced_accuracy"]),
                "f1_macro": round_metric(holdout["f1_macro"]),
                "f1_weighted": round_metric(holdout["f1_weighted"]),
                "per_class": per_class,
                "confusion_matrix": {
                    "rows": "true label",
                    "columns": "predicted label",
                    "labels": LABELS,
                    "values": matrix.tolist(),
                },
            },
            "feature_importance": importances,
        },
        "train_class_counts": {label: int((y_train == label).sum()) for label in LABELS},
        "test_class_counts": {label: int((y_test == label).sum()) for label in LABELS},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Train rows: {payload['n_train']}  Test rows: {payload['n_test']}")
    print(f"Selected: {describe(selected['params'])}")
    print(
        "5-fold train accuracy: "
        f"{payload['selected_model']['cross_validation_on_train']['accuracy_mean']} ± "
        f"{payload['selected_model']['cross_validation_on_train']['accuracy_std']}"
    )
    print(f"Test accuracy: {payload['selected_model']['test']['accuracy']}")
    print(f"Test balanced accuracy: {payload['selected_model']['test']['balanced_accuracy']}")
    print(f"Test macro F1: {payload['selected_model']['test']['f1_macro']}")
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
