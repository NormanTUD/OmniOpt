import os
import importlib.util
from typing import Tuple, List, Dict, Any
from rich.table import Table
from rich.text import Text
import numpy as np


# ----------------- Helpers import -----------------
script_dir = os.path.dirname(os.path.realpath(__file__))
helpers_file: str = f"{script_dir}/.helpers.py"
spec = importlib.util.spec_from_file_location(
    name="helpers",
    location=helpers_file,
)
if spec is not None and spec.loader is not None:
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
else:
    raise ImportError(f"Could not load module from {helpers_file}")

# ----------------- Table utilities -----------------
def pareto_front_table_filter_rows(rows: List[Dict[str, str]], idxs: List[int]) -> List[Dict[str, str]]:
    result = []
    for row in rows:
        try:
            trial_index = int(row["trial_index"])
        except (KeyError, ValueError):
            continue

        if row.get("trial_status", "").strip().upper() == "COMPLETED" and trial_index in idxs:
            result.append(row)
    return result

def pareto_front_table_add_headers(table: Table, param_cols: List[str], result_cols: List[str]) -> None:
    for col in param_cols:
        table.add_column(col, justify="center")
    for col in result_cols:
        table.add_column(Text(f"{col}", style="cyan"), justify="center")

def pareto_front_table_add_rows(table: Table, rows: List[Dict[str, str]], param_cols: List[str], result_cols: List[str]) -> None:
    for row in rows:
        values = [str(helpers.to_int_when_possible(row[col])) for col in param_cols]
        result_values = [Text(str(helpers.to_int_when_possible(row[col])), style="cyan") for col in result_cols]
        table.add_row(*values, *result_values, style="bold green")

# ----------------- Pareto internal utilities -----------------
def _validate_xy(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValueError("Input arrays x and y must have the same shape.")
    if x.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Input arrays must contain only finite numbers (no NaN/Inf).")

def _point_dominates(xi: float, yi: float, xj: float, yj: float, x_minimize: bool, y_minimize: bool) -> bool:
    """Return True if j dominates i (standard weak Pareto: >= in all, > in at least one)."""
    if x_minimize:
        x_better_eq = xj <= xi
        x_strict = xj < xi
    else:
        x_better_eq = xj >= xi
        x_strict = xj > xi

    if y_minimize:
        y_better_eq = yj <= yi
        y_strict = yj < yi
    else:
        y_better_eq = yj >= yi
        y_strict = yj > yi

    return x_better_eq and y_better_eq and (x_strict or y_strict)

def _find_dominated(x: np.ndarray, y: np.ndarray, x_minimize: bool, y_minimize: bool) -> np.ndarray:
    n = len(x)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        xi, yi = x[i], y[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = x[j], y[j]
            if _point_dominates(xi, yi, xj, yj, x_minimize, y_minimize):
                is_dominated[i] = True
                break
    return is_dominated

# ----------------- Public Pareto functions -----------------
def pareto_front_general(
    x: np.ndarray,
    y: np.ndarray,
    x_minimize: bool = True,
    y_minimize: bool = True
) -> np.ndarray:
    try:
        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        _validate_xy(xa, ya)
        is_dominated = _find_dominated(xa, ya, x_minimize, y_minimize)
        return np.where(~is_dominated)[0]
    except Exception as e:
        print("Error in pareto_front_general:", str(e))
        return np.array([], dtype=int)

def pareto_front_filter_complete_points(
    path_to_calculate: str,
    records: Dict[Tuple[int, str], Dict[str, Dict[str, float]]],
    primary_name: str,
    secondary_name: str
) -> List[Tuple[Tuple[int, str], float, float]]:
    points = []
    for key, metrics in records.items():
        means = metrics['means']
        if primary_name in means and secondary_name in means:
            points.append((key, means[primary_name], means[secondary_name]))
    if len(points) == 0:
        raise ValueError(f"No full data points with both objectives found in {path_to_calculate}.")
    return points

def pareto_front_select_pareto_points(
    x: np.ndarray,
    y: np.ndarray,
    x_minimize: bool,
    y_minimize: bool,
    points: List[Tuple[Any, float, float]],
    num_points: int
) -> List[Tuple[Any, float, float]]:
    indices = pareto_front_general(x, y, x_minimize, y_minimize)
    # Sort by x ascending
    sorted_indices = indices[np.argsort(x[indices])]
    selected_points = [points[i] for i in sorted_indices[:num_points]]
    return selected_points
