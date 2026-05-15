#!/usr/bin/env python3
"""
Fit Cd and Pb breakthrough-curve CSV data with a 1-D ADE finite-difference model.

Usage:
    python btc_fit_ade.py --cd Cd.csv --pb Pb.csv --output btc_fit_results.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erfc
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class BTCData:
    name: str
    pv: np.ndarray
    conc: np.ndarray


@dataclass(frozen=True)
class FitResult:
    name: str
    pv: np.ndarray
    conc_exp: np.ndarray
    conc_fit: np.ndarray
    k: float
    dispersion: float
    c0: float
    r2: float
    success: bool
    message: str


def validate_csv_path(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    return file_path


def load_btc_csv(path: str | Path, name: str) -> BTCData:
    file_path = validate_csv_path(path)
    data = pd.read_csv(file_path)

    if data.shape[1] < 2:
        raise ValueError(f"{file_path} must contain at least two columns: Pv and Conc")

    columns_lower = {str(col).strip().lower(): col for col in data.columns}
    pv_col = columns_lower.get("pv", data.columns[0])
    conc_col = columns_lower.get("conc", data.columns[1])

    numeric = pd.DataFrame(
        {
            "Pv": pd.to_numeric(data[pv_col], errors="coerce"),
            "Conc": pd.to_numeric(data[conc_col], errors="coerce"),
        }
    ).dropna()

    if numeric.empty:
        raise ValueError(f"{file_path} contains no valid numeric Pv/Conc rows")

    numeric = numeric.sort_values("Pv")
    pv = numeric["Pv"].to_numpy(dtype=float)
    conc = numeric["Conc"].to_numpy(dtype=float)

    if np.any(np.diff(pv) <= 0):
        raise ValueError(f"{file_path} contains duplicate or non-increasing pore-volume values")
    if np.any(pv < 0):
        raise ValueError(f"{file_path} contains negative pore-volume values")

    return BTCData(name=name, pv=pv, conc=conc)


def ogata_banks_step(x: np.ndarray | float, t: np.ndarray | float, velocity: float, dispersion: float, c0: float) -> np.ndarray:
    """Ogata-Banks semi-infinite, step-input ADE solution without kinetic loss."""
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    t_safe = np.maximum(t_arr, np.finfo(float).eps)
    d_safe = max(float(dispersion), np.finfo(float).eps)
    first = erfc((x_arr - velocity * t_safe) / (2.0 * np.sqrt(d_safe * t_safe)))
    second = np.exp(np.clip(velocity * x_arr / d_safe, -700.0, 700.0)) * erfc(
        (x_arr + velocity * t_safe) / (2.0 * np.sqrt(d_safe * t_safe))
    )
    return 0.5 * c0 * (first + second)


def default_sink_factor(x: np.ndarray, t: float) -> np.ndarray:
    """Dimensionless multiplier f(x,t) for first-order loss; replace if needed."""
    return np.ones_like(x, dtype=float)


def solve_ade_finite_difference(
    times: np.ndarray,
    k: float,
    dispersion: float,
    c0: float,
    *,
    length: float = 1.0,
    velocity: float = 1.0,
    nx: int = 151,
    dt: float | None = None,
    sink_factor: Callable[[np.ndarray, float], np.ndarray] = default_sink_factor,
) -> np.ndarray:
    """Backward-Euler finite-difference solver for 1-D ADE with first-order kinetic loss."""
    if np.any(times < 0):
        raise ValueError("Simulation times must be non-negative")
    if k < 0 or dispersion <= 0 or c0 < 0:
        raise ValueError("Model parameters must satisfy K >= 0, D > 0, and C0 >= 0")

    times = np.asarray(times, dtype=float)
    order = np.argsort(times)
    sorted_times = times[order]
    max_time = float(sorted_times[-1]) if sorted_times.size else 0.0
    if max_time == 0.0:
        return np.zeros_like(times, dtype=float)

    nx = max(int(nx), 21)
    x = np.linspace(0.0, length, nx)
    dx = length / (nx - 1)
    if dt is None:
        dt = min(0.0025, max_time / 1500.0) if max_time > 0 else 0.0025
    dt = max(float(dt), np.finfo(float).eps)

    alpha = dispersion / dx**2
    beta = velocity / dx
    concentration = np.zeros(nx, dtype=float)
    result_sorted = np.zeros_like(sorted_times, dtype=float)

    current_time = 0.0
    previous_outlet = 0.0
    target_index = 0

    while target_index < len(sorted_times) and np.isclose(sorted_times[target_index], 0.0):
        result_sorted[target_index] = previous_outlet
        target_index += 1

    while current_time < max_time - 1e-15:
        step = min(dt, max_time - current_time)
        next_time = current_time + step
        f_values = np.asarray(sink_factor(x, next_time), dtype=float)
        if f_values.shape != x.shape:
            raise ValueError("sink_factor must return an array with the same shape as x")

        n_unknown = nx - 2
        matrix = np.zeros((n_unknown, n_unknown), dtype=float)
        rhs = concentration[1:-1].copy()

        lower_value = -step * (alpha + beta)
        diagonal = 1.0 + step * (2.0 * alpha + beta + k * f_values[1:-1])
        upper_value = -step * alpha

        for j in range(n_unknown):
            matrix[j, j] = diagonal[j]
            if j > 0:
                matrix[j, j - 1] = lower_value
            if j < n_unknown - 1:
                matrix[j, j + 1] = upper_value

        inlet = ogata_banks_step(0.0, next_time, velocity, dispersion, c0)
        rhs[0] -= lower_value * inlet
        interior = np.linalg.solve(matrix, rhs)

        new_concentration = np.empty_like(concentration)
        new_concentration[0] = inlet
        new_concentration[1:-1] = np.maximum(interior, 0.0)
        new_concentration[-1] = new_concentration[-2]

        new_outlet = float(new_concentration[-1])
        while target_index < len(sorted_times) and sorted_times[target_index] <= next_time + 1e-15:
            target = sorted_times[target_index]
            fraction = 0.0 if next_time == current_time else (target - current_time) / (next_time - current_time)
            result_sorted[target_index] = previous_outlet + fraction * (new_outlet - previous_outlet)
            target_index += 1

        concentration = new_concentration
        current_time = next_time
        previous_outlet = new_outlet

    result = np.empty_like(result_sorted)
    result[order] = result_sorted
    return result


def fit_parameters(data: BTCData, *, fit_dispersion: bool = True) -> FitResult:
    pv = data.pv.astype(float)
    conc = data.conc.astype(float)

    max_conc = max(float(np.nanmax(conc)), np.finfo(float).eps)
    d0 = 0.02
    k0 = 0.05
    c00 = max_conc

    if fit_dispersion:
        initial = np.array([k0, d0, c00], dtype=float)
        lower = np.array([0.0, 1e-5, 0.0], dtype=float)
        upper = np.array([10.0, 2.0, max_conc * 10.0], dtype=float)
    else:
        initial = np.array([k0, c00], dtype=float)
        lower = np.array([0.0, 0.0], dtype=float)
        upper = np.array([10.0, max_conc * 10.0], dtype=float)

    def unpack(params: np.ndarray) -> tuple[float, float, float]:
        if fit_dispersion:
            return float(params[0]), float(params[1]), float(params[2])
        return float(params[0]), d0, float(params[1])

    def residuals(params: np.ndarray) -> np.ndarray:
        k, dispersion, c0 = unpack(params)
        predicted = solve_ade_finite_difference(pv, k, dispersion, c0)
        return predicted - conc

    optimization = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(np.std(conc), max_conc * 0.05, np.finfo(float).eps),
        max_nfev=300,
    )

    k_fit, d_fit, c0_fit = unpack(optimization.x)
    conc_fit = solve_ade_finite_difference(pv, k_fit, d_fit, c0_fit)
    r2 = float(r2_score(conc, conc_fit)) if len(conc) > 1 else float("nan")

    return FitResult(
        name=data.name,
        pv=pv,
        conc_exp=conc,
        conc_fit=conc_fit,
        k=k_fit,
        dispersion=d_fit,
        c0=c0_fit,
        r2=r2,
        success=bool(optimization.success),
        message=str(optimization.message),
    )


def plot_results(cd_result: FitResult, pb_result: FitResult, output_path: str | Path = "btc_fit_results.png") -> None:
    output = Path(output_path).expanduser().resolve()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    for result, color in ((cd_result, "tab:blue"), (pb_result, "tab:red")):
        pv_smooth = np.linspace(float(np.min(result.pv)), float(np.max(result.pv)), 250)
        conc_smooth = solve_ade_finite_difference(pv_smooth, result.k, result.dispersion, result.c0)
        ax.scatter(result.pv, result.conc_exp, s=35, color=color, alpha=0.75, label=f"{result.name} experimental")
        ax.plot(
            pv_smooth,
            conc_smooth,
            color=color,
            linewidth=2.0,
            label=f"{result.name} ADE fit (R²={result.r2:.3f})",
        )

    ax.set_xlabel("Pore volume (Pv)")
    ax.set_ylabel("Concentration (Conc)")
    ax.set_title("ADE breakthrough-curve fits for Cd and Pb")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Cd and Pb BTC data with an ADE finite-difference model.")
    parser.add_argument("--cd", default="Cd.csv", help="Path to Cd CSV file with Pv and Conc columns")
    parser.add_argument("--pb", default="Pb.csv", help="Path to Pb CSV file with Pv and Conc columns")
    parser.add_argument("--output", default="btc_fit_results.png", help="Output PNG path")
    parser.add_argument(
        "--fixed-dispersion",
        action="store_true",
        help="Fit only K and C0 while keeping dispersion fixed at the default value",
    )
    return parser.parse_args(argv)


def print_fit_result(result: FitResult) -> None:
    print(f"{result.name} fit:")
    print(f"  K = {result.k:.6g}")
    print(f"  D = {result.dispersion:.6g}")
    print(f"  C0 = {result.c0:.6g}")
    print(f"  R² = {result.r2:.6g}")
    print(f"  Optimization success = {result.success} ({result.message})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cd_data = load_btc_csv(args.cd, "Cd")
        pb_data = load_btc_csv(args.pb, "Pb")
        cd_result = fit_parameters(cd_data, fit_dispersion=not args.fixed_dispersion)
        pb_result = fit_parameters(pb_data, fit_dispersion=not args.fixed_dispersion)
        print_fit_result(cd_result)
        print_fit_result(pb_result)
        plot_results(cd_result, pb_result, args.output)
        print(f"Saved plot to {Path(args.output).expanduser().resolve()}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
