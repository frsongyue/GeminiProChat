#!/usr/bin/env python3
"""
使用一维 ADE 有限差分模型拟合 Cd 和 Pb 的穿透曲线（BTC）表格数据。

用法：
    python btc_fit_ade.py --cd Cd.csv --pb Pb.csv --output btc_fit_results.png
    # 也可以只拟合 Cd：python btc_fit_ade.py --cd Cd.txt

如果聊天框无法识别 CSV，可把文件改名为 .txt 后上传或直接粘贴两列数据；
脚本读取的是文件内容，不强制要求扩展名必须是 .csv。PDF 需要先另存/转换为
含 Pv 和 Conc 两列的 CSV/TXT 表格后再拟合。表格可以有表头，也可以像
“0.219  0.435”这样直接给两列数字，空格、Tab、逗号、分号分隔均可。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erfc


# 数据类：保存实验数据和拟合结果，便于函数之间传递。
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


# 输入读取与校验：检查表格文件是否存在，并提取 Pv/Conc 两列。
def validate_csv_path(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"找不到输入文件： {file_path}")
    if not file_path.is_file():
        raise ValueError(f"输入路径不是文件： {file_path}")
    return file_path


def load_btc_csv(path: str | Path, name: str) -> BTCData:
    file_path = validate_csv_path(path)

    # 先按“有表头”读取；若找不到 Pv/Conc 表头，则按“无表头两列数字”重读，
    # 这样可直接使用从聊天框或 PDF 识别结果中复制出来的 Tab/空格分隔数据。
    separator_pattern = r"[\s,;]+"
    data_with_header = pd.read_csv(file_path, sep=separator_pattern, engine="python")
    columns_lower = {str(col).strip().lower(): col for col in data_with_header.columns}

    if "pv" in columns_lower and "conc" in columns_lower:
        data = data_with_header
        pv_col = columns_lower["pv"]
        conc_col = columns_lower["conc"]
    else:
        data = pd.read_csv(file_path, sep=separator_pattern, engine="python", header=None)
        if data.shape[1] < 2:
            raise ValueError(f"{file_path} 必须至少包含两列：Pv 和 Conc；如果来自 PDF，请先转换为 CSV/TXT 表格")
        pv_col = data.columns[0]
        conc_col = data.columns[1]

    numeric = pd.DataFrame(
        {
            "Pv": pd.to_numeric(data[pv_col], errors="coerce"),
            "Conc": pd.to_numeric(data[conc_col], errors="coerce"),
        }
    ).dropna()

    if numeric.empty:
        raise ValueError(f"{file_path} 没有有效的数值型 Pv/Conc 数据行")

    numeric = numeric.sort_values("Pv")
    pv = numeric["Pv"].to_numpy(dtype=float)
    conc = numeric["Conc"].to_numpy(dtype=float)

    if np.any(np.diff(pv) <= 0):
        raise ValueError(f"{file_path} 包含重复或非递增的孔体积（Pv）值")
    if np.any(pv < 0):
        raise ValueError(f"{file_path} 包含负的孔体积（Pv）值")

    return BTCData(name=name, pv=pv, conc=conc)


# ADE 模型：Ogata-Banks 解析边界条件和有限差分数值求解。
def ogata_banks_step(x: np.ndarray | float, t: np.ndarray | float, velocity: float, dispersion: float, c0: float) -> np.ndarray:
    """Ogata-Banks 半无限域、阶跃输入 ADE 解析解（不含动力学损失）。"""
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
    """一阶损失项中的无量纲乘子 f(x,t)；如有需要可替换为自定义函数。"""
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
    """带一阶动力学损失项的一维 ADE 后向欧拉有限差分求解器。"""
    if np.any(times < 0):
        raise ValueError("模拟时间（Pv）必须为非负数")
    if k < 0 or dispersion <= 0 or c0 < 0:
        raise ValueError("模型参数必须满足 K >= 0、D > 0 且 C0 >= 0")

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
            raise ValueError("sink_factor 必须返回与 x 形状相同的数组")

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


def calculate_r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum == 0.0:
        return float("nan")
    return 1.0 - residual_sum / total_sum


# 参数拟合：使用最小二乘优化 K、D 和入口浓度 C0。
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
    r2 = calculate_r2_score(conc, conc_fit) if len(conc) > 1 else float("nan")

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


# 绘图：在无 GUI 的服务器环境中保存 PNG 文件。
def plot_results(results: Sequence[FitResult], output_path: str | Path = "btc_fit_results.png") -> None:
    if not results:
        raise ValueError("至少需要一个拟合结果才能绘图")

    output = Path(output_path).expanduser().resolve()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    colors = ("tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange")

    for result, color in zip(results, colors, strict=False):
        pv_smooth = np.linspace(float(np.min(result.pv)), float(np.max(result.pv)), 250)
        conc_smooth = solve_ade_finite_difference(pv_smooth, result.k, result.dispersion, result.c0)
        ax.scatter(result.pv, result.conc_exp, s=35, color=color, alpha=0.75, label=f"{result.name} 实验值")
        ax.plot(
            pv_smooth,
            conc_smooth,
            color=color,
            linewidth=2.0,
            label=f"{result.name} ADE 拟合 (R²={result.r2:.3f})",
        )

    ax.set_xlabel("孔体积 (Pv)")
    ax.set_ylabel("浓度 (Conc)")
    ax.set_title("ADE 穿透曲线拟合")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# 命令行入口：解析文件路径、运行拟合并输出结果。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 ADE 有限差分模型拟合 Cd 和 Pb 的 BTC 表格数据。")
    parser.add_argument("--cd", default=None, help="Cd 表格文件路径（CSV 或改名后的 TXT；可有表头，也可直接两列数字）")
    parser.add_argument("--pb", default=None, help="Pb 表格文件路径（CSV 或改名后的 TXT；可有表头，也可直接两列数字）")
    parser.add_argument("--output", default="btc_fit_results.png", help="输出 PNG 图片路径")
    parser.add_argument(
        "--fixed-dispersion",
        action="store_true",
        help="仅拟合 K 和 C0，并将弥散系数固定为默认值",
    )
    return parser.parse_args(argv)


def print_fit_result(result: FitResult) -> None:
    print(f"{result.name} 拟合结果：")
    print(f"  K = {result.k:.6g}")
    print(f"  D = {result.dispersion:.6g}")
    print(f"  C0 = {result.c0:.6g}")
    print(f"  R² = {result.r2:.6g}")
    print(f"  优化是否成功 = {result.success} ({result.message})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs: list[tuple[str, str]] = []
        if args.cd:
            inputs.append(("Cd", args.cd))
        if args.pb:
            inputs.append(("Pb", args.pb))

        # 如果没有显式指定路径，则自动使用当前目录中存在的默认文件；
        # 这样既兼容 Cd.csv/Pb.csv 的常见用法，也允许只拟合刚粘贴/保存的 Cd 数据。
        if not inputs:
            for name, default_path in (("Cd", "Cd.csv"), ("Pb", "Pb.csv")):
                if Path(default_path).exists():
                    inputs.append((name, default_path))

        if not inputs:
            raise ValueError("请至少提供一个输入文件，例如 --cd Cd.txt；如果要同时拟合 Pb，再加 --pb Pb.txt")

        results: list[FitResult] = []
        for name, file_path in inputs:
            data = load_btc_csv(file_path, name)
            result = fit_parameters(data, fit_dispersion=not args.fixed_dispersion)
            print_fit_result(result)
            results.append(result)

        plot_results(results, args.output)
        print(f"已保存图像到 {Path(args.output).expanduser().resolve()}")
    except Exception as exc:
        print(f"错误： {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
