"""正向 ITO/PEDOT/活性层/PDINN/Ag 器件的边缘取电二维模型。

Ag 和 PDINN 的面内电阻忽略，整面 Ag 与受光面积重合；ITO/PEDOT 构成
并联的阳极片电导，外部阳极接触位于正方形样品 x=0 的整条边。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import brentq, minimize_scalar

from models.device_geometry import DeviceGeometry, MaterialProperties
from simulation.network_solver import PEDOTNetworkSolver


@dataclass
class ForwardResult:
    side_mm: float
    area_mm2: float
    sigma_pedot: float
    r_ito: float
    r_pedot: float
    r_anode: float
    pedot_conductance_fraction: float
    jsc: float
    voc: float
    vmpp: float
    jmpp: float
    ff: float
    pce: float
    potential: Optional[np.ndarray] = None
    x_mm: Optional[np.ndarray] = None
    y_mm: Optional[np.ndarray] = None
    voltage: Optional[np.ndarray] = None
    j_ag: Optional[np.ndarray] = None


def anode_sheet_properties(sigma_pedot: float, thickness_nm: float = 30.0,
                           r_ito: float = 15.0) -> Dict[str, float]:
    """返回 ITO 与 PEDOT 并联后的片电阻及 PEDOT 面内分流占比。"""
    g_ito = 1.0 / r_ito
    g_pedot = sigma_pedot * thickness_nm * 1e-7
    g_total = g_ito + g_pedot
    return {
        "r_pedot": np.inf if g_pedot == 0 else 1.0 / g_pedot,
        "r_anode": 1.0 / g_total,
        "pedot_fraction": g_pedot / g_total,
        "sigma_equivalent": g_total / (thickness_nm * 1e-7),
    }


def _calibrated_material(sigma_pedot: float, thickness_nm: float,
                         r_ito: float, jph: float) -> MaterialProperties:
    """建立局部电池标定：Jsc=25 mA/cm²、Voc=0.9 V、FF≈0.77。"""
    diode_ideality = 1.5
    rsh = 466.15  # Ω cm²，与既有 30 nm 算例一致
    vt = MaterialProperties(temperature=300.0).thermal_voltage
    j0 = (jph - 1000.0 * 0.9 / rsh) / (
        np.exp(0.9 / (diode_ideality * vt)) - 1.0
    )
    props = anode_sheet_properties(sigma_pedot, thickness_nm, r_ito)
    # 复用二维非线性网络求解器：把 ITO+PEDOT 的总片电导映射到等效 30 nm 膜。
    return MaterialProperties(
        pedot_conductivity_lateral=props["sigma_equivalent"],
        jph=jph, j0=j0, diode_ideality=diode_ideality, rsh=rsh,
        contact_resistance_ag=0.0,
    )


def solve_forward_edge_case(side_mm: float, sigma_pedot: float,
                            r_ito: float = 15.0, thickness_nm: float = 30.0,
                            n_nodes: int = 51, save_map: bool = False,
                            diode_parameters: Optional[Dict[str, float]] = None,
                            v_max: float = 1.05) -> ForwardResult:
    """求解单个正方形整面 Ag 器件；阳极在 x=0 全边接触。"""
    side_um = side_mm * 1000.0
    geometry = DeviceGeometry(
        device_width=side_um, device_length=side_um,
        pedot_thickness=thickness_nm,
        # Ag 覆盖/受光面积即整个器件；mask 仅用于指定 ITO 的边缘探针接触。
        ag_width=side_um, ag_length=side_um,
    )
    x = np.linspace(0.0, side_um, n_nodes)
    y = np.linspace(0.0, side_um, n_nodes)
    X, Y = np.meshgrid(x, y)
    edge_contact = np.zeros_like(X, dtype=bool)
    edge_contact[:, 0] = True
    # 网络以节点为控制体积分；边界节点同样计入 dA 时，积分面积比几何面积
    # 略大。按该已知因子标定 Jph，使无横向损失时 Jsc 恰为 25 mA/cm²。
    grid_area_factor = (n_nodes / (n_nodes - 1)) ** 2
    material = _calibrated_material(sigma_pedot, thickness_nm, r_ito,
                                    jph=25.0 / grid_area_factor)
    if diode_parameters is not None:
        from models.local_diode import validate_diode
        validate_diode(diode_parameters)
        material = MaterialProperties(
            pedot_conductivity_lateral=material.pedot_conductivity_lateral,
            contact_resistance_ag=0.0, **diode_parameters)
    solver = PEDOTNetworkSolver(X, Y, geometry, material, edge_contact,
                                illumination="full")

    voltages = np.linspace(0.0, v_max, 43)
    currents = np.empty_like(voltages)
    warm = None
    for k, voltage in enumerate(voltages):
        solver.solve(float(voltage), v0=warm)
        warm = solver._vu
        currents[k] = solver.terminal_current()

    jsc = currents[0] / geometry.active_area
    i_sc = currents[0]
    crossings = np.flatnonzero((currents[:-1] > 0.0) & (currents[1:] <= 0.0))
    if crossings.size == 0:
        raise RuntimeError("J-V 扫描范围内未找到 Voc")
    k_voc = int(crossings[0])

    def current_at(voltage: float) -> float:
        solver.solve(float(voltage), v0=solver._vu)
        return solver.terminal_current()

    voc = float(brentq(current_at, voltages[k_voc], voltages[k_voc + 1],
                       xtol=1e-8, rtol=1e-10))
    powers = voltages * currents
    k_mpp = int(np.argmax(powers))
    bracket = (voltages[max(k_mpp - 1, 0)], voltages[min(k_mpp + 1, len(voltages) - 1)])
    optimum = minimize_scalar(lambda v: -v * current_at(v), bounds=bracket,
                               method="bounded", options={"xatol": 1e-7})
    vmpp = float(optimum.x)
    impp = float(-optimum.fun / vmpp)
    pmpp = vmpp * impp
    ff = pmpp / (i_sc * voc)
    pce = pmpp / (100.0 * geometry.active_area) * 100.0
    props = anode_sheet_properties(sigma_pedot, thickness_nm, r_ito)

    potential = x_mm = y_mm = None
    if save_map:
        potential = solver.solve(vmpp).copy()
        x_mm, y_mm = X / 1000.0, Y / 1000.0
    return ForwardResult(
        side_mm=side_mm, area_mm2=side_mm ** 2, sigma_pedot=sigma_pedot,
        r_ito=r_ito, r_pedot=props["r_pedot"], r_anode=props["r_anode"],
        pedot_conductance_fraction=props["pedot_fraction"], jsc=jsc, voc=voc,
        vmpp=vmpp, jmpp=impp / geometry.active_area, ff=ff, pce=pce,
        potential=potential, x_mm=x_mm, y_mm=y_mm,
        voltage=voltages.copy(), j_ag=currents / geometry.ag_area,
    )


def run_forward_size_scan(sides_mm: Sequence[float], sigmas: Sequence[float],
                          **kwargs) -> list[ForwardResult]:
    """扫描面积与 PEDOT 电导率。"""
    results = []
    total = len(sides_mm) * len(sigmas)
    for index, sigma in enumerate(sigmas, start=1):
        for side in sides_mm:
            item = solve_forward_edge_case(side, sigma, **kwargs)
            results.append(item)
            print(f"[{len(results):02d}/{total}] side={side:g} mm, sigma={sigma:g} S/cm: "
                  f"PCE={item.pce:.3f}%, FF={item.ff:.4f}")
    return results


def export_forward_scan(results: Iterable[ForwardResult], path: str) -> None:
    fields = list(ForwardResult.__dataclass_fields__.keys())[:13]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({name: getattr(item, name) for name in fields})


def plot_forward_size_scan(results: Sequence[ForwardResult], path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    sigmas = sorted({item.sigma_pedot for item in results})
    for sigma in sigmas:
        rows = sorted((item for item in results if item.sigma_pedot == sigma),
                      key=lambda item: item.area_mm2)
        x = [item.area_mm2 for item in rows]
        label = f"{sigma:g} S/cm"
        axes[0].semilogx(x, [item.pce for item in rows], "o-", ms=4, label=label)
        axes[1].semilogx(x, [item.ff for item in rows], "o-", ms=4, label=label)
        axes[2].semilogx(x, [item.jsc for item in rows], "o-", ms=4, label=label)
    axes[0].set_ylabel("PCE (%)")
    axes[1].set_ylabel("FF")
    axes[2].set_ylabel("Jsc (mA/cm²)")
    for axis, title in zip(axes, ["效率随器件面积变化", "FF 随器件面积变化", "Jsc 随器件面积变化"]):
        axis.set_xlabel("正方形器件面积 / Ag 覆盖面积 (mm²)")
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[0].legend(title="PEDOT σ", fontsize=8)
    fig.suptitle("正向 ITO/PEDOT/活性层/PDINN/Ag：ITO 边缘取电，R□,ITO = 15 Ω/□", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forward_heatmap(results: Sequence[ForwardResult], path: str) -> None:
    sides = sorted({item.side_mm for item in results})
    sigmas = sorted({item.sigma_pedot for item in results})
    pce = np.full((len(sigmas), len(sides)), np.nan)
    for item in results:
        pce[sigmas.index(item.sigma_pedot), sides.index(item.side_mm)] = item.pce
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    image = ax.imshow(pce, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sides)), [f"{side:g}" for side in sides])
    ax.set_yticks(range(len(sigmas)), [f"{sigma:g}" for sigma in sigmas])
    ax.set_xlabel("正方形边长 (mm)")
    ax.set_ylabel("PEDOT 横向电导率 σ (S/cm)")
    ax.set_title("PCE (%)：整面 Ag 受光、ITO x=0 边缘取电")
    for iy in range(len(sigmas)):
        for ix in range(len(sides)):
            ax.text(ix, iy, f"{pce[iy, ix]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if pce[iy, ix] < np.nanmean(pce) else "black")
    fig.colorbar(image, ax=ax, label="PCE (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_composite_sheet_resistance(path: str, r_ito: float = 15.0,
                                    thickness_nm: float = 30.0) -> None:
    """绘制 PEDOT/ITO 并联复合层方阻随 PEDOT 电导率的变化。"""
    sigma = np.logspace(-4, 6, 600)
    props = [anode_sheet_properties(value, thickness_nm, r_ito) for value in sigma]
    r_composite = np.array([item["r_anode"] for item in props])
    reduction = (1.0 - r_composite / r_ito) * 100.0
    sigma_match = 1.0 / (r_ito * thickness_nm * 1e-7)
    markers = [(2.3e-3, "AI 4083\n2.3e-3 S/cm", "tab:blue"),
               (2.0, "核心算例\n2 S/cm", "tab:red"),
               (sigma_match, "与 ITO 匹配\n2.22×10⁴ S/cm", "tab:purple")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    ax = axes[0]
    ax.semilogx(sigma, r_composite, color="tab:green", lw=2.5,
                label="PEDOT/ITO 复合层")
    ax.axhline(r_ito, color="black", ls="--", lw=1.2, label="ITO：15 Ω/□")
    for value, label, color in markers:
        result = anode_sheet_properties(value, thickness_nm, r_ito)
        ax.plot(value, result["r_anode"], "o", color=color, ms=7)
        ax.annotate(label, (value, result["r_anode"]), xytext=(7, -34),
                    textcoords="offset points", color=color, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
    ax.set_xlabel("PEDOT 横向电导率 σ (S/cm)")
    ax.set_ylabel("复合阳极方阻 R_sheet (Ω/□)")
    ax.set_title("30 nm PEDOT/ITO 并联复合层")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.semilogx(sigma, reduction, color="tab:orange", lw=2.5)
    ax.axvline(sigma_match, color="tab:purple", ls=":", lw=1.5,
               label="PEDOT 与 ITO 片电导相等")
    for value, label, color in markers[:2]:
        result = anode_sheet_properties(value, thickness_nm, r_ito)
        value_reduction = (1.0 - result["r_anode"] / r_ito) * 100.0
        ax.plot(value, value_reduction, "o", color=color, ms=7)
    ax.set_xlabel("PEDOT 横向电导率 σ (S/cm)")
    ax.set_ylabel("相对 ITO 单层的方阻降低 (%)")
    ax.set_title("PEDOT 对阳极片电导的贡献")
    ax.set_ylim(-1, 101)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_composite_sheet_resistance_map(path: str, thickness_nm: float = 30.0) -> None:
    """绘制 PEDOT 电导率--ITO 方阻--复合方阻的二维映射。"""
    sigma = np.logspace(-4, 6, 401)
    r_ito = np.logspace(0, 2, 301)  # 1--100 Ω/□，覆盖常用透明电极范围
    sigma_grid, r_ito_grid = np.meshgrid(sigma, r_ito)
    r_composite = 1.0 / (1.0 / r_ito_grid + sigma_grid * thickness_nm * 1e-7)

    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    levels = np.logspace(-2, 2, 81)
    image = ax.contourf(sigma_grid, r_ito_grid, r_composite,
                        levels=levels, norm=LogNorm(vmin=1e-2, vmax=1e2),
                        cmap="viridis", extend="both")
    contours = ax.contour(sigma_grid, r_ito_grid, r_composite,
                          levels=[0.3, 1, 3, 10, 15, 30], colors="white",
                          linewidths=0.8, alpha=0.85)
    ax.clabel(contours, fmt="%g Ω/□", fontsize=8)
    ax.axhline(15.0, color="tab:red", ls="--", lw=1.7, label="当前 ITO：15 Ω/□")
    ax.axvline(2.0, color="tab:orange", ls="--", lw=1.7, label="核心 PEDOT：2 S/cm")
    ax.plot(2.3e-3, 15.0, "o", color="white", markeredgecolor="black", ms=7,
            label="AI 4083：2.3e-3 S/cm")
    ax.plot(2.0, 15.0, "o", color="tab:red", markeredgecolor="white", ms=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PEDOT 横向电导率 σ (S/cm)")
    ax.set_ylabel("ITO 方阻 R_sheet,ITO (Ω/□)")
    ax.set_title("30 nm PEDOT/ITO 并联复合层方阻二维映射")
    ax.legend(loc="lower left", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, label="复合层方阻 R_sheet,comp (Ω/□)")
    colorbar.set_ticks([0.01, 0.1, 1, 10, 100])
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forward_potential_comparison(cases: Sequence[ForwardResult], path: str) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 5.2), squeeze=False)
    all_drop = [np.max(item.potential - item.vmpp) * 1e3 for item in cases]
    vmax = max(all_drop) if max(all_drop) > 0 else 1.0
    for ax, item in zip(axes[0], cases):
        drop_mv = (item.potential - item.vmpp) * 1e3
        image = ax.contourf(item.x_mm, item.y_mm, drop_mv, levels=40,
                            cmap="magma", vmin=0.0, vmax=vmax)
        ax.axvline(0.0, color="cyan", lw=3, label="ITO 边缘接触")
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"σ={item.sigma_pedot:g} S/cm\nPCE={item.pce:.2f}%, FF={item.ff:.3f}")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="局部结电压相对端电压的偏差 (mV)")
    fig.suptitle("MPP：阳极横向压降造成的局部结电压偏差", y=1.02)
    fig.subplots_adjust(top=0.86, wspace=0.30)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
