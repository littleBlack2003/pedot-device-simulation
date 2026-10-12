"""
器件性能分析

包含:
    - 终端 J-V 扫描与特征参数提取（Jsc、Voc、FF、PCE）；
    - 双面积归一化: J_ag = I/A_ag 与 J_active = I/A_active，避免把横向
      扩展收集误解为单位活性层面积光电转换能力的提高；
    - 局部收集效率与有效收集长度诊断；
    - 参数扫描（电导率、PEDOT 厚度、Ag 尺寸、接触电阻）；
    - 光照模式对比（整个活性区受光 vs 仅 Ag 覆盖区受光）；
    - 短路 / 最大功率点 / 开路工作点验证；
    - 结果图表与文本报告。
"""

import csv
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq, minimize_scalar

from models.device_geometry import DeviceGeometry, MaterialProperties
from simulation.network_solver import PEDOTNetworkSolver


# ----------------------------------------------------------------------
# J-V 结果容器
# ----------------------------------------------------------------------
@dataclass
class JVResult:
    """终端 J-V 特性（含双面积归一化）"""
    voltage: np.ndarray          # 扫描电压 (V)
    current: np.ndarray          # 终端电流 (mA)
    j_ag: np.ndarray             # J_ag = I / A_ag (mA/cm^2)
    j_active: np.ndarray         # J_active = I / A_active (mA/cm^2)
    area_ag: float               # A_ag (cm^2)
    area_active: float           # A_active (cm^2)

    jsc_ag: float                # 短路电流密度，以 A_ag 归一 (mA/cm^2)
    jsc_active: float            # 短路电流密度，以 A_active 归一 (mA/cm^2)
    voc: float                   # 开路电压 (V，不依赖面积定义)
    vmpp: float                  # 最大功率点电压 (V)
    impp: float                  # 最大功率点电流 (mA)
    pmpp: float                  # 最大功率 (mW)
    ff: float                    # 填充因子（不依赖面积定义）
    pce_ag: float                # PCE，以 A_ag 归一 (%)
    pce_active: float            # PCE，以 A_active 归一 (%)

    collection: Dict = field(default_factory=dict)   # 短路点收集诊断
    mpp_collection: Dict = field(default_factory=dict)  # 最大功率点收集诊断


# ----------------------------------------------------------------------
# J-V 扫描
# ----------------------------------------------------------------------
def compute_jv(solver: PEDOTNetworkSolver,
               v_range: Sequence[float] = (0.0, 1.10),
               n_points: int = 41,
               light_intensity: float = 100.0,
               with_collection: bool = True,
               collection_maps: bool = False,
               verbose: bool = True) -> JVResult:
    """
    扫描外加电压，计算终端 J-V 特性

    Args:
        solver: 已配置的 PEDOTNetworkSolver
        v_range: 电压扫描范围 (V)
        n_points: 采样点数
        light_intensity: 入射光功率密度 (mW/cm^2)，AM1.5G 为 100
        with_collection: 是否在短路点附加收集效率诊断
        collection_maps: 是否在结果中保留收集效率分布图
        verbose: 打印进度

    Returns:
        JVResult
    """
    voltages = np.linspace(v_range[0], v_range[1], n_points)
    current = np.empty(n_points)
    warm = None

    if verbose:
        print(f"  扫描 J-V 曲线 ({n_points} 个电压点)...")

    for k, v in enumerate(voltages):
        solver.solve(v, v0=warm)
        warm = solver._vu
        current[k] = solver.terminal_current()
        if verbose and (k % 10 == 0 or k == n_points - 1):
            print(f"    V = {v:.3f} V, I = {current[k]:.4f} mA")

    j_ag = current / solver.geometry.ag_area
    j_active = current / solver.geometry.active_area
    power = voltages * current              # mW

    k_mpp = int(np.argmax(power))
    jsc_ag = float(np.interp(0.0, voltages, j_ag))
    jsc_active = float(np.interp(0.0, voltages, j_active))
    i_sc = float(np.interp(0.0, voltages, current))

    # 用连续的非线性求解器精确定位 Voc，而不是仅在两个离散 J-V 点间插值。
    # 这也使开路 KCL 检查不再受扫描步长影响。
    voc = np.nan
    for k in range(n_points - 1):
        if current[k] > 0.0 >= current[k + 1]:
            def current_at(v: float) -> float:
                solver.solve(v, v0=solver._vu)
                return solver.terminal_current()

            voc = float(brentq(current_at, voltages[k], voltages[k + 1],
                                xtol=1e-8, rtol=1e-10))
            break

    # 在离散功率峰两侧作一维有界优化；返回的 MPP 不再被 15 mV 的扫描
    # 步长量化。边界峰值时保留扫描点，避免错误外推。
    if 0 < k_mpp < n_points - 1:
        def negative_power(v: float) -> float:
            solver.solve(v, v0=solver._vu)
            return -v * solver.terminal_current()

        optimum = minimize_scalar(
            negative_power,
            bounds=(voltages[k_mpp - 1], voltages[k_mpp + 1]),
            method="bounded",
            options={"xatol": 1e-7},
        )
        vmpp = float(optimum.x)
        impp = float(-optimum.fun / vmpp)
    else:
        vmpp = float(voltages[k_mpp])
        impp = float(current[k_mpp])
    pmpp = float(vmpp * impp)
    ff = pmpp / (i_sc * voc) if (voc and voc > 0 and i_sc > 0) else np.nan
    pce_ag = pmpp / (light_intensity * solver.geometry.ag_area) * 100.0
    pce_active = pmpp / (light_intensity * solver.geometry.active_area) * 100.0

    collection = {}
    if with_collection:
        solver.solve(0.0)
        collection = solver.collection_efficiency_map()
        collection["voltage_drop_max"] = float(solver.potential.max())
        collection["i_sc"] = i_sc
        if not collection_maps:
            collection.pop("eta_map", None)

    # 对局部 Ag 器件而言，短路收集范围不能代表工作状态下的收集范围。
    # 在每条 J-V 曲线的 MPP 再求解一次，保留可比的 A_eff 和 L_col；
    # 空间大图仍按需在 plot_spatial_conductivity_scan 中重新计算，避免
    # 在整个参数扫描中存储大量二维数组。
    solver.solve(vmpp, v0=warm)
    mpp_collection = solver.collection_efficiency_map()
    if not collection_maps:
        mpp_collection.pop("eta_map", None)

    return JVResult(
        voltage=voltages, current=current,
        j_ag=j_ag, j_active=j_active,
        area_ag=solver.geometry.ag_area,
        area_active=solver.geometry.active_area,
        jsc_ag=jsc_ag, jsc_active=jsc_active, voc=voc,
        vmpp=vmpp, impp=impp, pmpp=pmpp, ff=ff,
        pce_ag=pce_ag, pce_active=pce_active,
        collection=collection,
        mpp_collection=mpp_collection,
    )


# ----------------------------------------------------------------------
# 工作点验证（短路 / 最大功率点 / 开路）
# ----------------------------------------------------------------------
def verify_operating_points(solver: PEDOTNetworkSolver,
                            v_range: Sequence[float] = (0.0, 1.10),
                            n_points: int = 26) -> Dict:
    """
    在短路、最大功率点与开路条件下验证电势与电流分布的合理性

    Returns:
        dict: {
            'short_circuit': {...}, 'mpp': {...}, 'open_circuit': {...},
            'checks': {名称: bool}, 'jv': JVResult
        }
    """
    # 短路
    solver.solve(0.0)
    i_sc = solver.terminal_current()
    injected = solver.total_injected_current()
    coll = solver.collection_efficiency_map()
    v_hump = float(solver.potential.max())
    short_circuit = {
        "i_terminal_mA": i_sc,
        "i_injected_mA": injected,
        "kcl_error": abs(i_sc - injected) / max(abs(i_sc), 1e-30),
        "a_effective_cm2": coll["a_effective"],
        "a_effective_over_ag": coll["a_effective"] / solver.geometry.ag_area,
        "collection_length_cm": coll["collection_length"],
        "v_hump_V": v_hump,
    }

    # J-V 扫描（最大功率点）
    jv = compute_jv(solver, v_range=v_range, n_points=n_points,
                    with_collection=False, verbose=False)

    # 开路: 在 Voc 复核终端电流
    solver.solve(jv.voc)
    i_oc = solver.terminal_current()
    open_circuit = {
        "voc_V": jv.voc,
        "i_terminal_mA": i_oc,
        "v_pedot_max_V": float(solver.potential.max()),
    }

    checks = {
        "短路终端电流为正": i_sc > 0,
        "短路下 KCL 自洽(终端≈注入)": short_circuit["kcl_error"] < 1e-3,
        "开路终端电流接近零": abs(i_oc) < 0.02 * i_sc,
        "Voc 落在扫描范围内": v_range[0] < jv.voc < v_range[1],
        "Voc 不超过二极管极限": jv.voc <= solver.n_ideal * solver.vt
                                 * np.log(solver.material.jph / solver.material.j0 + 1) + 0.05,
        "Jsc_active 不超过理想光生电流": jv.jsc_active <= solver.material.jph * 1.02,
        "FF 在合理区间": 0.1 < jv.ff <= 1.0,
        "最大功率点为正输出": jv.vmpp > 0 and jv.impp > 0,
        "Jsc_ag 可大于 Jph(A_effective > A_ag 的表现)": True,
    }
    return {
        "short_circuit": short_circuit,
        "mpp": {"vmpp": jv.vmpp, "impp": jv.impp, "ff": jv.ff},
        "open_circuit": open_circuit,
        "checks": checks,
        "jv": jv,
    }


# ----------------------------------------------------------------------
# 参数扫描
# ----------------------------------------------------------------------
def _run_scan(values: Sequence[float], label: str,
              build: Callable[[float], PEDOTNetworkSolver],
              v_range: Sequence[float], n_points: int,
              verbose: bool = True) -> Dict[float, JVResult]:
    results = {}
    if verbose:
        print(f"\n扫描 {label}: {values}")
    for value in values:
        if verbose:
            print(f"\n{label} = {value:g}")
        solver = build(value)
        results[float(value)] = compute_jv(solver, v_range=v_range,
                                           n_points=n_points, verbose=verbose)
    return results


def run_conductivity_scan(geometry: DeviceGeometry, material: MaterialProperties,
                          X: np.ndarray, Y: np.ndarray, ag_mask: np.ndarray,
                          values: Sequence[float],
                          illumination: str = "full",
                          v_range: Sequence[float] = (0.0, 1.10),
                          n_points: int = 31,
                          verbose: bool = True) -> Dict[float, JVResult]:
    """扫描 PEDOT 横向电导率 sigma_parallel (S/cm)"""
    return _run_scan(
        values, "PEDOT 横向电导率 (S/cm)",
        lambda s: PEDOTNetworkSolver(
            X, Y, geometry, replace(material, pedot_conductivity_lateral=float(s)),
            ag_mask, illumination=illumination),
        v_range, n_points, verbose,
    )


def run_thickness_scan(geometry: DeviceGeometry, material: MaterialProperties,
                       X: np.ndarray, Y: np.ndarray, ag_mask: np.ndarray,
                       values: Sequence[float],
                       illumination: str = "full",
                       v_range: Sequence[float] = (0.0, 1.10),
                       n_points: int = 31,
                       verbose: bool = True) -> Dict[float, JVResult]:
    """扫描 PEDOT 厚度 t_PEDOT (nm)"""
    return _run_scan(
        values, "PEDOT 厚度 (nm)",
        lambda t: PEDOTNetworkSolver(
            X, Y, replace(geometry, pedot_thickness=float(t)), material,
            ag_mask, illumination=illumination),
        v_range, n_points, verbose,
    )


def run_ag_size_scan(geometry: DeviceGeometry, material: MaterialProperties,
                     X: np.ndarray, Y: np.ndarray, ag_mask: np.ndarray,
                     values: Sequence[float],
                     illumination: str = "full",
                     v_range: Sequence[float] = (0.0, 1.10),
                     n_points: int = 31,
                     verbose: bool = True) -> Dict[float, JVResult]:
    """扫描 Ag 掩模板尺寸（矩形边长或圆形直径, μm）"""
    return _run_scan(
        values, "Ag 掩模板尺寸 (μm)",
        lambda s: PEDOTNetworkSolver(
            X, Y, replace(geometry, ag_width=float(s), ag_length=float(s)),
            material, ag_mask, illumination=illumination),
        v_range, n_points, verbose,
    )


def run_contact_resistance_scan(geometry: DeviceGeometry, material: MaterialProperties,
                                X: np.ndarray, Y: np.ndarray, ag_mask: np.ndarray,
                                values: Sequence[float],
                                illumination: str = "full",
                                v_range: Sequence[float] = (0.0, 1.10),
                                n_points: int = 31,
                                verbose: bool = True) -> Dict[float, JVResult]:
    """扫描 Ag/PEDOT 垂直接触电阻 (Ω·cm^2)"""
    return _run_scan(
        values, "Ag 接触电阻 (Ω·cm^2)",
        lambda rc: PEDOTNetworkSolver(
            X, Y, geometry, replace(material, contact_resistance_ag=float(rc)),
            ag_mask, illumination=illumination),
        v_range, n_points, verbose,
    )


def compare_illumination(geometry: DeviceGeometry, material: MaterialProperties,
                         X: np.ndarray, Y: np.ndarray, ag_mask: np.ndarray,
                         v_range: Sequence[float] = (0.0, 1.10),
                         n_points: int = 41) -> Dict[str, JVResult]:
    """
    对比“整个活性区均匀受光”与“仅 Ag 覆盖区受光”，
    用于识别 Ag 覆盖区外产生的额外电流贡献
    """
    print("\n对比光照模式: full（整个活性区均匀受光） vs ag_only（仅 Ag 覆盖区受光）")
    out = {}
    for mode in ("full", "ag_only"):
        print(f"\n光照模式 = {mode}")
        solver = PEDOTNetworkSolver(X, Y, geometry, material, ag_mask,
                                    illumination=mode)
        out[mode] = compute_jv(solver, v_range=v_range, n_points=n_points)
    return out


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def _fmt(value: float, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "  --  "
    return format(value, spec)


def generate_report(geometry: DeviceGeometry, material: MaterialProperties,
                    sigma_results: Optional[Dict[float, JVResult]] = None,
                    thickness_results: Optional[Dict[float, JVResult]] = None,
                    ag_size_results: Optional[Dict[float, JVResult]] = None,
                    contact_results: Optional[Dict[float, JVResult]] = None,
                    illumination: Optional[Dict[str, JVResult]] = None,
                    verification: Optional[Dict] = None) -> str:
    """生成包含双面积归一化与参数扫描的文本报告"""
    g_sheet = material.pedot_conductivity_lateral * geometry.pedot_thickness * 1e-7
    lines = []
    add = lines.append

    add("=" * 92)
    add("PEDOT 横向电导率对局部 Ag 顶电极有机太阳能电池影响 - 仿真报告")
    add("=" * 92)
    add("")
    add("器件设定:")
    add(f"  活性区尺寸: {geometry.device_width / 1000:.1f} mm × {geometry.device_length / 1000:.1f} mm"
        f"   A_active = {geometry.active_area * 100:.2f} mm^2")
    shape_txt = (f"矩形 {geometry.ag_width / 1000:.1f} × {geometry.ag_length / 1000:.1f} mm"
                 if geometry.ag_shape == "rectangle"
                 else f"圆形直径 {geometry.ag_width / 1000:.1f} mm")
    add(f"  Ag 掩模板: {shape_txt}   A_ag = {geometry.ag_area * 100:.2f} mm^2"
        f"   (覆盖率 {geometry.ag_area / geometry.active_area * 100:.1f}%)")
    add(f"  PEDOT: σ∥ = {material.pedot_conductivity_lateral:g} S/cm, "
        f"t = {geometry.pedot_thickness:g} nm → R_sheet = {1 / g_sheet:.3g} Ω/sq")
    add(f"  微电池: Jph = {material.jph:g} mA/cm^2, J0 = {material.j0:g} mA/cm^2, "
        f"n = {material.diode_ideality:g}, Rsh = {material.rsh:g} Ω·cm^2, "
        f"T = {material.temperature:g} K")
    add(f"  Ag/PEDOT 接触电阻: {material.contact_resistance_ag:g} Ω·cm^2")
    add("")
    add("面积归一化说明:")
    add("  J_ag     = I_terminal / A_ag（实验名义器件面积）")
    add("  J_active = I_terminal / A_active（连续活性层受光面积）")
    add("  若 A_effective > A_ag，J_ag 偏高不代表单位活性层面积光电转换能力提高;")
    add("  Voc 与 FF 不依赖面积定义。")
    add("")

    if sigma_results:
        add("-" * 92)
        add("1) PEDOT 横向电导率扫描（短路点收集诊断在 V = 0 求得）")
        add("-" * 92)
        header = (f"{'σ∥ (S/cm)':>10} {'R_sheet(Ω/sq)':>14} {'Jsc_ag':>9} {'Jsc_active':>11} "
                  f"{'Voc(V)':>8} {'FF':>7} {'PCE_ag(%)':>10} {'PCE_act(%)':>11} "
                  f"{'Aeff/Ag_SC':>11} {'Lsc(μm)':>9} {'Aeff/Ag_MPP':>12} {'Lmpp(μm)':>10}")
        add(header)
        add("-" * 92)
        for sigma in sorted(sigma_results):
            jv = sigma_results[sigma]
            r_sheet = 1.0 / (sigma * geometry.pedot_thickness * 1e-7)
            coll = jv.collection
            coll_mpp = jv.mpp_collection
            a_ratio = (coll.get("a_effective", np.nan) / geometry.ag_area
                       if coll else np.nan)
            l_col = coll.get("collection_length", np.nan) * 1e4 if coll else np.nan
            a_ratio_mpp = (coll_mpp.get("a_effective", np.nan) / geometry.ag_area
                           if coll_mpp else np.nan)
            l_col_mpp = (coll_mpp.get("collection_length", np.nan) * 1e4
                         if coll_mpp else np.nan)
            add(f"{sigma:>10g} {r_sheet:>14.3g} {_fmt(jv.jsc_ag, '.2f'):>9} "
                f"{_fmt(jv.jsc_active, '.2f'):>11} {_fmt(jv.voc, '.3f'):>8} "
                f"{_fmt(jv.ff, '.3f'):>7} {_fmt(jv.pce_ag, '.2f'):>10} "
                f"{_fmt(jv.pce_active, '.2f'):>11} {_fmt(a_ratio, '.2f'):>11} "
                f"{_fmt(l_col, '.0f'):>9} {_fmt(a_ratio_mpp, '.2f'):>12} "
                f"{_fmt(l_col_mpp, '.0f'):>10}")
        add("")

        sigmas = sorted(sigma_results)
        s_lo, s_hi = sigmas[0], sigmas[-1]
        jv_lo, jv_hi = sigma_results[s_lo], sigma_results[s_hi]
        add("关键发现:")
        add(f"  - Jsc_ag 由 {jv_lo.jsc_ag:.2f}（σ∥={s_lo:g}）变化至 {jv_hi.jsc_ag:.2f} mA/cm^2"
            f"（σ∥={s_hi:g}），Jsc_active 由 {jv_lo.jsc_active:.2f} 至 {jv_hi.jsc_active:.2f} mA/cm^2;")
        add(f"  - FF 由 {jv_lo.ff:.3f} 变化至 {jv_hi.ff:.3f}，Voc 变化 "
            f"{jv_hi.voc - jv_lo.voc:+.3f} V;")
        add(f"  - 低电导率下 A_effective/A_ag = "
            f"{jv_lo.collection.get('a_effective', float('nan')) / geometry.ag_area:.2f}，"
            f"高电导率下 = "
            f"{jv_hi.collection.get('a_effective', float('nan')) / geometry.ag_area:.2f}:")
        add("    A_effective > A_ag 时，以 A_ag 归一化的 Jsc、PCE 被横向扩展收集抬高。")
        add("")

    if thickness_results:
        add("-" * 92)
        add("2) PEDOT 厚度扫描")
        add("-" * 92)
        add(f"{'t (nm)':>8} {'R_sheet(Ω/sq)':>14} {'Jsc_ag':>9} {'Jsc_active':>11} "
            f"{'Voc(V)':>8} {'FF':>7} {'PCE_ag(%)':>10} {'L_col(μm)':>10}")
        add("-" * 92)
        for t in sorted(thickness_results):
            jv = thickness_results[t]
            r_sheet = 1.0 / (material.pedot_conductivity_lateral * t * 1e-7)
            l_col = jv.collection.get("collection_length", np.nan) * 1e4 if jv.collection else np.nan
            add(f"{t:>8g} {r_sheet:>14.3g} {_fmt(jv.jsc_ag, '.2f'):>9} "
                f"{_fmt(jv.jsc_active, '.2f'):>11} {_fmt(jv.voc, '.3f'):>8} "
                f"{_fmt(jv.ff, '.3f'):>7} {_fmt(jv.pce_ag, '.2f'):>10} {_fmt(l_col, '.0f'):>10}")
        add("")

    if ag_size_results:
        add("-" * 92)
        add("3) Ag 掩模板尺寸扫描")
        add("-" * 92)
        add(f"{'尺寸(μm)':>10} {'A_ag/A_act':>11} {'Jsc_ag':>9} {'Jsc_active':>11} "
            f"{'Voc(V)':>8} {'FF':>7} {'PCE_ag(%)':>10} {'L_col(μm)':>10}")
        add("-" * 92)
        for size in sorted(ag_size_results):
            jv = ag_size_results[size]
            area_ratio = jv.area_ag / jv.area_active
            l_col = jv.collection.get("collection_length", np.nan) * 1e4 if jv.collection else np.nan
            add(f"{size:>10g} {area_ratio:>11.3f} {_fmt(jv.jsc_ag, '.2f'):>9} "
                f"{_fmt(jv.jsc_active, '.2f'):>11} {_fmt(jv.voc, '.3f'):>8} "
                f"{_fmt(jv.ff, '.3f'):>7} {_fmt(jv.pce_ag, '.2f'):>10} {_fmt(l_col, '.0f'):>10}")
        add("")

    if contact_results:
        add("-" * 92)
        add("4) Ag/PEDOT 接触电阻扫描")
        add("-" * 92)
        add(f"{'R_c (Ω·cm^2)':>12} {'Jsc_ag':>9} {'Jsc_active':>11} {'Voc(V)':>8} "
            f"{'FF':>7} {'PCE_ag(%)':>10}")
        add("-" * 92)
        for rc in sorted(contact_results):
            jv = contact_results[rc]
            add(f"{rc:>12g} {_fmt(jv.jsc_ag, '.2f'):>9} {_fmt(jv.jsc_active, '.2f'):>11} "
                f"{_fmt(jv.voc, '.3f'):>8} {_fmt(jv.ff, '.3f'):>7} {_fmt(jv.pce_ag, '.2f'):>10}")
        add("")

    if illumination:
        add("-" * 92)
        add("5) 光照模式对比（同一 σ∥ 下）")
        add("-" * 92)
        add(f"{'模式':<12} {'Jsc_ag':>9} {'Jsc_active':>11} {'Voc(V)':>8} {'FF':>7} {'PCE_ag(%)':>10}")
        add("-" * 92)
        for mode, jv in illumination.items():
            add(f"{mode:<12} {_fmt(jv.jsc_ag, '.2f'):>9} {_fmt(jv.jsc_active, '.2f'):>11} "
                f"{_fmt(jv.voc, '.3f'):>8} {_fmt(jv.ff, '.3f'):>7} {_fmt(jv.pce_ag, '.2f'):>10}")
        if "full" in illumination and "ag_only" in illumination:
            extra = illumination["full"].jsc_ag - illumination["ag_only"].jsc_ag
            add("")
            add(f"  full 与 ag_only 的 Jsc_ag 差值 = {extra:.2f} mA/cm^2:")
            add("  该差值即 Ag 覆盖区外经 PEDOT 横向输运被收集的额外电流贡献。")
        add("")

    if verification:
        add("-" * 92)
        add("6) 工作点验证（短路 / 最大功率点 / 开路）")
        add("-" * 92)
        sc = verification["short_circuit"]
        oc = verification["open_circuit"]
        mpp = verification["mpp"]
        add(f"  短路: I_terminal = {sc['i_terminal_mA']:.4f} mA, "
            f"A_effective = {sc['a_effective_cm2'] * 100:.2f} mm^2 "
            f"(A_effective/A_ag = {sc['a_effective_over_ag']:.2f})")
        add(f"        有效收集长度 L_col = {sc['collection_length_cm'] * 1e4:.0f} μm, "
            f"PEDOT 最大电势抬升 = {sc['v_hump_V'] * 1000:.2f} mV")
        add(f"  MPP : V_mpp = {mpp['vmpp']:.3f} V, I_mpp = {mpp['impp']:.4f} mA, FF = {mpp['ff']:.3f}")
        add(f"  开路: Voc = {oc['voc_V']:.3f} V, |I_terminal| = {abs(oc['i_terminal_mA']):.2e} mA")
        add("  检查项:")
        for name, passed in verification["checks"].items():
            add(f"    [{'通过' if passed else '未通过'}] {name}")
        add("")

    add("=" * 92)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot_jv_comparison(sigma_results: Dict[float, JVResult],
                       material: MaterialProperties,
                       save_path: Optional[str] = None):
    """不同电导率下的 J-V 曲线与特征参数对比（双面积归一化）"""
    sigmas = sorted(sigma_results)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(sigmas)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # 1. J-V 曲线（实线 J_ag，虚线 J_active）
    ax = axes[0, 0]
    for sigma, color in zip(sigmas, colors):
        jv = sigma_results[sigma]
        ax.plot(jv.voltage, jv.j_ag, "-", color=color,
                label=f"σ∥={sigma:g}")
        ax.plot(jv.voltage, jv.j_active, "--", color=color)
    ax.set_xlabel("电压 (V)")
    ax.set_ylabel("电流密度 (mA/cm^2)")
    ax.set_title("终端 J-V 曲线（实线: J_ag, 虚线: J_active）")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # 2. Jsc vs σ
    ax = axes[0, 1]
    ax.semilogx(sigmas, [sigma_results[s].jsc_ag for s in sigmas], "o-",
                label="Jsc_ag (/A_ag)")
    ax.semilogx(sigmas, [sigma_results[s].jsc_active for s in sigmas], "s-",
                label="Jsc_active (/A_active)")
    ax.axhline(material.jph, color="gray", ls=":", label=f"Jph = {material.jph:g}")
    ax.set_xlabel("PEDOT 电导率 σ∥ (S/cm)")
    ax.set_ylabel("Jsc (mA/cm^2)")
    ax.set_title("短路电流密度 vs 电导率")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. FF 与 Voc vs σ
    ax = axes[0, 2]
    ax.semilogx(sigmas, [sigma_results[s].ff for s in sigmas], "o-", color="tab:red")
    ax.set_xlabel("PEDOT 电导率 σ∥ (S/cm)")
    ax.set_ylabel("填充因子 FF", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.semilogx(sigmas, [sigma_results[s].voc for s in sigmas], "s--", color="tab:blue")
    ax2.set_ylabel("Voc (V)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title("FF 与 Voc vs 电导率（不依赖面积定义）")
    ax.grid(alpha=0.3)

    # 4. PCE vs σ
    ax = axes[1, 0]
    ax.semilogx(sigmas, [sigma_results[s].pce_ag for s in sigmas], "o-",
                label="PCE_ag (/A_ag)")
    ax.semilogx(sigmas, [sigma_results[s].pce_active for s in sigmas], "s-",
                label="PCE_active (/A_active)")
    ax.set_xlabel("PEDOT 电导率 σ∥ (S/cm)")
    ax.set_ylabel("PCE (%)")
    ax.set_title("光电转换效率 vs 电导率（两种面积归一化）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5. P-V 曲线
    ax = axes[1, 1]
    for sigma, color in zip(sigmas, colors):
        jv = sigma_results[sigma]
        ax.plot(jv.voltage, jv.voltage * jv.current, "-", color=color,
                label=f"σ∥={sigma:g}")
        ax.plot(jv.vmpp, jv.pmpp, "r*", markersize=9)
    ax.set_xlabel("电压 (V)")
    ax.set_ylabel("功率 (mW)")
    ax.set_title("P-V 曲线（红星为最大功率点）")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 6. A_eff/A_ag 与 L_col vs σ
    ax = axes[1, 2]
    a_ratio = [sigma_results[s].collection.get("a_effective", np.nan)
               / sigma_results[s].area_ag for s in sigmas]
    l_col = [sigma_results[s].collection.get("collection_length", np.nan) * 1e4
             for s in sigmas]
    ax.semilogx(sigmas, a_ratio, "o-", color="tab:green", label="A_effective / A_ag (V=0)")
    ax.set_xlabel("PEDOT 电导率 σ∥ (S/cm)")
    ax.set_ylabel("A_effective / A_ag", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax2 = ax.twinx()
    ax2.semilogx(sigmas, l_col, "s--", color="tab:purple",
                 label="L_col (μm)")
    ax2.set_ylabel("有效收集长度 L_col (μm)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("有效收集面积与收集长度 vs 电导率")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"对比图已保存至: {save_path}")
    return fig


def export_conductivity_scan_csv(
        geometry: DeviceGeometry,
        sigma_results: Dict[float, JVResult],
        save_path: str,
        ag_only_jsc_ag: Optional[float] = None,
        material_labels: Optional[Dict[float, str]] = None,
) -> None:
    """导出 30 nm 电导率扫描的可复用数值表。

    ``Jsc_extra_ag`` 以“仅 Ag 覆盖区受光”的 Jsc 为参照，因此只计入
    Ag 外区域经横向 PEDOT 收集到的额外电流；它不是材料本征 Jph 的提升。
    """
    fieldnames = [
        "material", "pedot_thickness_nm", "sigma_parallel_S_per_cm",
        "r_sheet_ohm_per_sq", "jsc_ag_mA_per_cm2", "jsc_active_mA_per_cm2",
        "jsc_extra_ag_mA_per_cm2", "jsc_extra_over_ag_only_percent",
        "voc_V", "vmpp_V", "impp_mA", "ff", "pce_ag_percent",
        "pce_active_percent", "a_eff_over_a_ag_sc", "l_col_sc_um",
        "a_eff_over_a_ag_mpp", "l_col_mpp_um",
    ]
    labels = material_labels or {}
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sigma in sorted(sigma_results):
            jv = sigma_results[sigma]
            sc = jv.collection
            mpp = jv.mpp_collection
            extra = (jv.jsc_ag - ag_only_jsc_ag
                     if ag_only_jsc_ag is not None else np.nan)
            writer.writerow({
                "material": labels.get(sigma, ""),
                "pedot_thickness_nm": geometry.pedot_thickness,
                "sigma_parallel_S_per_cm": sigma,
                "r_sheet_ohm_per_sq": 1.0 / (sigma * geometry.pedot_thickness * 1e-7),
                "jsc_ag_mA_per_cm2": jv.jsc_ag,
                "jsc_active_mA_per_cm2": jv.jsc_active,
                "jsc_extra_ag_mA_per_cm2": extra,
                "jsc_extra_over_ag_only_percent": (
                    100.0 * extra / ag_only_jsc_ag
                    if ag_only_jsc_ag and ag_only_jsc_ag > 0 else np.nan
                ),
                "voc_V": jv.voc,
                "vmpp_V": jv.vmpp,
                "impp_mA": jv.impp,
                "ff": jv.ff,
                "pce_ag_percent": jv.pce_ag,
                "pce_active_percent": jv.pce_active,
                "a_eff_over_a_ag_sc": sc.get("a_effective", np.nan) / jv.area_ag,
                "l_col_sc_um": sc.get("collection_length", np.nan) * 1e4,
                "a_eff_over_a_ag_mpp": mpp.get("a_effective", np.nan) / jv.area_ag,
                "l_col_mpp_um": mpp.get("collection_length", np.nan) * 1e4,
            })


def plot_conductivity_summary(
        sigma_results: Dict[float, JVResult],
        ag_only_jsc_ag: float,
        material_labels: Optional[Dict[float, str]] = None,
        save_path: Optional[str] = None,
):
    """绘制适用于固定 30 nm HTL 的电导率扫描总览图。

    同时呈现短路和 MPP 下的收集诊断，并明确以仅 Ag 区受光的结果为
    Ag 外额外电流基线，避免把面积扩展误读为本征光伏性能提升。
    """
    sigmas = np.array(sorted(sigma_results), dtype=float)
    jvs = [sigma_results[s] for s in sigmas]
    labels = material_labels or {}

    def series(name, collection_name="collection"):
        return np.array([
            getattr(jv, collection_name).get(name, np.nan) for jv in jvs
        ])

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10), layout="constrained")
    fig.suptitle("30 nm PEDOT HTL：横向电导率完整扫描", fontsize=16)

    ax = axes[0, 0]
    ax.semilogx(sigmas, [jv.jsc_ag for jv in jvs], "o-", label=r"$J_{sc,Ag}$")
    ax.semilogx(sigmas, [jv.jsc_active for jv in jvs], "s--", label=r"$J_{sc,active}$")
    ax.axhline(ag_only_jsc_ag, color="0.35", ls=":", label="仅 Ag 区受光基线")
    ax.set(xlabel=r"$\sigma_\parallel$ (S/cm)", ylabel=r"$J_{sc}$ (mA/cm²)",
           title="短路电流：两种面积归一化")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    extra = np.array([jv.jsc_ag for jv in jvs]) - ag_only_jsc_ag
    ax.semilogx(sigmas, extra, "o-", color="tab:purple")
    ax.axhline(0.0, color="0.35", lw=0.9)
    ax.set(xlabel=r"$\sigma_\parallel$ (S/cm)",
           ylabel=r"$\Delta J_{sc,Ag}$ (mA/cm²)",
           title="Ag 外横向收集的额外短路电流")
    ax.grid(alpha=0.28)

    ax = axes[0, 2]
    ax.semilogx(sigmas, [jv.ff for jv in jvs], "o-", color="tab:red", label="FF")
    ax.set(xlabel=r"$\sigma_\parallel$ (S/cm)", ylabel="FF", title="填充因子与开路电压变化")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    voc0 = jvs[0].voc
    ax2.semilogx(sigmas, [(jv.voc - voc0) * 1e3 for jv in jvs], "s--",
                 color="tab:blue", label="ΔVoc")
    ax2.set_ylabel("ΔVoc relative to lowest σ (mV)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.28)

    ax = axes[1, 0]
    ax.semilogx(sigmas, [jv.pce_ag for jv in jvs], "o-", label="PCE_ag")
    ax.semilogx(sigmas, [jv.pce_active for jv in jvs], "s--", label="PCE_active")
    ax.set(xlabel=r"$\sigma_\parallel$ (S/cm)", ylabel="PCE (%)",
           title="最大功率：两种面积归一化")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sc_ratio = series("a_effective") / np.array([jv.area_ag for jv in jvs])
    mpp_ratio = series("a_effective", "mpp_collection") / np.array([jv.area_ag for jv in jvs])
    ax.semilogx(sigmas, sc_ratio, "o-", label="短路 V = 0")
    ax.semilogx(sigmas, mpp_ratio, "s--", label="最大功率点")
    ax.set(xlabel=r"$\sigma_\parallel$ (S/cm)", ylabel=r"$A_{eff}/A_{Ag}$",
           title="有效收集面积")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.semilogx(sigmas, series("collection_length") * 1e4, "o-", label="短路 V = 0")
    ax.semilogx(sigmas, series("collection_length", "mpp_collection") * 1e4,
                "s--", label="最大功率点")
    ax.set(xlabel=r"$\sigma_\parallel$ (S/cm)", ylabel=r"$L_{col}$ (μm)",
           title="Ag 边缘外的等效收集长度")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8)

    # 实测材料点仅作定位标签，计算曲线本身使用完整对数扫描点。
    for index, (sigma, name) in enumerate(labels.items()):
        if sigma in sigma_results:
            axes[0, 1].annotate(name, (sigma, sigma_results[sigma].jsc_ag - ag_only_jsc_ag),
                                xytext=(3, 7 + 12 * index), textcoords="offset points",
                                fontsize=6.5, rotation=20)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"30 nm 电导率扫描总览图已保存至: {save_path}")
    return fig


def plot_distributions(solver: PEDOTNetworkSolver, save_path: Optional[str] = None):
    """
    电势、横向电流密度、焦耳热与局部收集效率分布图（须先调用 solve）
    """
    V = solver.potential
    jx, jy = solver.current_density_map()
    j_mag = np.sqrt(jx ** 2 + jy ** 2) * 1000.0          # mA/cm^2
    p_density, p_total = solver.joule_heating_map()      # W/cm^2
    coll = solver.collection_efficiency_map()
    eta = coll["eta_map"]

    X_mm, Y_mm = solver.X / 1000.0, solver.Y / 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    def _panel(ax, data, cmap, title, cbar_label, levels=60):
        im = ax.contourf(X_mm, Y_mm, data, levels=levels, cmap=cmap)
        ax.contour(X_mm, Y_mm, solver.ag_mask.astype(float), levels=[0.5],
                   colors="k", linewidths=1.2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label=cbar_label)

    _panel(axes[0, 0], V, "RdYlBu_r",
           f"PEDOT 电势 V(x,y)（V_ext = {solver.v_ext:.2f} V）", "电势 (V)")
    _panel(axes[0, 1], j_mag, "hot", "横向电流密度 |J|（PEDOT 膜内）", "|J| (mA/cm^2)")

    ax = axes[0, 2]
    step = max(1, solver.nx // 25)
    ax.quiver(X_mm[::step, ::step], Y_mm[::step, ::step],
              jx[::step, ::step], jy[::step, ::step], j_mag[::step, ::step],
              cmap="viridis")
    ax.contour(X_mm, Y_mm, solver.ag_mask.astype(float), levels=[0.5],
               colors="k", linewidths=1.2)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("横向电流矢量场（箭头指向电流方向）")
    ax.set_aspect("equal")

    _panel(axes[1, 0], p_density, "YlOrRd",
           f"焦耳热分布（总损耗 {p_total * 1000:.3f} mW）", "功率密度 (W/cm^2)")

    eta_plot = eta
    im = axes[1, 1].contourf(X_mm, Y_mm, eta_plot, levels=np.linspace(0, 1, 21),
                             cmap="RdYlGn")
    axes[1, 1].contour(X_mm, Y_mm, solver.ag_mask.astype(float), levels=[0.5],
                       colors="k", linewidths=1.2)
    axes[1, 1].set_xlabel("x (mm)")
    axes[1, 1].set_ylabel("y (mm)")
    axes[1, 1].set_title(f"局部收集效率 η = J_local/Jph\n"
                         f"A_effective = {coll['a_effective'] * 100:.2f} mm^2, "
                         f"L_col = {coll['collection_length'] * 1e4:.0f} μm")
    axes[1, 1].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 1], label="η")

    ax = axes[1, 2]
    i_center = solver.ny // 2
    ax.plot(X_mm[i_center, :], eta_plot[i_center, :], "o-", markersize=3,
            color="tab:blue", label="η 沿中心线")
    ax.axvspan((solver.geometry.ag_center_x - solver.geometry.ag_width / 2) / 1000,
               (solver.geometry.ag_center_x + solver.geometry.ag_width / 2) / 1000,
               color="gray", alpha=0.25, label="Ag 覆盖区")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("η")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("收集效率沿 y = 器件中心的横向衰减")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"分布图已保存至: {save_path}")
    return fig


def plot_readability_dashboard(solver: PEDOTNetworkSolver, jv: JVResult,
                               save_path: Optional[str] = None,
                               jv_display_max: float = 0.95,
                               comparison_jv: Optional[JVResult] = None,
                               comparison_label: str = "对比器件"):
    """
    生成一页式结果总览，便于从空间分布直接读出终端 J-V 的来源。

    调用前应将 ``solver`` 求解在感兴趣的工作点；通常使用 ``jv.vmpp``。
    最后一幅图把 Ag 覆盖区外的网格按“到最近 Ag 边缘的距离”分箱，
    同时展示局部收集效率及其对 Ag 外额外电流的累计贡献。
    """
    if solver.potential is None or solver.v_ext is None:
        raise RuntimeError("请先调用 solver.solve()，再绘制结果总览")

    V = solver.potential
    jx, jy = solver.current_density_map()
    j_lateral = np.hypot(jx, jy) * 1000.0  # mA/cm^2
    collection = solver.collection_efficiency_map()
    eta = collection["eta_map"]
    j_vertical = solver.j_local(V, solver.jph_map)  # A/cm^2

    # 对 Ag 外区域，distance_transform_edt(~mask) 给出到最近 Ag 节点的距离。
    # dx、dy 的单位是 cm；图中统一换算为 μm。
    edge_distance_um = distance_transform_edt(
        ~solver.ag_mask, sampling=(solver.dy, solver.dx)
    ) * 1e4

    X_mm, Y_mm = solver.X / 1000.0, solver.Y / 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "局部 Ag 顶电极器件：空间收集与终端 J–V 总览"
        f"（空间图对应 V = {solver.v_ext:.3f} V）",
        fontsize=15, y=0.98,
    )

    def _outline_ag(ax):
        ax.contour(X_mm, Y_mm, solver.ag_mask.astype(float), levels=[0.5],
                   colors="black", linewidths=1.2)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")

    # 1. 几何与面积定义
    ax = axes[0, 0]
    ax.contourf(X_mm, Y_mm, solver.ag_mask.astype(int), levels=[-0.5, 0.5, 1.5],
                colors=["#fff4d6", "#8e8e8e"])
    _outline_ag(ax)
    ax.set_title("Ag 掩模板与连续活性区")
    ax.text(
        0.03, 0.03,
        f"A_ag = {solver.geometry.ag_area * 100:.2f} mm²\n"
        f"A_active = {solver.geometry.active_area * 100:.2f} mm²\n"
        f"A_eff(V) = {collection['a_effective'] * 100:.2f} mm²",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    # 2. PEDOT 电势
    ax = axes[0, 1]
    im = ax.contourf(X_mm, Y_mm, V, levels=50, cmap="RdYlBu_r")
    _outline_ag(ax)
    ax.set_title("PEDOT 电势分布")
    plt.colorbar(im, ax=ax, label="V_PEDOT (V)")

    # 3. 横向电流。对数色阶可以同时看清 Ag 边缘的电流拥挤和远端弱电流。
    ax = axes[0, 2]
    # Ag 区为等势接触，PEDOT 面内 J∥ 接近零；用灰色而非对数下限色表示，
    # 避免将其误读为极大的电流损失区。
    log_j = np.ma.masked_where(solver.ag_mask, np.log10(np.maximum(j_lateral, 1e-8)))
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#9a9a9a")
    im = ax.contourf(X_mm, Y_mm, log_j, levels=50, cmap=cmap)
    _outline_ag(ax)
    ax.set_title("PEDOT 横向电流密度（对数）")
    plt.colorbar(im, ax=ax, label="log10 |J_parallel| (mA/cm²)")

    # 4. 局部收集效率
    ax = axes[1, 0]
    im = ax.contourf(X_mm, Y_mm, eta, levels=np.linspace(0, 1, 21),
                     cmap="RdYlGn", extend="both")
    _outline_ag(ax)
    ax.set_title("局部收集效率 η = J_local / J_ph")
    plt.colorbar(im, ax=ax, label="η")

    # 5. 终端 J-V 统一按局部 Ag 电极的名义面积归一化。可叠加一个同几何、
    # 同照光条件下的材料对比曲线。数值求解范围可高于 Voc；仅裁剪绘图窗口，
    # 避免高压指数电流拉大纵轴而掩盖第一象限。
    ax = axes[1, 1]
    display = jv.voltage <= jv_display_max + 1e-12
    if np.count_nonzero(display) < 2:
        raise ValueError("J-V 显示上限内至少需要两个电压点")
    ax.plot(jv.voltage[display], jv.j_ag[display], color="tab:red", lw=2.2,
            label="核心算例：J_ag = I / A_ag")
    if comparison_jv is not None:
        comparison_display = comparison_jv.voltage <= jv_display_max + 1e-12
        ax.plot(comparison_jv.voltage[comparison_display],
                comparison_jv.j_ag[comparison_display], color="tab:blue",
                lw=2.0, ls="--", label=f"{comparison_label}：J_ag = I / A_ag")
    ax.axvline(jv.vmpp, color="gray", ls=":", lw=1.2)
    ax.plot(jv.vmpp, jv.impp / jv.area_ag, "o", color="tab:red", ms=6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("终端电压 (V)")
    ax.set_ylabel("电流密度 (mA/cm²)")
    ax.set_title(f"终端 J–V（按 Ag 面积归一化）：Voc={jv.voc:.3f} V，FF={jv.ff:.3f}")
    ax.set_xlim(0.0, jv_display_max)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 6. Ag 边缘外的收集衰减和累计电流贡献。
    ax = axes[1, 2]
    outside = (~solver.ag_mask) & (solver.jph_map > 0)
    d = edge_distance_um[outside]
    eta_outside = eta[outside]
    i_outside = j_vertical[outside] * solver.dA  # A / 网格点
    if d.size and np.any(np.isfinite(eta_outside)):
        n_bins = min(30, max(8, int(np.sqrt(d.size))))
        bins = np.linspace(0.0, d.max(), n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2.0
        which = np.digitize(d, bins) - 1
        eta_mean = np.full(n_bins, np.nan)
        for k in range(n_bins):
            values = eta_outside[which == k]
            if values.size:
                eta_mean[k] = np.nanmean(values)
        ax.plot(centers, eta_mean, "o-", color="tab:green", ms=3,
                label="Ag 外平均 η")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="η = 0.5")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("距最近 Ag 边缘的距离 (μm)")
        ax.set_ylabel("局部收集效率 η", color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")
        ax.grid(alpha=0.3)

        # 距 Ag 最近的区域排在前，累计曲线表示 Ag 外电流贡献的空间来源。
        order = np.argsort(d)
        i_total = float(np.sum(i_outside[order]))
        if abs(i_total) > 1e-20:
            cumulative = np.cumsum(i_outside[order]) / i_total
            ax2 = ax.twinx()
            ax2.plot(d[order], cumulative, color="tab:purple", alpha=0.8,
                     lw=1.5, label="Ag 外累计电流占比")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Ag 外累计电流占比", color="tab:purple")
            ax2.tick_params(axis="y", labelcolor="tab:purple")
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles + handles2, labels + labels2, fontsize=7,
                      loc="lower left")
        else:
            ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "无 Ag 外受光网格", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("Ag 外区域：收集衰减与累计贡献")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"结果总览图已保存至: {save_path}")
    return fig


def plot_spatial_conductivity_scan(
        geometry: DeviceGeometry,
        material: MaterialProperties,
        X: np.ndarray,
        Y: np.ndarray,
        ag_mask: np.ndarray,
        sigma_results: Dict[float, JVResult],
        sigma_values: Optional[Sequence[float]] = None,
        illumination: str = "full",
        save_path: Optional[str] = None,
        title: Optional[str] = None,
):
    """
    绘制 PEDOT 横向电导率的二维空间扫描图谱。

    每一列对应一个电导率，并使用其自身的最大功率点电压。三行依次为：
    1) 局部收集效率；2) 相对 Ag 端子的 PEDOT 横向压降；3) 横向电流密度。
    同一行统一色标，因此可直接比较不同电导率下的收集范围和电流拥挤。
    """
    if not sigma_results:
        raise ValueError("sigma_results 不能为空；请先运行 run_conductivity_scan")

    available = np.array(sorted(sigma_results), dtype=float)
    if sigma_values is None:
        # 默认选取覆盖低导、中间过渡与高导区的最多 6 个代表点。
        target = np.array([3e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0])
        selected = []
        for value in target:
            nearest = float(available[np.argmin(np.abs(np.log10(available) - np.log10(value)))])
            if nearest not in selected:
                selected.append(nearest)
        sigma_values = selected
    else:
        sigma_values = [float(value) for value in sigma_values]
        missing = [value for value in sigma_values if value not in sigma_results]
        if missing:
            raise ValueError(f"以下电导率不在扫描结果中: {missing}")

    snapshots = []
    for sigma in sigma_values:
        jv = sigma_results[float(sigma)]
        solver = PEDOTNetworkSolver(
            X, Y, geometry,
            replace(material, pedot_conductivity_lateral=float(sigma)),
            ag_mask, illumination=illumination,
        )
        solver.solve(jv.vmpp)
        eta = solver.collection_efficiency_map()["eta_map"]
        jx, jy = solver.current_density_map()
        j_lateral = np.hypot(jx, jy) * 1000.0  # mA/cm^2
        snapshots.append({
            "sigma": float(sigma),
            "vmpp": jv.vmpp,
            "ff": jv.ff,
            "eta": eta,
            "voltage_drop": solver.potential - jv.vmpp,
            "j_lateral": j_lateral,
        })

    # 公共色标：避免每个子图自动缩放而产生视觉误导。
    voltage_max = max(float(np.nanmax(item["voltage_drop"])) for item in snapshots)
    voltage_max = max(voltage_max, 1e-6)
    j_values = np.concatenate([
        item["j_lateral"][~ag_mask].ravel() for item in snapshots
    ])
    log_j = np.log10(np.maximum(j_values, 1e-12))
    log_j_min, log_j_max = np.percentile(log_j, [1, 99])
    if log_j_max <= log_j_min:
        log_j_max = log_j_min + 1.0

    n_cols = len(snapshots)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.0 * n_cols, 10.5),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    fig.suptitle(title or "PEDOT 横向电导率二维空间扫描（各列均取各自 MPP）",
                 fontsize=15)
    X_mm, Y_mm = X / 1000.0, Y / 1000.0
    eta_levels = np.linspace(0.0, 1.0, 21)
    voltage_levels = np.linspace(0.0, voltage_max, 31)
    current_levels = np.linspace(log_j_min, log_j_max, 31)
    cmap_current = plt.colormaps["magma"].copy()
    cmap_current.set_bad("#9a9a9a")
    row_images = [None, None, None]

    for col, item in enumerate(snapshots):
        sigma = item["sigma"]
        r_sheet = 1.0 / (sigma * geometry.pedot_thickness * 1e-7)
        titles = (
            f"σ∥ = {sigma:g} S/cm\nR_sheet = {r_sheet:.2g} Ω/□\n"
            f"V_mpp = {item['vmpp']:.3f} V, FF = {item['ff']:.3f}"
        )

        data_rows = (
            (item["eta"], eta_levels, "RdYlGn", "局部收集效率 η"),
            (item["voltage_drop"], voltage_levels, "YlOrRd", "横向压降 V_PEDOT − V_MPP (V)"),
            (np.ma.masked_where(ag_mask, np.log10(np.maximum(item["j_lateral"], 1e-12))),
             current_levels, cmap_current, "log10 |J_parallel| (mA/cm²)"),
        )
        for row, (data, levels, cmap, row_label) in enumerate(data_rows):
            ax = axes[row, col]
            image = ax.contourf(X_mm, Y_mm, data, levels=levels, cmap=cmap,
                                extend="both" if row in (0, 2) else "neither")
            ax.contour(X_mm, Y_mm, ag_mask.astype(float), levels=[0.5],
                       colors="black", linewidths=0.9)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(titles, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{row_label}\ny (mm)")
            else:
                ax.set_ylabel("")
            if row == 2:
                ax.set_xlabel("x (mm)")
            row_images[row] = image

    fig.colorbar(row_images[0], ax=axes[0, :].tolist(), shrink=0.88,
                 pad=0.015, label="η")
    fig.colorbar(row_images[1], ax=axes[1, :].tolist(), shrink=0.88,
                 pad=0.015, label="V_PEDOT − V_MPP (V)")
    fig.colorbar(row_images[2], ax=axes[2, :].tolist(), shrink=0.88,
                 pad=0.015, label="log10 |J_parallel| (mA/cm²)")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"电导率二维空间扫描图已保存至: {save_path}")
    return fig


def plot_illumination_comparison(illumination: Dict[str, JVResult],
                                 save_path: Optional[str] = None):
    """对比“整个活性区均匀受光”与“仅 Ag 覆盖区受光”的 J-V 与 Jsc"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"full": ("tab:red", "-"), "ag_only": ("tab:blue", "--")}
    names = {"full": "整个活性区均匀受光", "ag_only": "仅 Ag 覆盖区受光"}

    ax = axes[0]
    for mode, jv in illumination.items():
        color, ls = styles[mode]
        ax.plot(jv.voltage, jv.j_ag, ls, color=color, label=f"{names[mode]} (J_ag)")
        ax.plot(jv.voltage, jv.j_active, ls, color=color, alpha=0.45,
                label=f"{names[mode]} (J_active)")
    ax.set_xlabel("电压 (V)")
    ax.set_ylabel("电流密度 (mA/cm^2)")
    ax.set_title("J-V 曲线对比（实色: J_ag, 淡色: J_active）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    modes = list(illumination)
    x = np.arange(len(modes))
    width = 0.35
    ax.bar(x - width / 2, [illumination[m].jsc_ag for m in modes], width,
           label="Jsc_ag (/A_ag)", color="tab:red", alpha=0.8)
    ax.bar(x + width / 2, [illumination[m].jsc_active for m in modes], width,
           label="Jsc_active (/A_active)", color="tab:blue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in modes], fontsize=9)
    ax.set_ylabel("Jsc (mA/cm^2)")
    ax.set_title("短路电流密度对比: Ag 外区域的额外电流贡献")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"光照模式对比图已保存至: {save_path}")
    return fig
