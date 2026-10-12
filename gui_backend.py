"""GUI 的计算与数据接口；本模块不操作窗口或创建图形。"""

from dataclasses import asdict
from threading import Event

import numpy as np

from analysis.device_performance import compute_jv
from analysis.forward_edge_contact import solve_forward_edge_case
from models.device_geometry import DeviceGeometry, MaterialProperties, create_ag_mask
from simulation.network_solver import PEDOTNetworkSolver


def parse_values(text):
    values = [float(value) for value in text.replace("，", ",").replace(";", ",").split(",")]
    if not values or not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("参数必须为有限正数；多个数值请用逗号分隔。")
    return sorted(set(values))


def validate_config(config):
    """统一校验参数，避免无效几何或过大的交互扫描。"""
    for name in ("thickness_nm", "ito", "ag_mm"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} 必须为有限正数。")
    if not 11 <= config["nodes"] <= 301:
        raise ValueError("每边网格节点数应在 11–301 之间。")
    if config["mode"] not in ("forward", "inverted", "mapping"):
        raise ValueError("未知模型。")
    for name in ("sides", "sigmas"):
        if not config[name] or not all(np.isfinite(v) and v > 0 for v in config[name]):
            raise ValueError(f"{name} 必须为非空有限正数列表。")
    if config["mode"] != "mapping" and len(config["sides"]) * len(config["sigmas"]) > 60:
        raise ValueError("交互扫描最多 60 组算例；请减少边长或电导率点数。")
    if config["mode"] == "inverted" and config["ag_mm"] > min(config["sides"]):
        raise ValueError("Ag 边长不能大于最小活性区边长。")
    if config["mode"] == "mapping":
        if len(config["sigmas"]) < 2 or len(config["ito_range"]) != 2:
            raise ValueError("方阻映射需要至少两个电导率端点与两个 ITO 方阻端点。")
        if config["ito_range"][0] == config["ito_range"][1]:
            raise ValueError("ITO 方阻范围的两个端点不能相同。")


def run_cases(config, progress=lambda *args: None, cancel=None):
    validate_config(config)
    cancel = cancel or Event()
    if config["mode"] == "mapping":
        sigma = np.logspace(np.log10(min(config["sigmas"])),
                            np.log10(max(config["sigmas"])), 301)
        ito = np.logspace(np.log10(min(config["ito_range"])),
                          np.log10(max(config["ito_range"])), 201)
        S, R = np.meshgrid(sigma, ito)
        return {"config": config, "sigma": S, "ito": R,
                "resistance": 1 / (1 / R + S * config["thickness_nm"] * 1e-7),
                "rows": [], "cases": [], "cancelled": cancel.is_set()}
    total = len(config["sides"]) * len(config["sigmas"])
    cases, rows = [], []
    for sigma in config["sigmas"]:
        for side in config["sides"]:
            if cancel.is_set():
                return {"config": config, "rows": rows, "cases": cases, "cancelled": True}
            progress(len(rows), total, f"L={side:g} mm, σ={sigma:g} S/cm")
            if config["mode"] == "forward":
                item = solve_forward_edge_case(side, sigma, config["ito"],
                                               config["thickness_nm"], config["nodes"], True)
                row = {key: value for key, value in asdict(item).items()
                       if key not in ("potential", "x_mm", "y_mm", "voltage", "j_ag")}
                case = {"x": item.x_mm, "y": item.y_mm, "potential": item.potential,
                        "voltage": item.voltage, "j_ag": item.j_ag,
                        "side": side, "sigma": sigma, "vmpp": item.vmpp}
            else:
                geometry = DeviceGeometry(device_width=side * 1000, device_length=side * 1000,
                                          pedot_thickness=config["thickness_nm"],
                                          ag_width=config["ag_mm"] * 1000,
                                          ag_length=config["ag_mm"] * 1000)
                X, Y = np.meshgrid(np.linspace(0, side * 1000, config["nodes"]),
                                   np.linspace(0, side * 1000, config["nodes"]))
                mask = create_ag_mask(X, Y, geometry)
                dA = (side * 0.1 / (config["nodes"] - 1)) ** 2
                jph = 25 / (np.count_nonzero(mask) * dA / geometry.ag_area)
                vt = MaterialProperties().thermal_voltage
                j0 = (jph - 1000 * 0.9 / 466.15) / (np.exp(0.9 / (1.5 * vt)) - 1)
                if j0 <= 0:
                    raise ValueError("当前网格与 Ag 尺寸的面积标定无效；请提高网格分辨率。")
                material = MaterialProperties(pedot_conductivity_lateral=sigma,
                                              jph=jph, j0=j0, diode_ideality=1.5,
                                              rsh=466.15, contact_resistance_ag=0)
                solver = PEDOTNetworkSolver(X, Y, geometry, material, mask,
                                            illumination=config["illumination"])
                jv = compute_jv(solver, n_points=61, verbose=False)
                solver.solve(jv.vmpp, v0=solver._vu)
                row = {"side_mm": side, "area_mm2": side ** 2, "sigma_pedot": sigma,
                       "ag_area_mm2": geometry.ag_area * 100,
                       "r_pedot": 1 / (sigma * config["thickness_nm"] * 1e-7),
                       "jsc": jv.jsc_ag, "voc": jv.voc, "vmpp": jv.vmpp,
                       "ff": jv.ff, "pce": jv.pce_ag, "jsc_active": jv.jsc_active,
                       "pce_active": jv.pce_active, "mesh_spacing_um": side * 1000 / (config["nodes"] - 1)}
                case = {"x": X / 1000, "y": Y / 1000, "potential": solver.potential.copy(),
                        "voltage": jv.voltage, "j_ag": jv.j_ag,
                        "side": side, "sigma": sigma, "vmpp": jv.vmpp}
            rows.append(row)
            cases.append(case)
            progress(len(rows), total, f"PCE={row['pce']:.4f}% / FF={row['ff']:.4f}")
    return {"config": config, "rows": rows, "cases": cases, "cancelled": False}
