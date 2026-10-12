"""
主运行脚本 - PEDOT 横向电导率对局部 Ag 顶电极有机太阳能电池的影响

按 README 的实施顺序编排:
    1. 定义实际器件尺寸、活性层范围和 Ag 掩模板区域；
    2. 实现局部单二极管 J-V 与二维 PEDOT 电阻网络的耦合求解；
    3. 在短路、最大功率点和开路条件下验证电势与电流分布；
    4. 在固定 30 nm HTL 厚度下完整扫描 PEDOT 横向电导率；
    5. 分别报告短路和最大功率点的空间收集，并量化 Ag 外额外电流。
"""

import os
import sys
import time
import numpy as np
import matplotlib

sys.path.insert(0, os.path.dirname(__file__))

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

from models.device_geometry import (  # noqa: E402
    DeviceGeometry, MaterialProperties, create_2d_mesh, create_ag_mask,
    visualize_geometry,
)
from simulation.network_solver import PEDOTNetworkSolver  # noqa: E402
from analysis.device_performance import (  # noqa: E402
    compute_jv, verify_operating_points,
    run_conductivity_scan, compare_illumination,
    generate_report, plot_jv_comparison,
    plot_illumination_comparison, plot_readability_dashboard,
    plot_spatial_conductivity_scan, plot_conductivity_summary,
    export_conductivity_scan_csv,
)


def main():
    t_start = time.time()
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 78)
    print("PEDOT 横向电导率对局部 Ag 顶电极有机太阳能电池的影响 - 二维分布式微电池网络仿真")
    print("=" * 78)

    # ==================================================================
    # 步骤 1: 器件尺寸、活性层范围与 Ag 掩模板区域
    # ==================================================================
    print("\n步骤 1: 定义器件几何与 Ag 掩模板区域")

    geometry = DeviceGeometry(
        device_width=5000.0,      # 活性区 5 mm × 5 mm
        device_length=5000.0,
        pedot_thickness=30.0,     # 固定为实际采用的 30 nm HTL
        ag_shape="rectangle",     # 阴影掩模板定义的连续矩形 Ag 区域
        ag_width=2000.0,          # 2 mm × 2 mm
        ag_length=2000.0,
    )
    # 40 节点/mm，即 25 μm 网格。用于避免将数十微米的收集长度直接
    # 量化为一个 50 μm 网格；仍需把 <25 μm 的结果视作数值分辨率上限。
    mesh_density = 40
    X, Y = create_2d_mesh(geometry, mesh_density=mesh_density)
    ag_mask = create_ag_mask(X, Y, geometry)

    # ---- 本征微电池标定：Jsc = 25 mA/cm², Voc = 0.900 V, FF = 0.770 ----
    # 选定 n = 1.5 后，以 Rsh 调节 FF，再由 Voc 方程反求 J0。当前节点型
    # 网格中 Ag 边界节点的积分面积比名义 2 mm × 2 mm 略大，因此将 Jph
    # 按该面积因子归一，使 ag_only 的 Jsc_ag 恰为实验名义的 25 mA/cm²。
    target_jsc_ag = 25.0       # mA/cm²
    target_voc = 0.900         # V
    target_ff = 0.770
    diode_ideality = 1.5
    # 包含 Ag 外暗区通过 PEDOT 的微弱并联泄漏后，ag_only 的 FF = 0.770。
    rsh_calibrated = 466.15  # Ω·cm²；n=1.5、Voc=0.9 V 的网络标定值
    dx_cm = float(X[0, 1] - X[0, 0]) * 1e-4
    dy_cm = float(Y[1, 0] - Y[0, 0]) * 1e-4
    ag_area_factor = np.count_nonzero(ag_mask) * dx_cm * dy_cm / geometry.ag_area
    jph_calibrated = target_jsc_ag / ag_area_factor
    vt = MaterialProperties(temperature=300.0).thermal_voltage
    j0_calibrated = (
        jph_calibrated - 1000.0 * target_voc / rsh_calibrated
    ) / (np.exp(target_voc / (diode_ideality * vt)) - 1.0)

    material = MaterialProperties(
        pedot_conductivity_lateral=2.3e-3,  # 实测 PEDOT:PSS AI 4083 σ∥ (S/cm)
        jph=jph_calibrated,
        j0=j0_calibrated,
        diode_ideality=diode_ideality,
        rsh=rsh_calibrated,
        contact_resistance_ag=0.0,          # 0 → Ag 区等势（Dirichlet）
    )

    print(f"  活性区: {geometry.device_width / 1000:.1f} mm × {geometry.device_length / 1000:.1f} mm"
          f"  (A_active = {geometry.active_area * 100:.2f} mm^2)")
    print(f"  Ag 掩模板: {geometry.ag_shape} {geometry.ag_width / 1000:.1f} × "
          f"{geometry.ag_length / 1000:.1f} mm  (A_ag = {geometry.ag_area * 100:.2f} mm^2)")
    print(f"  PEDOT: σ∥ = {material.pedot_conductivity_lateral:g} S/cm, "
          f"t = {geometry.pedot_thickness:g} nm")
    print(f"  本征微电池标定目标: Jsc = {target_jsc_ag:.1f} mA/cm^2, "
          f"Voc = {target_voc:.3f} V, FF = {target_ff:.3f}"
          f"  (Jph = {material.jph:.4f} mA/cm^2, J0 = {material.j0:.3e} mA/cm^2)")
    print(f"  网格: {ag_mask.shape[0]} × {ag_mask.shape[1]} 节点，Ag 节点 "
          f"{np.count_nonzero(ag_mask)} 个 ({np.mean(ag_mask) * 100:.1f}%)")

    fig = visualize_geometry(
        geometry, material, save_path=os.path.join(results_dir, "device_geometry.png"))
    plt.close(fig)

    # ==================================================================
    # 步骤 2+3: 耦合求解与短路/最大功率点/开路验证
    # ==================================================================
    print("\n" + "=" * 78)
    print("步骤 2+3: 耦合求解与工作点验证 (σ∥ = "
          f"{material.pedot_conductivity_lateral:g} S/cm)")
    print("=" * 78)

    solver = PEDOTNetworkSolver(X, Y, geometry, material, ag_mask,
                                illumination="full")
    verification = verify_operating_points(solver, v_range=(0.0, 1.10), n_points=51)

    sc = verification["short_circuit"]
    print(f"\n  短路: I_terminal = {sc['i_terminal_mA']:.4f} mA, "
          f"KCL 偏差 = {sc['kcl_error']:.2e}")
    print(f"        A_effective = {sc['a_effective_cm2'] * 100:.2f} mm^2 "
          f"(A_effective/A_ag = {sc['a_effective_over_ag']:.2f}), "
          f"L_col = {sc['collection_length_cm'] * 1e4:.0f} μm")
    print(f"        PEDOT 最大电势抬升 = {sc['v_hump_V'] * 1000:.2f} mV")
    mpp = verification["mpp"]
    print(f"  MPP : V_mpp = {mpp['vmpp']:.3f} V, I_mpp = {mpp['impp']:.4f} mA, "
          f"FF = {mpp['ff']:.3f}")
    oc = verification["open_circuit"]
    print(f"  开路: Voc = {oc['voc_V']:.3f} V, |I_terminal| = {abs(oc['i_terminal_mA']):.2e} mA")
    for name, passed in verification["checks"].items():
        print(f"    [{'通过' if passed else '未通过'}] {name}")

    # 总览图包含 MPP 下的电势、电流、收集效率及终端 J-V，
    # 因此不再额外生成内容重叠的 distributions_mpp.png。
    solver.solve(verification["mpp"]["vmpp"])
    fig = plot_readability_dashboard(
        solver, verification["jv"],
        save_path=os.path.join(results_dir, "results_dashboard_mpp.png"),
    )
    plt.close(fig)

    # ==================================================================
    # 步骤 4: 固定 30 nm 的完整电导率扫描
    # ==================================================================
    print("\n" + "=" * 78)
    print("步骤 4: 30 nm PEDOT HTL 横向电导率完整扫描")
    print("=" * 78)

    # 0.25 decade 对数网格覆盖低导 HTL 到高导情形；将五个实测材料点和
    # 收集阈值点显式并入，确保它们不是插值结果。
    measured_labels = {
        6.8e-4: "PEDOT:F 1:6.56",
        2.3e-3: "AI 4083",
        3.1e-3: "PEDOT:F 1:4.56",
        7.4e-3: "PEDOT:F 1:3.33",
        6.2e-2: "PEDOT:F 1:1.96",
    }
    threshold_values = [1e-1, 3e-1, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0]
    conductivity_values = sorted(set(
        np.logspace(-4, 3, 29).tolist()
        + list(measured_labels) + threshold_values
    ))
    print(f"  电导率点数: {len(conductivity_values)}（{conductivity_values[0]:g}–"
          f"{conductivity_values[-1]:g} S/cm）")
    sigma_results = run_conductivity_scan(
        geometry, material, X, Y, ag_mask, conductivity_values,
        v_range=(0.0, 1.10), n_points=61, verbose=True)

    # 仅 Ag 覆盖区受光的计算给出额外电流的严格零点；对给定局部微电池
    # 参数它几乎不依赖 sigma，但仍以相同几何/数值求解器显式计算。
    print("\n步骤 5: 计算 Ag 外区域的额外电流基线")
    illumination = compare_illumination(geometry, material, X, Y, ag_mask,
                                        v_range=(0.0, 1.10), n_points=61)
    ag_only_jsc_ag = illumination["ag_only"].jsc_ag

    selected_jv_values = [6.8e-4, 2.3e-3, 6.2e-2, 3e-1, 1.0, 10.0, 100.0]
    selected_jv = {sigma: sigma_results[sigma] for sigma in selected_jv_values}
    fig = plot_jv_comparison(
        selected_jv, material,
        save_path=os.path.join(results_dir, "jv_conductivity_comparison.png"))
    plt.close(fig)
    fig = plot_conductivity_summary(
        sigma_results, ag_only_jsc_ag, measured_labels,
        save_path=os.path.join(results_dir, "conductivity_scan_30nm_summary.png"))
    plt.close(fig)
    fig = plot_spatial_conductivity_scan(
        geometry, material, X, Y, ag_mask, sigma_results,
        sigma_values=[6.8e-4, 2.3e-3, 6.2e-2, 3e-1, 1.0, 10.0],
        title="30 nm HTL：实测材料与横向收集阈值（各列均取各自 MPP）",
        save_path=os.path.join(results_dir, "spatial_material_comparison_30nm.png"),
    )
    plt.close(fig)
    export_conductivity_scan_csv(
        geometry, sigma_results,
        save_path=os.path.join(results_dir, "conductivity_scan_30nm.csv"),
        ag_only_jsc_ag=ag_only_jsc_ag,
        material_labels=measured_labels,
    )

    fig = plot_illumination_comparison(
        illumination, save_path=os.path.join(results_dir, "illumination_comparison.png"))
    plt.close(fig)

    # ==================================================================
    # 报告
    # ==================================================================
    report = generate_report(
        geometry, material,
        sigma_results=sigma_results,
        illumination=illumination,
        verification=verification,
    )
    report_path = os.path.join(results_dir, "performance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"报告已保存至: {report_path}")

    print("\n" + "=" * 78)
    print(f"分析完成，用时 {time.time() - t_start:.1f} s。所有结果已保存至 results/ 目录")
    print("=" * 78)

    plt.show()


if __name__ == "__main__":
    main()
