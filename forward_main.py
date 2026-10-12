"""运行正向 ITO/PEDOT/活性层/PDINN/Ag 边缘取电的面积扫描。"""

import os
import sys
import matplotlib

sys.path.insert(0, os.path.dirname(__file__))
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from analysis.forward_edge_contact import (
    export_forward_scan, plot_composite_sheet_resistance, plot_composite_sheet_resistance_map, plot_forward_heatmap, plot_forward_potential_comparison,
    plot_forward_size_scan, run_forward_size_scan, solve_forward_edge_case,
)


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results", "forward_edge_contact")
    os.makedirs(results_dir, exist_ok=True)
    # 仅比较普通 AI 4083 与本研究的 2 S/cm PEDOT；其余参数完全相同。
    sigmas = [2.3e-3, 2.0]
    sides_mm = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    results = run_forward_size_scan(sides_mm, sigmas, r_ito=15.0,
                                    thickness_nm=30.0, n_nodes=51)
    export_forward_scan(results, os.path.join(results_dir, "forward_size_scan.csv"))
    plot_composite_sheet_resistance(
        os.path.join(results_dir, "pedot_ito_composite_sheet_resistance.png"),
        r_ito=15.0, thickness_nm=30.0)
    plot_composite_sheet_resistance_map(
        os.path.join(results_dir, "pedot_ito_composite_sheet_resistance_map.png"),
        thickness_nm=30.0)
    plot_forward_size_scan(results, os.path.join(results_dir, "forward_size_scan_summary.png"))
    plot_forward_heatmap(results, os.path.join(results_dir, "forward_size_scan_heatmap.png"))
    # 用 10 mm × 10 mm 样品直接展示普通 4083 与 2 S/cm PEDOT 的差异。
    maps = [
        solve_forward_edge_case(10.0, 2.3e-3, r_ito=15.0, save_map=True),
        solve_forward_edge_case(10.0, 2.0, r_ito=15.0, save_map=True),
    ]
    plot_forward_potential_comparison(
        maps, os.path.join(results_dir, "forward_10mm_potential_comparison.png"))


if __name__ == "__main__":
    main()
