"""
器件几何、Ag 掩模板与材料参数定义

器件结构（倒置器件，自下而上）:
    连续底电极 / ETL（第一版近似为等势，V_bottom = 0）
    光活性层（均匀光生、局部非线性 J-V 微电池）
    PEDOT HTL（有限横向片电导 G_sheet = sigma_parallel * t_PEDOT）
    局部 Ag 顶电极（阴影掩模板定义的连续区域，而非周期栅线）

约定:
    - 平面尺寸单位 μm，层厚单位 nm；
    - 面积在求解器与分析中统一使用 cm^2；
    - 电压以底电极电势 V_bottom = 0 为参考。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Q_ELEMENTARY = 1.602176634e-19   # C
K_BOLTZMANN = 1.380649e-23       # J/K


@dataclass
class DeviceGeometry:
    """器件平面几何与层厚（俯视二维模型）"""

    # 活性层受光范围（连续底电极上的器件面元，即 A_active）
    device_width: float = 5000.0      # μm
    device_length: float = 5000.0     # μm

    # 层厚
    pedot_thickness: float = 100.0    # nm
    active_layer_thickness: float = 300.0   # nm
    etl_thickness: float = 30.0             # nm

    # 阴影掩模板定义的局部 Ag 区域（连续图案）
    ag_shape: str = "rectangle"              # "rectangle" | "circle"
    ag_width: float = 2000.0                 # μm（circle 时为直径）
    ag_length: float = 2000.0                # μm（仅 rectangle 使用）
    ag_center_x: Optional[float] = None      # 缺省居中
    ag_center_y: Optional[float] = None

    def __post_init__(self):
        if self.ag_shape not in ("rectangle", "circle"):
            raise ValueError(f"未知的 Ag 图案类型: {self.ag_shape!r}，可选 'rectangle' 或 'circle'")
        if self.ag_center_x is None:
            self.ag_center_x = self.device_width / 2.0
        if self.ag_center_y is None:
            self.ag_center_y = self.device_length / 2.0
        if self.ag_shape == "rectangle":
            if not (0 < self.ag_width <= self.device_width and 0 < self.ag_length <= self.device_length):
                raise ValueError("Ag 区域尺寸必须位于器件范围内")
        else:
            if not 0 < self.ag_width <= min(self.device_width, self.device_length):
                raise ValueError("Ag 圆形直径必须位于器件范围内")

    # ---- 面积（cm^2）----
    @property
    def active_area(self) -> float:
        """A_active: 连续活性层的受光面积 (cm^2)"""
        return self.device_width * 1e-4 * self.device_length * 1e-4

    @property
    def ag_area(self) -> float:
        """A_ag: Ag 顶电极实际覆盖面积 (cm^2)"""
        if self.ag_shape == "rectangle":
            return self.ag_width * 1e-4 * self.ag_length * 1e-4
        radius = self.ag_width / 2.0 * 1e-4   # cm
        return np.pi * radius ** 2


@dataclass
class MaterialProperties:
    """材料电学参数与局部光伏微电池的单二极管模型参数"""

    # PEDOT 横向电导率 sigma_parallel (S/cm)
    # 未改性 Clevios P VP AI 4083 的代表性横向电导率约为 3×10^-3 S/cm。
    # 更高数值代表经溶剂/酸处理或使用高导 PEDOT:PSS 等级的情形。
    pedot_conductivity_lateral: float = 3.0e-3
    silver_conductivity: float = 6.3e5       # S/cm（仅用于展示，Ag 区按等势/接触处理）

    # 温度
    temperature: float = 300.0               # K

    # 局部单二极管微电池: J_local(V) = Jph - J0*[exp(qV/(n kT)) - 1] - V/Rsh
    jph: float = 12.0                        # 局部光生电流密度 (mA/cm^2)
    j0: float = 4.0e-7                       # 二极管反向饱和电流密度 (mA/cm^2)
    diode_ideality: float = 1.5              # 理想因子 n
    rsh: float = 2000.0                      # 并联电阻 (Ω·cm^2)

    # Ag/PEDOT 垂直接触电阻 (Ω·cm^2)；<=0 表示 Ag 区为等势（Dirichlet 边界）
    contact_resistance_ag: float = 0.0

    @property
    def thermal_voltage(self) -> float:
        """kT/q (V)"""
        return K_BOLTZMANN * self.temperature / Q_ELEMENTARY


def create_2d_mesh(geometry: DeviceGeometry, mesh_density: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建 2D 网格（俯视图，xy 平面）

    Args:
        geometry: 器件几何参数
        mesh_density: 网格密度（每 mm 的网格点数）

    Returns:
        X, Y: 网格坐标 (μm)，形状 (ny, nx)
    """
    nx = max(int(round(geometry.device_width / 1000.0 * mesh_density)) + 1, 5)
    ny = max(int(round(geometry.device_length / 1000.0 * mesh_density)) + 1, 5)
    x = np.linspace(0.0, geometry.device_width, nx)
    y = np.linspace(0.0, geometry.device_length, ny)
    return np.meshgrid(x, y)


def create_ag_mask(X: np.ndarray, Y: np.ndarray, geometry: DeviceGeometry) -> np.ndarray:
    """
    创建阴影掩模板定义的局部 Ag 区域掩膜

    Args:
        X, Y: 网格坐标 (μm)
        geometry: 器件几何参数

    Returns:
        mask: 布尔数组，True 表示 Ag 覆盖区
    """
    cx, cy = geometry.ag_center_x, geometry.ag_center_y
    if geometry.ag_shape == "rectangle":
        mask = (
            (np.abs(X - cx) <= geometry.ag_width / 2.0)
            & (np.abs(Y - cy) <= geometry.ag_length / 2.0)
        )
    else:  # circle
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (geometry.ag_width / 2.0) ** 2
    return mask


def create_conductivity_map(X: np.ndarray, Y: np.ndarray,
                            geometry: DeviceGeometry,
                            material: MaterialProperties,
                            ag_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    PEDOT 面内电导率分布 (S/cm)。

    PEDOT 在整个器件面内连续（含 Ag 覆盖区下方），Ag 仅作为顶接触存在，
    因此横向电导率全场均匀；保留分布形式以便将来扩展图案化 PEDOT。
    """
    return np.full_like(X, material.pedot_conductivity_lateral, dtype=float)


def sheet_conductance(geometry: DeviceGeometry, material: MaterialProperties) -> float:
    """
    PEDOT 片电导 G_sheet = sigma_parallel * t_PEDOT (S，每方块电导)

    对应 R_sheet = 1 / (sigma_parallel * t_PEDOT)。
    """
    return material.pedot_conductivity_lateral * geometry.pedot_thickness * 1e-7


def mask_perimeter(mask: np.ndarray, dx: float, dy: float) -> float:
    """
    由掩膜边界统计区域周长 (cm)

    仅统计掩膜与外部的相邻边界段（掩膜若触及计算域边界，该侧不计入）。

    Args:
        mask: 布尔掩膜
        dx, dy: 网格间距 (cm)
    """
    edge_x = np.count_nonzero(mask[:, 1:] != mask[:, :-1]) * dy
    edge_y = np.count_nonzero(mask[1:, :] != mask[:-1, :]) * dx
    return edge_x + edge_y


def visualize_geometry(geometry: DeviceGeometry, material: MaterialProperties,
                       mesh_density: int = 100, save_path: Optional[str] = None):
    """可视化器件几何：Ag 掩模板图案与 PEDOT 片电导"""
    import matplotlib.pyplot as plt

    X, Y = create_2d_mesh(geometry, mesh_density=mesh_density)
    ag_mask = create_ag_mask(X, Y, geometry)
    g_sheet = sheet_conductance(geometry, material)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 局部 Ag 掩模板图案
    ax = axes[0]
    ax.contourf(X / 1000.0, Y / 1000.0, ag_mask.astype(int), levels=[-0.5, 0.5, 1.5],
                colors=["#f5e6c8", "#808080"])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Ag 掩模板图案（阴影掩模板定义的连续区域）")
    ax.set_aspect("equal")
    ax.annotate(
        f"A_active = {geometry.active_area * 100:.2f} mm^2\n"
        f"A_ag = {geometry.ag_area * 100:.2f} mm^2\n"
        f"覆盖率 = {geometry.ag_area / geometry.active_area * 100:.1f}%",
        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    # PEDOT 片电导（均匀）
    ax = axes[1]
    im = ax.contourf(X / 1000.0, Y / 1000.0, ag_mask.astype(int), levels=[-0.5, 0.5, 1.5],
                     colors=["#cfe8ff", "#f0f0f0"])
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"PEDOT 片电导（均匀）\n"
                 f"σ∥ = {material.pedot_conductivity_lateral:g} S/cm, "
                 f"t = {geometry.pedot_thickness:g} nm")
    ax.set_aspect("equal")
    ax.annotate(
        f"G_sheet = {g_sheet:.3e} S/sq\n"
        f"R_sheet = {1.0 / g_sheet:.3e} Ω/sq",
        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"几何结构图已保存至: {save_path}")
    return fig


if __name__ == "__main__":
    geometry = DeviceGeometry()
    material = MaterialProperties()

    print("器件几何参数:")
    print(f"  器件尺寸: {geometry.device_width / 1000:.1f} mm × {geometry.device_length / 1000:.1f} mm")
    print(f"  PEDOT 厚度: {geometry.pedot_thickness:g} nm")
    print(f"  Ag 图案: {geometry.ag_shape} {geometry.ag_width / 1000:.1f} × {geometry.ag_length / 1000:.1f} mm")
    print(f"  A_active = {geometry.active_area * 100:.2f} mm^2, A_ag = {geometry.ag_area * 100:.2f} mm^2")

    print("\n材料参数:")
    print(f"  PEDOT 横向电导率: {material.pedot_conductivity_lateral:g} S/cm")
    print(f"  Jph = {material.jph:g} mA/cm^2, J0 = {material.j0:g} mA/cm^2, n = {material.diode_ideality:g}")
    print(f"  Rsh = {material.rsh:g} Ω·cm^2, T = {material.temperature:g} K")
    print(f"  Ag/PEDOT 接触电阻: {material.contact_resistance_ag:g} Ω·cm^2")
