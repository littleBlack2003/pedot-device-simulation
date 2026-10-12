"""
二维分布式微电池网络求解器

物理模型（对应 README 的建模思路）:
    - 器件平面离散为网格，每个网格点具有局部 PEDOT 电势 V_pedot(x, y)；
    - 相邻网格点之间通过 PEDOT 片电导 G_sheet = sigma_parallel * t_PEDOT 相连；
    - 每个网格单元是一个垂直方向的光伏微电池（局部单二极管模型）:

          J_local(V) = Jph - J0*[exp(qV / (n kT)) - 1] - V / Rsh,
          V = V_pedot - V_bottom,  V_bottom = 0（底电极等势）；

    - 节点电流连续性（有限体积离散）:

          Σ_nb G_ij (V_i - V_j) - J_local(V_i) * dA = 0,

      即连续性方程 ∇·(G_sheet ∇V) + J_local(V) = 0 的离散形式；

    - Ag 掩模板区域: 接触电阻为 0 时施加顶电极电势（Dirichlet），
      否则通过 Ag/PEDOT 垂直接触电阻 g_c = 1/R_c 汇入外电路；
    - PEDOT 自由边界为零法向横向电流（自然 Neumann 边界）。

非线性方程组采用阻尼 Newton 迭代求解，Jacobian 为对称正定 M-矩阵。
单位约定: 电导 S，电流密度 A/cm^2，节点面积 dA (cm^2)，输出电流 mA。
"""

import os
import sys
import warnings
from typing import Dict, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.device_geometry import (
    DeviceGeometry, MaterialProperties, mask_perimeter, sheet_conductance,
)


class PEDOTNetworkSolver:
    """横向 PEDOT 电阻网络 + 局部非线性光伏二极管的自洽求解器"""

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 geometry: DeviceGeometry,
                 material: MaterialProperties,
                 ag_mask: np.ndarray,
                 illumination: str = "full"):
        """
        Args:
            X, Y: 网格坐标 (μm)，形状 (ny, nx)
            geometry: 器件几何
            material: 材料与微电池参数
            ag_mask: Ag 覆盖区掩膜（布尔数组）
            illumination: 光照模式
                "full"       —— 整个活性区均匀受光（含 Ag 覆盖区下方，底电极入射）
                "ag_only"    —— 仅 Ag 覆盖区受光（对照: 名义器件行为）
                "outside_ag" —— 仅 Ag 覆盖区外受光（Ag 不透光情形）
        """
        if illumination not in ("full", "ag_only", "outside_ag"):
            raise ValueError(f"未知光照模式: {illumination!r}")

        self.X, self.Y = X, Y
        self.geometry = geometry
        self.material = material
        self.ag_mask = np.asarray(ag_mask, dtype=bool)
        self.illumination = illumination

        self.ny, self.nx = X.shape
        dx_um = float(X[0, 1] - X[0, 0])
        dy_um = float(Y[1, 0] - Y[0, 0])
        self.dx = dx_um * 1e-4   # cm
        self.dy = dy_um * 1e-4   # cm
        self.dA = self.dx * self.dy

        # 片电导与微电池参数
        self.g_sheet = sheet_conductance(geometry, material)          # S
        self.r_sheet = 1.0 / self.g_sheet                             # Ω/sq
        self.sigma_parallel = material.pedot_conductivity_lateral     # S/cm
        self.vt = material.thermal_voltage                            # V
        self.n_ideal = material.diode_ideality
        self.j0 = material.j0 / 1000.0                                # A/cm^2
        self.rsh = material.rsh                                       # Ω·cm^2
        self.r_contact = material.contact_resistance_ag               # Ω·cm^2
        self.g_contact = 1.0 / self.r_contact if self.r_contact > 0 else None

        # 局部光生电流密度分布 (A/cm^2)
        jph = material.jph / 1000.0
        jph_map = np.full((self.ny, self.nx), jph)
        if illumination == "ag_only":
            jph_map[~self.ag_mask] = 0.0
        elif illumination == "outside_ag":
            jph_map[self.ag_mask] = 0.0
        self.jph_map = jph_map

        # 未知量划分
        n_all = self.ny * self.nx
        flat = np.arange(n_all).reshape(self.ny, self.nx)
        self.ag_flat = flat[self.ag_mask]
        if self.r_contact > 0:
            # 含接触电阻时 Ag 节点仍为未知量，通过 g_c 汇入外电路
            self.dirichlet_mask = np.zeros_like(self.ag_mask)
            self.unknown_mask = np.ones_like(self.ag_mask)
        else:
            self.dirichlet_mask = self.ag_mask
            self.unknown_mask = ~self.ag_mask
        self.unknown_idx = flat[self.unknown_mask]
        self.n_unknown = self.unknown_idx.size
        self._uof = -np.ones(n_all, dtype=int)
        self._uof[self.unknown_idx] = np.arange(self.n_unknown)

        self.jph_u = self.jph_map.ravel()[self.unknown_idx]
        self.jph_ag = self.jph_map.ravel()[self.ag_flat]

        # Ag 区几何量
        self.ag_perimeter = mask_perimeter(self.ag_mask, self.dx, self.dy)  # cm

        # 组装线性电导矩阵
        self._build_matrix()

        # 求解结果
        self.potential: Optional[np.ndarray] = None
        self.v_ext: Optional[float] = None
        self._vu: Optional[np.ndarray] = None
        self.newton_iterations = 0

    # ------------------------------------------------------------------
    # 矩阵组装
    # ------------------------------------------------------------------
    def _build_matrix(self):
        ny, nx = self.ny, self.nx
        gx = self.g_sheet * self.dy / self.dx   # x 方向相邻节点电导 (S)
        gy = self.g_sheet * self.dx / self.dy   # y 方向相邻节点电导 (S)

        # 无向边表（每条边只列一次，k1 -> k2）
        ex1 = np.arange(ny)[:, None] * nx + np.arange(nx - 1)[None, :]   # x 方向
        ey1 = np.arange(ny - 1)[:, None] * nx + np.arange(nx)[None, :]   # y 方向
        k1 = np.concatenate([ex1.ravel(), ey1.ravel()])
        k2 = np.concatenate([ex1.ravel() + 1, ey1.ravel() + nx])
        g_edge = np.concatenate([np.full(ex1.size, gx), np.full(ey1.size, gy)])

        u1, u2 = self._uof[k1], self._uof[k2]
        ff = (u1 >= 0) & (u2 >= 0)    # 自由-自由
        fd = (u1 >= 0) & (u2 < 0)     # 自由 -> Dirichlet(Ag)
        df = (u1 < 0) & (u2 >= 0)     # Dirichlet(Ag) -> 自由

        # 自由-Ag 边耦合（每条边一次）: 自由节点局部序号、Ag 节点扁平序号、电导
        self.dc_local = np.concatenate([u1[fd], u2[df]]).astype(int)
        self.dc_free_flat = np.concatenate([k1[fd], k2[df]]).astype(int)
        self.dc_vals = np.concatenate([g_edge[fd], g_edge[df]])

        n_u = self.n_unknown
        idx_u = np.arange(n_u)
        # 对角元: 每个节点对每条关联边计一次电导
        diag = np.bincount(
            np.concatenate([u1[ff], u2[ff], u1[fd], u2[df]]),
            weights=np.concatenate([g_edge[ff], g_edge[ff], g_edge[fd], g_edge[df]]),
            minlength=n_u,
        )
        a_rows = np.concatenate([u1[ff], u2[ff], idx_u])
        a_cols = np.concatenate([u2[ff], u1[ff], idx_u])
        # 行格式: Σ_nb G(V_i - V_nb) = (ΣG)·V_i - Σ G·V_nb → 非对角元为 -G
        a_vals = np.concatenate([-g_edge[ff], -g_edge[ff], diag])

        if self.g_contact is not None:
            ag_local = self._uof[self.ag_flat]
            a_rows = np.concatenate([a_rows, ag_local])
            a_cols = np.concatenate([a_cols, ag_local])
            a_vals = np.concatenate([a_vals, np.full(ag_local.size, self.g_contact * self.dA)])
            self.contact_local = ag_local

        self.a_free = sparse.coo_matrix((a_vals, (a_rows, a_cols)),
                                        shape=(n_u, n_u)).tocsr()

    # ------------------------------------------------------------------
    # 局部微电池 J-V
    # ------------------------------------------------------------------
    def j_local(self, v, jph):
        """局部单二极管微电池电流密度 (A/cm^2)，v 为 PEDOT-底电极电压 (V)"""
        arg = np.minimum(np.asarray(v, dtype=float) / (self.n_ideal * self.vt), 60.0)
        return jph - self.j0 * (np.exp(arg) - 1.0) - v / self.rsh

    # ------------------------------------------------------------------
    # 自洽求解
    # ------------------------------------------------------------------
    def solve(self, v_ext: float, v0: Optional[np.ndarray] = None,
              tol: float = 1e-9, max_iter: int = 100) -> np.ndarray:
        """
        求解给定顶电极电势 v_ext 下的电势分布

        Args:
            v_ext: Ag 顶电极电势 (V)，以底电极为参考
            v0: 初值（完整电势图或未知量向量），用于扫压热启动
            tol: 残差收敛容差 (A)
            max_iter: Newton 最大迭代次数

        Returns:
            potential: 电势分布 (V)，形状 (ny, nx)
        """
        n_u = self.n_unknown
        rhs = np.zeros(n_u)
        if self.dc_local.size:
            rhs += np.bincount(self.dc_local, weights=self.dc_vals * v_ext,
                               minlength=n_u)
        if self.g_contact is not None:
            rhs[self.contact_local] += self.g_contact * self.dA * v_ext

        if v0 is None:
            vu = np.zeros(n_u)
        else:
            vu = np.ravel(np.asarray(v0, dtype=float)).copy()
            if vu.size == self.ny * self.nx:
                vu = vu[self.unknown_idx]
            if vu.size != n_u:
                raise ValueError("v0 尺寸与未知量数量不符")

        dA, a_free, jph_u = self.dA, self.a_free, self.jph_u

        def residual(v):
            return a_free.dot(v) - rhs - self.j_local(v, jph_u) * dA

        f = residual(vu)
        converged = False
        for it in range(max_iter):
            fmax = float(np.max(np.abs(f)))
            if fmax < tol:
                converged = True
                break
            # Newton 步（二极管项进入对角元）
            dvec = (self.j0 / (self.n_ideal * self.vt)
                    * np.exp(np.minimum(vu / (self.n_ideal * self.vt), 60.0))
                    + 1.0 / self.rsh) * dA
            jac = (a_free + sparse.diags(dvec)).tocsc()
            delta = spsolve(jac, -f)
            if not np.all(np.isfinite(delta)):
                warnings.warn("Newton 增量出现非有限值，提前终止")
                break
            # 阻尼线搜索
            lam = 1.0
            v_new, f_new = vu, f
            for _ in range(25):
                v_try = vu + lam * delta
                f_try = residual(v_try)
                if float(np.max(np.abs(f_try))) <= fmax:
                    v_new, f_new = v_try, f_try
                    break
                lam *= 0.5
            else:
                v_new = vu + lam * delta
                f_new = residual(v_new)
            vu, f = v_new, f_new
        else:
            fmax = float(np.max(np.abs(f)))

        if not converged:
            warnings.warn(f"未收敛: v_ext={v_ext:.4f} V, 残差={fmax:.2e} A")

        self._vu = vu
        self.v_ext = float(v_ext)
        self.potential = np.full(self.ny * self.nx, float(v_ext))
        self.potential[self.unknown_idx] = vu
        self.potential = self.potential.reshape(self.ny, self.nx)
        self.newton_iterations = it + 1
        return self.potential

    # ------------------------------------------------------------------
    # 终端电流
    # ------------------------------------------------------------------
    def terminal_current(self, potential: Optional[np.ndarray] = None,
                         v_ext: Optional[float] = None) -> float:
        """
        终端电流 I_terminal (mA)，正值表示电流经 Ag 顶电极流出器件
        """
        V = self.potential if potential is None else potential
        v_ext = self.v_ext if v_ext is None else v_ext
        vf = np.ravel(np.asarray(V, dtype=float))
        if self.g_contact is not None:
            i_terminal = np.sum(self.g_contact * self.dA * (vf[self.ag_flat] - v_ext))
        else:
            # 经片电阻流入 Ag 覆盖区的电流 + Ag 区下方微电池直接注入的电流
            i_sheet = float(np.sum(self.dc_vals * (vf[self.dc_free_flat] - v_ext)))
            i_under_ag = float(np.sum(self.j_local(v_ext, self.jph_ag))) * self.dA
            i_terminal = i_sheet + i_under_ag
        return i_terminal * 1000.0

    def total_injected_current(self, potential: Optional[np.ndarray] = None) -> float:
        """所有微电池注入 PEDOT 网络的净电流 (mA)（稳态下应与终端电流一致）"""
        V = self.potential if potential is None else potential
        return float(np.sum(self.j_local(np.ravel(V), self.jph_map.ravel()))) * self.dA * 1000.0

    # ------------------------------------------------------------------
    # 分布量
    # ------------------------------------------------------------------
    def current_density_map(self, potential: Optional[np.ndarray] = None):
        """PEDOT 膜内横向电流密度 J = -sigma_parallel ∇V (A/cm^2)，返回 (Jx, Jy)"""
        V = self.potential if potential is None else potential
        dVdy, dVdx = np.gradient(V, self.dy, self.dx)
        jx = -self.sigma_parallel * dVdx
        jy = -self.sigma_parallel * dVdy
        return jx, jy

    def joule_heating_map(self, potential: Optional[np.ndarray] = None):
        """
        片电阻欧姆损耗分布 (W/cm^2) 与总损耗 (W)

        边耗散 G·(ΔV)^2 平均分配到两端节点。
        """
        V = self.potential if potential is None else potential
        gx = self.g_sheet * self.dy / self.dx
        gy = self.g_sheet * self.dx / self.dy
        px = gx * (V[:, 1:] - V[:, :-1]) ** 2
        py = gy * (V[1:, :] - V[:-1, :]) ** 2
        p = np.zeros_like(V)
        p[:, :-1] += px / 2.0
        p[:, 1:] += px / 2.0
        p[:-1, :] += py / 2.0
        p[1:, :] += py / 2.0
        return p / self.dA, float(px.sum() + py.sum())

    def collection_efficiency_map(self, potential: Optional[np.ndarray] = None) -> Dict:
        """
        局部收集效率图与有效收集面积

        η_local = J_local(V_local) / Jph：给定工作点下每个微电池实际
        注入网络的电流占其光生电流的比例（未被收集的部分来自二极管
        复合与并联漏电）。

        Returns:
            dict:
                eta_map             局部收集效率（无光照处为 NaN）
                a_effective         A_effective = Σ η·dA (cm^2)（积分定义）
                a_effective_50      η > 0.5 的面积 (cm^2)（阈值定义）
                collection_length   有效收集长度 L_col = (A_eff - A_ag)/P_ag (cm)
        """
        V = self.potential if potential is None else potential
        jl = self.j_local(V, self.jph_map)
        eta = np.full_like(V, np.nan)
        illuminated = self.jph_map > 0
        eta[illuminated] = jl[illuminated] / self.jph_map[illuminated]

        a_effective = float(np.nansum(eta)) * self.dA
        a_eff50 = float(np.count_nonzero(np.nan_to_num(eta, nan=0.0) > 0.5)) * self.dA
        collection_length = max(0.0, (a_effective - self.geometry.ag_area)
                                / max(self.ag_perimeter, 1e-30))
        return {
            "eta_map": eta,
            "a_effective": a_effective,
            "a_effective_50": a_eff50,
            "collection_length": collection_length,
        }


# ----------------------------------------------------------------------
# 快速自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from models.device_geometry import create_2d_mesh, create_ag_mask, visualize_geometry

    geometry = DeviceGeometry()
    material = MaterialProperties(pedot_conductivity_lateral=100.0)
    X, Y = create_2d_mesh(geometry, mesh_density=20)
    ag_mask = create_ag_mask(X, Y, geometry)

    solver = PEDOTNetworkSolver(X, Y, geometry, material, ag_mask)

    v = 0.0
    solver.solve(v)
    i_sc = solver.terminal_current()
    print(f"SC (V={v}): I_terminal = {i_sc:.4f} mA")
    print(f"  全局注入电流校验: {solver.total_injected_current():.4f} mA")
    print(f"  max(V) - V_ag = {(solver.potential.max() - v) * 1000:.4f} mV")

    coll = solver.collection_efficiency_map()
    print(f"  A_effective = {coll['a_effective'] * 100:.2f} mm^2 "
          f"(A_ag = {geometry.ag_area * 100:.2f} mm^2)")
    print(f"  有效收集长度 L_col = {coll['collection_length'] * 1e4:.0f} μm")

    voc_test = 0.7
    solver.solve(voc_test)
    print(f"V = {voc_test} V: I_terminal = {solver.terminal_current():.4f} mA")
