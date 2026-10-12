"""PEDOT Device Simulation 桌面界面。启动：python gui.py。"""

import csv
import json
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import numpy as np

from gui_backend import parse_values, run_cases, validate_config


MODES = {"正向 / Conventional": "forward", "倒置局部 Ag / Inverted": "inverted",
         "复合方阻映射 / Sheet resistance": "mapping"}


class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        root.title("PEDOT Device Simulation · 横向输运模拟")
        root.geometry("1340x880")
        root.minsize(1100, 740)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure("TLabel", background="#f4f6fa", font=("Microsoft YaHei", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei", 19, "bold"), foreground="#123250")
        style.configure("TButton", padding=7)
        style.configure("Treeview", rowheight=27)
        top = ttk.Frame(root, padding=(20, 14))
        top.pack(fill="x")
        ttk.Label(top, text="PEDOT Device Simulation", style="Title.TLabel").pack(anchor="w")
        ttk.Label(top, text="有机太阳能电池 · 横向收集 / 面积损失 / 复合阳极方阻").pack(anchor="w")
        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        sidebar = ttk.Frame(body, width=320)
        body.add(sidebar, weight=0)
        control_canvas = tk.Canvas(sidebar, width=295, background="#f4f6fa", highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=control_canvas.yview)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_scrollbar.pack(side="right", fill="y")
        control_canvas.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(control_canvas, padding=12)
        control_window = control_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda event: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.bind("<Configure>", lambda event: control_canvas.itemconfigure(control_window, width=event.width))
        view = ttk.Frame(body, padding=5)
        body.add(view, weight=1)
        self.fields = {}
        self.mode = tk.StringVar(value=list(MODES)[0])
        ttk.Label(controls, text="模型 / Model").pack(anchor="w")
        combo = ttk.Combobox(controls, textvariable=self.mode, values=list(MODES), state="readonly", width=32)
        combo.pack(fill="x", pady=(4, 12))
        combo.bind("<<ComboboxSelected>>", self.change_mode)
        self.field(controls, "sides", "正方形边长 (mm，逗号分隔)", "1, 2, 5, 10, 15, 20")
        self.field(controls, "sigmas", "PEDOT 电导率 (S/cm，逗号分隔)", "0.0023, 2")
        self.field(controls, "thickness_nm", "PEDOT 厚度 (nm)", "30")
        self.field(controls, "ito", "ITO 方阻 (Ω/□，仅正向)", "15")
        self.field(controls, "ag_mm", "局部 Ag 边长 (mm，仅倒置)", "2")
        self.field(controls, "nodes", "每边网格节点数 (11–301)", "51")
        self.field(controls, "ito_range", "ITO 方阻范围 (Ω/□，仅映射)", "1, 100")
        ttk.Label(controls, text="倒置照光区域 / Illumination").pack(anchor="w", pady=(4, 2))
        self.illumination = tk.StringVar(value="full")
        self.illumination_combo = ttk.Combobox(controls, textvariable=self.illumination,
                                               values=["full", "ag_only", "outside_ag"], state="readonly")
        self.illumination_combo.pack(fill="x", pady=(0, 12))
        self.description = tk.StringVar()
        ttk.Label(controls, textvariable=self.description, wraplength=275,
                  foreground="#52647c").pack(fill="x", pady=5)
        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=10)
        self.run_button = ttk.Button(buttons, text="运行 / Run", command=self.start)
        self.run_button.pack(side="left", expand=True, fill="x")
        self.stop_button = ttk.Button(buttons, text="停止 / Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.progress = ttk.Progressbar(controls, mode="determinate")
        self.progress.pack(fill="x", pady=4)
        self.status = tk.StringVar(value="就绪 / Ready")
        ttk.Label(controls, textvariable=self.status, wraplength=275).pack(anchor="w", pady=5)
        self.export_button = ttk.Button(controls, text="导出数据与当前图 / Export", command=self.export, state="disabled")
        self.export_button.pack(fill="x", pady=5)
        notebook = ttk.Notebook(view)
        notebook.pack(fill="both", expand=True)
        plot_frame = ttk.Frame(notebook)
        table_frame = ttk.Frame(notebook, padding=8)
        log_frame = ttk.Frame(notebook, padding=8)
        notebook.add(plot_frame, text="曲线与二维图 / Plots")
        notebook.add(table_frame, text="数据 / Data")
        notebook.add(log_frame, text="运行记录 / Log")
        self.figure = Figure(figsize=(10, 7), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()
        self.columns = ("side_mm", "area_mm2", "sigma_pedot", "jsc", "voc", "ff", "pce")
        self.table = ttk.Treeview(table_frame, columns=self.columns, show="headings")
        for key, title in zip(self.columns, ["L (mm)", "Area (mm²)", "σ (S/cm)", "Jsc (mA/cm²)", "Voc (V)", "FF", "PCE (%)"]):
            self.table.heading(key, text=title)
            self.table.column(key, width=115, anchor="center")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select_case)
        ttk.Label(table_frame, text="选择一行更新 MPP 空间图；Jsc 和 PCE 均以 Ag 面积归一化。完整指标见导出 CSV。").pack(anchor="w", pady=7)
        self.log = tk.Text(log_frame, wrap="word", font=("Consolas", 10), state="disabled")
        self.log.pack(fill="both", expand=True)
        self.change_mode()
        self.figure.text(0.5, 0.5, "选择模型与参数，点击运行\nChoose a model and press Run",
                         ha="center", va="center", fontsize=17, color="#52647c")
        self.canvas.draw()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def field(self, parent, key, label, default):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 2))
        variable = tk.StringVar(value=default)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.pack(fill="x", pady=(0, 3))
        self.fields[key] = (variable, entry)

    def change_mode(self, event=None):
        mode = MODES[self.mode.get()]
        if event:
            self.fields["sides"][0].set("5" if mode == "inverted" else "1, 2, 5, 10, 15, 20")
            self.fields["sigmas"][0].set("1e-4, 1e6" if mode == "mapping" else "0.0023, 2")
            self.fields["nodes"][0].set("201" if mode == "inverted" else "51")
        for key, (_, entry) in self.fields.items():
            enabled = key in {"thickness_nm", "sigmas"} or (
                mode == "mapping" and key == "ito_range") or (
                mode != "mapping" and key in {"sides", "nodes"}) or (
                mode == "forward" and key == "ito") or (mode == "inverted" and key == "ag_mm")
            entry.configure(state="normal" if enabled else "disabled")
        self.illumination_combo.configure(state="readonly" if mode == "inverted" else "disabled")
        self.description.set({
            "forward": "ITO/PEDOT/活性层/PDINN/Ag\n整面 Ag；ITO x=0 整边取电。ITO/PEDOT 理想并联。",
            "inverted": "底电极/ETL/活性层/PEDOT/局部 Ag\n中央正方形 Ag 接触，底电极等势。网格间距以下的收集长度不定量解析。",
            "mapping": "电导率列表的最小/最大值定义横轴范围。纵轴为 ITO 方阻范围，颜色为复合层方阻。解析计算，无需网格求解。",
        }[mode] + "\n局部电池基准：Jsc=25、Voc=0.9 V、FF≈0.77；光强 100 mW/cm²。")

    def config(self):
        mode = MODES[self.mode.get()]
        # 不读取禁用字段中的用户文本，避免无关字段阻塞有效算例。
        return {"mode": mode,
                "sides": parse_values(self.fields["sides"][0].get()) if mode != "mapping" else [5.0],
                "sigmas": parse_values(self.fields["sigmas"][0].get()),
                "thickness_nm": float(self.fields["thickness_nm"][0].get()),
                "ito": float(self.fields["ito"][0].get()) if mode == "forward" else 15.0,
                "ag_mm": float(self.fields["ag_mm"][0].get()) if mode == "inverted" else 2.0,
                "nodes": int(self.fields["nodes"][0].get()) if mode != "mapping" else 51,
                "ito_range": parse_values(self.fields["ito_range"][0].get()) if mode == "mapping" else [1.0, 100.0],
                "illumination": self.illumination.get()}

    def start(self):
        if self.busy:
            return
        try:
            config = self.config()
            validate_config(config)
        except (ValueError, OverflowError) as error:
            messagebox.showerror("参数错误 / Invalid parameters", str(error))
            return
        self.cancel.clear()
        self.busy = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("正在计算 / Computing…")
        self.progress["value"] = 0
        self.append_log(json.dumps(config, ensure_ascii=False))
        threading.Thread(target=self.worker, args=(config,), daemon=True).start()

    def worker(self, config):
        try:
            result = run_cases(config, lambda *args: self.events.put(("progress", args)), self.cancel)
            self.events.put(("result", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    done, total, text = value
                    self.progress["maximum"] = total
                    self.progress["value"] = done
                    self.status.set(f"{done}/{total} · {text}")
                    self.append_log(self.status.get())
                else:
                    self.busy = False
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if kind == "error":
                        self.status.set("计算失败 / Failed")
                        self.append_log(value)
                        messagebox.showerror("计算失败", value.splitlines()[-1])
                    else:
                        self.result = value
                        self.status.set("已停止；保留完成结果" if value["cancelled"] else "完成 / Complete")
                        self.append_log(self.status.get())
                        self.table.delete(*self.table.get_children())
                        for index, row in enumerate(value["rows"]):
                            self.table.insert("", "end", iid=str(index),
                                              values=[f"{row[key]:.6g}" for key in self.columns])
                        if value["cases"] or "resistance" in value:
                            self.draw_result()
                            self.export_button.configure(state="normal")
                        else:
                            self.figure.clear()
                            self.canvas.draw_idle()
                            self.export_button.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def stop(self):
        self.cancel.set()
        self.status.set("将在当前算例结束后停止…")

    def select_case(self, event=None):
        selection = self.table.selection()
        if selection and self.result and self.result["cases"]:
            self.draw_result(int(selection[0]))

    def draw_result(self, index=0):
        result = self.result
        self.figure.clear()
        if "resistance" in result:
            ax = self.figure.add_subplot(111)
            image = ax.pcolormesh(result["sigma"], result["ito"], result["resistance"],
                                 norm=LogNorm(), cmap="viridis", shading="auto")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("PEDOT σ (S/cm)")
            ax.set_ylabel("ITO R_sheet (Ω/□)")
            ax.set_title(f"PEDOT / ITO composite · t = {result['config']['thickness_nm']:g} nm")
            self.figure.colorbar(image, ax=ax, label="Composite R_sheet (Ω/□)")
        else:
            axes = self.figure.subplots(2, 2)
            for case in result["cases"]:
                display = case["voltage"] <= 0.95
                axes[0, 0].plot(case["voltage"][display], case["j_ag"][display],
                                label=f"{case['side']:g} mm / {case['sigma']:g} S/cm")
            axes[0, 0].set(xlabel="Voltage (V)", ylabel="J_Ag (mA/cm²)", title="J–V (Ag area)", xlim=(0, 0.95))
            axes[0, 0].legend(fontsize=7, ncol=2)
            for sigma in result["config"]["sigmas"]:
                rows = [r for r in result["rows"] if r["sigma_pedot"] == sigma]
                if rows:
                    area = [r["area_mm2"] for r in rows]
                    axes[0, 1].plot(area, [r["pce"] for r in rows], "o-", label=f"{sigma:g} S/cm")
                    axes[1, 0].plot(area, [r["ff"] for r in rows], "o-", label=f"{sigma:g} S/cm")
            for ax, metric in [(axes[0, 1], "PCE_Ag (%)"), (axes[1, 0], "FF")]:
                ax.set(xlabel="Square active area (mm²)", ylabel=metric, title=metric)
                ax.legend(fontsize=8)
            # 常规纵轴尺度，避免将普通 4083 与 2 S/cm 的微小差异视觉放大。
            axes[0, 1].set_ylim(0, max(r["pce"] for r in result["rows"]) * 1.1)
            axes[1, 0].set_ylim(0, 1)
            for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
                ax.grid(alpha=0.3)
            case = result["cases"][index]
            image = axes[1, 1].pcolormesh(case["x"], case["y"],
                                         (case["potential"] - case["vmpp"]) * 1000,
                                         shading="auto", cmap="magma")
            axes[1, 1].set(xlabel="x (mm)", ylabel="y (mm)",
                           title=f"MPP · {case['side']:g} mm / {case['sigma']:g} S/cm")
            axes[1, 1].set_aspect("equal")
            self.figure.colorbar(image, ax=axes[1, 1], label="V_local − V_terminal (mV)")
        self.canvas.draw_idle()

    def export_to(self, directory):
        result = self.result
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "parameters.json").write_text(json.dumps(result["config"], ensure_ascii=False, indent=2), encoding="utf-8")
        self.figure.savefig(path / "plots.png", dpi=200)
        if "resistance" in result:
            np.savetxt(path / "sheet_resistance_map.csv",
                       np.column_stack([result["sigma"].ravel(), result["ito"].ravel(), result["resistance"].ravel()]),
                       delimiter=",", header="sigma_S_cm,ito_ohm_sq,composite_ohm_sq", comments="")
        elif result["rows"]:
            with (path / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
                writer.writeheader()
                writer.writerows(result["rows"])
            for index, case in enumerate(result["cases"], 1):
                np.savetxt(path / f"case_{index:02d}_jv.csv", np.column_stack([case["voltage"], case["j_ag"]]),
                           delimiter=",", header="voltage_V,J_Ag_mA_cm2", comments="")
                np.savetxt(path / f"case_{index:02d}_mpp_map.csv",
                           np.column_stack([case["x"].ravel(), case["y"].ravel(), case["potential"].ravel()]),
                           delimiter=",", header="x_mm,y_mm,local_junction_voltage_V", comments="")

    def export(self):
        directory = filedialog.askdirectory(title="选择导出父目录 / Export folder")
        if not directory:
            return
        try:
            # 每次创建新目录，避免覆盖之前的研究结果。
            import tempfile
            target = tempfile.mkdtemp(prefix="pedot_gui_", dir=directory)
            self.export_to(target)
            self.append_log(f"Export: {target}")
            messagebox.showinfo("导出完成 / Export complete", target)
        except Exception as error:
            messagebox.showerror("导出失败", str(error))

    def close(self):
        if self.busy and not messagebox.askyesno("计算仍在运行", "退出将终止本次计算，是否退出？"):
            return
        self.cancel.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    SimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
