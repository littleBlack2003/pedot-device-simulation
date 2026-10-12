"""在有桌面的环境中验证 Tk 窗口、后台完成事件与三类数据导出。"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk
from gui import MODES, SimulationApp
from gui_backend import run_cases


def main():
    root = tk.Tk()
    app = SimulationApp(root)
    root.update()
    app.fields["sides"][0].set("5")
    app.fields["sigmas"][0].set("0.0023, 2")
    app.fields["nodes"][0].set("21")
    app.start()
    deadline = time.monotonic() + 60
    while app.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)
    assert not app.busy, "Background worker timed out"
    assert app.result and len(app.table.get_children()) == 2
    app.table.selection_set("1")
    root.update()
    output = Path("results/gui_smoke")
    app.export_to(output / "forward")
    assert (output / "forward/summary.csv").exists()
    assert (output / "forward/case_02_jv.csv").exists()
    for mode in ("mapping", "inverted"):
        app.mode.set(next(label for label, value in MODES.items() if value == mode))
        app.change_mode()
        app.fields["sigmas"][0].set("0.0023, 2")
        app.fields["sides"][0].set("5")
        app.result = run_cases(app.config())
        app.draw_result()
        root.update()
        app.export_to(output / mode)
        assert (output / mode / "plots.png").exists()
    assert (output / "mapping/sheet_resistance_map.csv").exists()
    assert (output / "inverted/case_01_mpp_map.csv").exists()
    app.diode_mode.set("manual")
    app.change_diode_mode()
    app.fields["jph"][0].set("30")
    app.fields["j0"][0].set("1e-12")
    assert app.config()["diode"]["jph"] == 30
    assert str(app.fields["jph"][1].cget("state")) == "normal"
    app.preview_diode()
    root.update()
    assert any(isinstance(child, tk.Toplevel) for child in root.winfo_children())
    app.result = run_cases(app.config())
    app.draw_result()
    app.export_to(output / "manual")
    assert (output / "manual/parameters.json").exists()
    print("Desktop smoke check passed: window, background worker, case selection, three-mode export")
    print("Export button bottom / window height:",
          app.export_button.winfo_rooty() - root.winfo_rooty() + app.export_button.winfo_height(),
          root.winfo_height())
    root.destroy()


if __name__ == "__main__":
    main()
