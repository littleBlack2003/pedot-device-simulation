"""局部单二极管参数、无横向压降的 J–V 参考与网格面积换算。"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from models.device_geometry import MaterialProperties


DEFAULT_DIODE = {"jph": 25.0, "diode_ideality": 1.5, "rsh": 466.15, "temperature": 300.0}
_vt = MaterialProperties(temperature=300.0).thermal_voltage
DEFAULT_DIODE["j0"] = (25.0 - 1000.0 * 0.9 / 466.15) / (np.expm1(0.9 / (1.5 * _vt)))


def validate_diode(parameters):
    for key in DEFAULT_DIODE:
        if key not in parameters or not np.isfinite(parameters[key]) or parameters[key] <= 0:
            raise ValueError(f"单二极管参数 {key} 必须为有限正数。")


def local_reference(parameters):
    """单位 mA/cm²、V、Ω·cm²、K；输入密度按真实几何面积定义。"""
    validate_diode(parameters)
    vt = MaterialProperties(temperature=parameters["temperature"]).thermal_voltage
    scale = parameters["diode_ideality"] * vt

    def current(voltage):
        return (parameters["jph"] - parameters["j0"] * np.expm1(np.minimum(np.asarray(voltage) / scale, 60))
                - 1000 * np.asarray(voltage) / parameters["rsh"])

    # 忽略分流的 Voc 是完整模型 Voc 的上界。避免预设 0.9 V 限制手动调参。
    upper = scale * (np.logaddexp(np.log(parameters["jph"]), np.log(parameters["j0"]))
                     - np.log(parameters["j0"]))
    if not 1e-5 < upper < 5:
        raise ValueError("这组参数的电压尺度超出交互模型范围（约 0–5 V），请检查 J0、n 和温度。")
    voc = brentq(current, 0, upper * 1.01, xtol=1e-12)
    optimum = minimize_scalar(lambda voltage: -voltage * current(voltage),
                               bounds=(0, voc), method="bounded")
    vmpp = float(optimum.x)
    pmpp = float(-optimum.fun)
    voltage = np.linspace(0, voc * 1.05, 201)
    return {"voltage": voltage, "current": current(voltage), "voc": voc,
            "jsc": parameters["jph"], "vmpp": vmpp,
            "ff": pmpp / (parameters["jph"] * voc), "pce": pmpp}


def solver_diode_parameters(config, area_factor):
    """将几何面积密度换算为现有节点型求解器的积分密度。

    自动模式保持既有标定。手动模式同步换算全部面积项，保持输入 J–V 形状。
    """
    if config.get("diode_mode", "calibrated") == "manual":
        parameters = dict(config["diode"])
        validate_diode(parameters)
        parameters["jph"] /= area_factor
        parameters["j0"] /= area_factor
        parameters["rsh"] *= area_factor
    else:
        parameters = dict(DEFAULT_DIODE)
        parameters["jph"] /= area_factor
        vt = MaterialProperties(temperature=parameters["temperature"]).thermal_voltage
        parameters["j0"] = (parameters["jph"] - 1000 * 0.9 / parameters["rsh"]) / np.expm1(0.9 / (parameters["diode_ideality"] * vt))
        validate_diode(parameters)
    return parameters
