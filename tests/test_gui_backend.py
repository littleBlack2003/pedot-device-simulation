"""GUI 参数、数据和原求解器的一致性检查。"""

import unittest
from threading import Event

import numpy as np

from analysis.forward_edge_contact import solve_forward_edge_case
from gui_backend import parse_values, run_cases, validate_config
from models.local_diode import DEFAULT_DIODE, local_reference, solver_diode_parameters


def config(mode="forward"):
    return {"mode": mode, "sides": [5.0], "sigmas": [2.0], "thickness_nm": 30.0,
            "ito": 15.0, "ag_mm": 2.0, "nodes": 21,
            "ito_range": [1.0, 100.0], "illumination": "full"}


class BackendTests(unittest.TestCase):
    def test_parse_and_invalid_values(self):
        self.assertEqual(parse_values("2，0.0023, 2"), [0.0023, 2.0])
        for value in ("", "nan", "inf", "0", "-1"):
            with self.assertRaises(ValueError):
                parse_values(value)

    def test_geometry_and_mesh_validation(self):
        item = config("inverted")
        item["ag_mm"] = 6
        with self.assertRaises(ValueError):
            validate_config(item)
        item = config()
        item["nodes"] = 1
        with self.assertRaises(ValueError):
            validate_config(item)

    def test_forward_matches_existing_solver(self):
        item = config()
        gui_result = run_cases(item)
        reference = solve_forward_edge_case(5, 2, 15, 30, 21, True)
        self.assertAlmostEqual(gui_result["rows"][0]["pce"], reference.pce, places=10)
        self.assertAlmostEqual(gui_result["rows"][0]["ff"], reference.ff, places=10)
        np.testing.assert_allclose(gui_result["cases"][0]["j_ag"], reference.j_ag)
        self.assertLess(abs(reference.j_ag[0] - reference.jsc), 1e-12)

    def test_mapping_bounds(self):
        item = config("mapping")
        item["sigmas"] = [1e-4, 1e6]
        result = run_cases(item)
        self.assertTrue(np.all(result["resistance"] > 0))
        self.assertTrue(np.all(result["resistance"] <= result["ito"]))
        self.assertTrue(np.all(np.diff(result["resistance"], axis=1) < 0))

    def test_cancel_before_first_case(self):
        cancel = Event()
        cancel.set()
        result = run_cases(config(), cancel=cancel)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["rows"], [])

    def test_inverted_ag_only_reference(self):
        item = config("inverted")
        item["sigmas"] = [0.0023]
        item["illumination"] = "ag_only"
        result = run_cases(item)
        self.assertAlmostEqual(result["rows"][0]["jsc"], 25.0, places=3)
        self.assertAlmostEqual(result["rows"][0]["voc"], 0.9, places=3)
        self.assertTrue(np.isfinite(result["cases"][0]["potential"]).all())

    def test_manual_area_conversion_preserves_local_curve(self):
        item = config()
        item["diode_mode"] = "manual"
        item["diode"] = dict(DEFAULT_DIODE)
        physical = local_reference(item["diode"])
        converted = local_reference(solver_diode_parameters(item, 1.2))
        self.assertAlmostEqual(physical["voc"], converted["voc"], places=9)
        self.assertAlmostEqual(physical["ff"], converted["ff"], places=9)
        self.assertAlmostEqual(physical["jsc"], converted["jsc"] * 1.2, places=9)

    def test_manual_parameters_change_forward_and_inverted(self):
        for mode in ("forward", "inverted"):
            item = config(mode)
            item["illumination"] = "ag_only"
            item["sigmas"] = [0.0023]
            item["diode_mode"] = "manual"
            item["diode"] = dict(DEFAULT_DIODE, jph=30.0, j0=1e-12)
            first = run_cases(item)
            self.assertGreater(first["rows"][0]["voc"], 1.05)
            self.assertGreater(first["cases"][0]["voltage"][-1], first["rows"][0]["voc"])
            item["diode"]["jph"] = 15.0
            second = run_cases(item)
            self.assertGreater(first["rows"][0]["jsc"], second["rows"][0]["jsc"] * 1.9)
            self.assertGreater(first["rows"][0]["voc"], second["rows"][0]["voc"])
            self.assertIn("solver_j0", first["rows"][0])

    def test_manual_invalid_parameter(self):
        item = config()
        item["diode_mode"] = "manual"
        item["diode"] = dict(DEFAULT_DIODE, temperature=0)
        with self.assertRaises(ValueError):
            validate_config(item)


if __name__ == "__main__":
    unittest.main()
