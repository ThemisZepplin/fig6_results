"""Execute production validators with synthetic NumPy dataset fixtures.

No raw files, NetCDF I/O, science recomputation, or plotting dependencies needed.
"""
import ast
from datetime import datetime
from pathlib import Path
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class DatasetFixture:
    def __init__(self, fields, attrs):
        self.fields = {
            name: types.SimpleNamespace(values=value, shape=value.shape, attrs={})
            for name, value in fields.items()
        }
        self.attrs = attrs
        self.data_vars = {}

    def __getitem__(self, name):
        return self.fields[name]

    def __getattr__(self, name):
        return self.fields[name]


def validators(filename, specific):
    names = {"_initialization_from_attrs", "_validate_identity", "_validate_pair", specific}
    tree = ast.parse((ROOT / filename).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + selected, type_ignores=[])
    namespace = {"np": np, "datetime": datetime}
    exec(compile(module, filename, "exec"), namespace)
    return namespace


class InitializationChecks(unittest.TestCase):
    def setUp(self):
        self.cases = []
        for filename, specific, hov in (
            ("paper_fig7_from_cache.py", "_fig7_validate_significance", False),
            ("paper_fig8_from_cache.py", "_fig8_validate_significance", True),
        ):
            ns = validators(filename, specific)
            start, end, coord = ("2023-06-14", "2023-06-24", "time") if hov else ("2023-06-18", "2023-06-20", "date")
            dates = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"))
            shape = (len(dates), 2) if hov else (len(dates), 2, 2)
            difference = np.ones(shape)
            common = {coord: dates, "s2s_lon": np.array([114., 119.])}
            if not hov:
                common["s2s_lat"] = np.array([35., 44.])
            cache = DatasetFixture({**common, "tp_diff_hov" if hov else "tp_diff": difference}, {
                "init_date": "2023-06-08", "main_init_date": "2023-06-12",
                "grouping_stage": "Total", "analysis_window": f"{start} to {end}",
                "z500_type": "absolute", "z500_units": "gpm", "tp_units": "mm day^-1",
                "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
            })
            sig = DatasetFixture({**common, "hot_minus_cold": difference.copy(),
                "significant_local_p005": np.ones(shape, dtype=np.int8),
                "p_value": np.full(shape, .01), "n_hot": np.full(shape, 17),
                "n_cold": np.full(shape, 17)}, {
                    "init_date": "2023-06-08", "grouping_stage": "Total",
                    "units": "mm day^-1", "difference_direction": "Hot17 minus Cold17",
                })
            self.cases.append((ns, specific, hov, cache, sig, start, end, coord))

    def check_case(self, case, init="2023-06-08"):
        ns, specific, hov, cache, sig, start, end, coord = case
        ns["_validate_identity"](cache, init, start, end, coord)
        ns["_validate_pair"](cache, sig, init)
        if hov:
            ns[specific](cache, sig, None, coord)
        else:
            ns[specific](cache, sig)

    def test_legacy_june8_overrides_stale_main_init(self):
        for case in self.cases:
            self.check_case(case)

    def test_new_initialization_has_priority(self):
        for case in self.cases:
            for ds in case[3:5]:
                ds.attrs.update(initialization="2023-06-08T00:00:00Z", init_date="2023-06-12")
            self.check_case(case)

    def test_blank_or_invalid_primary_uses_valid_legacy_value(self):
        for case in self.cases:
            for primary in ("", "not-a-date", "NaT", None):
                for ds in case[3:5]:
                    ds.attrs["initialization"] = primary
                self.check_case(case)

    def test_wrong_valid_init_is_rejected_without_lower_priority_override(self):
        for case in self.cases:
            for target in (case[3], case[4]):
                target.attrs["initialization"] = "2023-06-12"
                with self.assertRaises(ValueError):
                    self.check_case(case)
                del target.attrs["initialization"]

    def test_missing_metadata_never_infers_june8(self):
        for case in self.cases:
            for ds in case[3:5]:
                ds.attrs.pop("init_date", None)
                ds.attrs.pop("main_init_date", None)
            with self.assertRaises(ValueError):
                case[0]["_validate_identity"](case[3], "2023-06-08", case[5], case[6], case[7])
            with self.assertRaises(ValueError):
                case[0]["_validate_pair"](case[3], case[4], "2023-06-08")
            # Preserve only the existing metadata-free June 12 significance case.
            case[0]["_validate_pair"](case[3], case[4], "2023-06-12")

    def test_main_init_only_and_invalid_metadata(self):
        for case in self.cases:
            resolver = case[0]["_initialization_from_attrs"]
            self.assertEqual(resolver({"main_init_date": "2023-06-08"}), "2023-06-08")
            self.assertEqual(resolver({"init_date": b"2023-06-08"}), "2023-06-08")
            with self.assertRaises(ValueError):
                resolver({"initialization": "not-a-date"})

    def test_wrong_coordinates_dates_and_differences_still_rejected(self):
        for case in self.cases:
            ns, specific, hov, cache, sig, start, end, coord = case
            keys = [coord, "s2s_lon", "hot_minus_cold"] + ([] if hov else ["s2s_lat"])
            for key in keys:
                saved = sig.fields[key]
                value = saved.values + (np.timedelta64(1, "D") if key == coord else 1)
                sig.fields[key] = types.SimpleNamespace(values=value, shape=value.shape)
                with self.assertRaises(ValueError):
                    self.check_case(case)
                sig.fields[key] = saved

    def test_wrong_grouping_and_tp_units_still_rejected(self):
        for case in self.cases:
            for target, key, wrong in ((case[3], "grouping_stage", "StageII"),
                                       (case[4], "grouping_stage", "StageII"),
                                       (case[4], "units", "gpm")):
                saved = target.attrs[key]
                target.attrs[key] = wrong
                with self.assertRaises(ValueError):
                    self.check_case(case)
                target.attrs[key] = saved

    def test_audit_cleanup(self):
        daily = (ROOT / "cell_fig7_stageII_daily_synoptic_paper.py").read_text()
        hov = (ROOT / "cell_hovmoller_tp_z500_obs_s2s_stageII.py").read_text()
        self.assertNotIn("Default run flags:", daily)
        self.assertNotIn("Archived Z500 contours indicate", hov)
        self.assertIn("without displaying Z500 significance", hov)


if __name__ == "__main__":
    unittest.main()
