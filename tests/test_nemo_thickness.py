import ast
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative_path, names, namespace):
    """Load real dataset logic without optional ESMF/geospatial imports."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing production definitions: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


definitions = load_definitions(
    "libs/validation/datasets/base.py",
    ["Dataset", "ModelThickDataset"],
    {
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "Path": Path,
        "np": np,
        "xr": xr,
        "warnings": warnings,
        "datetime": datetime,
    },
)
load_definitions(
    "libs/validation/datasets/nemo.py",
    ["NemoDataset", "NemoThickDataset"],
    definitions,
)
NemoThickDataset = definitions["NemoThickDataset"]


class NemoThicknessTests(unittest.TestCase):
    @staticmethod
    def _fields(**variables):
        return xr.Dataset(
            {
                name: (("time_counter", "y", "x"), np.asarray(values, dtype=float))
                for name, values in variables.items()
            }
        )

    @staticmethod
    def _grid(shape, land=None):
        mask = np.zeros(shape, dtype=bool) if land is None else np.asarray(land)
        return SimpleNamespace(land_mask=lambda: mask)

    @staticmethod
    def _netcdf_loader(fields):
        """Read genuine NetCDF data while keeping fixtures independent of disk access."""
        payload = bytes(fields.to_netcdf(engine="scipy"))
        return Mock(side_effect=lambda file: xr.open_dataset(BytesIO(payload), engine="scipy"))

    def _dataset(self, shape, land=None, **kwargs):
        with patch.object(NemoThickDataset, "_create_dates_dict", return_value={}), patch.object(
            NemoThickDataset, "_create_grid", return_value=self._grid(shape, land)
        ):
            return NemoThickDataset(ROOT, **kwargs)

    def test_explicit_source_selection_when_both_fields_exist(self):
        for source, expected in [("sithic", 7.0), ("sivolu", 2.0)]:
            with self.subTest(source=source):
                fields = self._fields(sithic=[[[7]]], sivolu=[[[1]]], siconc=[[[0.5]]])
                load = Mock(return_value=fields)
                dataset = self._dataset((1, 1), thickness_source=source)

                result = dataset._extract_data(Path("ice.nc"), load_fn=load)

                np.testing.assert_array_equal(result, [[[[expected]]]])
                load.assert_called_once_with(Path("ice.nc"))

    def test_invalid_source_fails_before_file_discovery(self):
        for source in ["auto", "SIVOLU", "unknown", None]:
            with self.subTest(source=source), patch.object(
                NemoThickDataset, "_create_dates_dict"
            ) as discover, patch.object(NemoThickDataset, "_create_grid") as create_grid:
                with self.assertRaises(ValueError):
                    NemoThickDataset("missing-directory", thickness_source=source)
                discover.assert_not_called()
                create_grid.assert_not_called()

    def test_volume_division_masks_invalid_data_and_preserves_fraction_units(self):
        fields = self._fields(
            sivolu=[[[1, 1, 1, 1, 1, 1, 1, 0, -1, np.nan, np.inf, -np.inf]]],
            siconc=[[[0.5, 0, -0.5, np.nan, np.inf, 1.1, 50, 0.5, 0.5, 0.5, 0.5, 0.5]]],
        )
        dataset = self._dataset((1, 12), thickness_source="sivolu")

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = dataset._extract_data("ice.nc", load_fn=Mock(return_value=fields))

        expected = np.full((1, 1, 1, 12), np.nan)
        expected[0, 0, 0, 0] = 2.0
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_default_direct_source_works_without_concentration(self):
        fields = self._fields(sithic=[[[3, 0, -1, np.nan, np.inf, -np.inf, 5]]])
        dataset = self._dataset((1, 7), land=[[False] * 6 + [True]])

        result = dataset._extract_data("ice.nc", load_fn=Mock(return_value=fields))

        np.testing.assert_allclose(
            result, [[[[3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]]], equal_nan=True
        )

    def test_concentration_and_land_masks_apply_to_both_sources(self):
        for source in ["sithic", "sivolu"]:
            with self.subTest(source=source):
                fields = self._fields(
                    sithic=[[[4, 4, 4, 4, 4, 4, 4]]],
                    sivolu=[[[2, 2, 2, 2, 2, 2, 2]]],
                    siconc=[[[0.5, 0, -0.5, np.nan, np.inf, 1.1, 0.5]]],
                )
                dataset = self._dataset(
                    (1, 7), land=[[False] * 6 + [True]], thickness_source=source
                )

                result = dataset._extract_data("ice.nc", load_fn=Mock(return_value=fields))

                np.testing.assert_allclose(
                    result,
                    [[[[4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]]],
                    equal_nan=True,
                )

    def test_hourly_ratios_are_averaged_and_empty_hours_stay_missing(self):
        class Wrapped(NemoThickDataset):
            @staticmethod
            def _parse_date(file):
                date_part = file.name.split("_")[-1].split("-")[0]
                return datetime.strptime(date_part, "%Y%m%d").date()

        fields = self._fields(
            sivolu=[[[1, 2, 0]], [[3, 0, 0]], [[0, 0, 0]]],
            siconc=[[[0.5, 0.5, 0]], [[1, 0, 0]], [[0, 0, 0]]],
        )
        date = datetime(2015, 1, 21).date()
        root = Path("nemo_data")
        file = root / "NESTP12_1h_icemod_state_20150121-20150121.nc"
        load = self._netcdf_loader(fields)
        with patch.object(Path, "glob", return_value=iter([file])) as discover, patch.object(
            Wrapped, "_create_grid", return_value=self._grid((1, 3))
        ):
            dataset = Wrapped(
                root,
                None,
                [0, 1, 2],
                "hourly thickness",
                files_template="*.nc",
                thickness_source="sivolu",
            )

            discover.assert_called_once_with("*.nc")
            self.assertEqual(dataset.dates_dict, {date: [file]})
            self.assertEqual(dataset._files_template, "*.nc")
            self.assertEqual(dataset.name, "hourly thickness")

        extract = dataset._extract_data
        hourly = extract(file, load_fn=load)
        with patch.object(dataset, "_extract_data", side_effect=lambda file: extract(file, load_fn=load)):
            result = dataset[date]

        self.assertEqual(hourly.shape, (3, 1, 1, 3))
        self.assertTrue(np.isnan(hourly[2]).all())
        np.testing.assert_allclose(result, [[[2.5, 4.0, np.nan]]], equal_nan=True)
        ratio_of_means = fields.sivolu[:, 0, 0].mean() / fields.siconc[:, 0, 0].mean()
        self.assertNotAlmostEqual(result[0, 0, 0], float(ratio_of_means))

    def test_default_source_preserves_legacy_constructor_and_file_discovery(self):
        root = Path("nemo_data")
        file = root / "run_001" / "NESTP12-VP1_y2015m01d21_forecast.1h_icemod.nc"
        load = self._netcdf_loader(self._fields(sithic=[[[3.0]]]))
        with patch.object(Path, "glob", return_value=iter([file])) as discover, patch.object(
            NemoThickDataset, "_create_grid", return_value=self._grid((1, 1))
        ):
            dataset = NemoThickDataset(root, None, [0], "legacy thickness")

            discover.assert_called_once_with("run_*/NESTP12-VP1_*_forecast.1h_icemod.nc")
            self.assertEqual(dataset.dates_dict, {datetime(2015, 1, 21).date(): [file]})

        extract = dataset._extract_data
        with patch.object(dataset, "_extract_data", side_effect=lambda file: extract(file, load_fn=load)):
            np.testing.assert_array_equal(dataset[datetime(2015, 1, 21).date()], [[[3.0]]])
        load.assert_called_once_with(file)


if __name__ == "__main__":
    unittest.main()
