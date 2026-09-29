"""Array compatibility and buoy alignment through the production Validator.

Private package aliases avoid importing unrelated native geospatial dependencies.
Metric bodies are loaded from the production source, as in other unit tests.
"""

import ast
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date
import importlib.util
from io import StringIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.neighbors import BallTree, KDTree


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "libs" / "validation"
ALIAS = "_input_adapter_test_validation"
_previous_alias_modules = {
    key: value for key, value in sys.modules.items()
    if key == ALIAS or key.startswith(ALIAS + ".")
}


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package(ALIAS, VALIDATION)
package(ALIAS + ".datasets", VALIDATION / "datasets")
buoy_package = package(ALIAS + ".datasets.buoy", VALIDATION / "datasets" / "buoy")
buoy_types = load_module(
    ALIAS + ".datasets.buoy.buoy_types",
    VALIDATION / "datasets" / "buoy" / "buoy_types.py",
)
buoy_package.BuoyBatch = buoy_types.BuoyBatch

metrics = ModuleType(ALIAS + ".metrics")
metrics.__dict__.update(np=np, ABC=ABC, abstractmethod=abstractmethod, List=List)
metric_path = VALIDATION / "metrics.py"
metric_tree = ast.parse(metric_path.read_text(encoding="utf-8-sig"))
metric_names = {
    "Metric", "MSE", "Difference", "AngleError", "VectorNorm", "SequentialMetric",
    "_unpack_tuple", "ReversedSkillScore", "StatTransformed", "StatSquared", "IdentityStat", "MAE",
}
metric_nodes = [node for node in metric_tree.body if getattr(node, "name", None) in metric_names]
assert {node.name for node in metric_nodes} == metric_names
exec(compile(ast.Module(body=metric_nodes, type_ignores=[]), str(metric_path), "exec"), metrics.__dict__)
sys.modules[metrics.__name__] = metrics

aggregation = ModuleType(ALIAS + ".aggregators")
aggregation.__dict__.update(np=np, ABC=ABC, abstractmethod=abstractmethod, Any=Any, Dict=Dict, Tuple=Tuple)
aggregation_path = VALIDATION / "aggregators.py"
aggregation_tree = ast.parse(aggregation_path.read_text(encoding="utf-8-sig"))
aggregation_names = {"Aggregator", "SpatialAggregator", "AverageAggregator", "RawFieldAggregator"}
aggregation_nodes = [node for node in aggregation_tree.body if getattr(node, "name", None) in aggregation_names]
assert {node.name for node in aggregation_nodes} == aggregation_names
exec(compile(ast.Module(body=aggregation_nodes, type_ignores=[]), str(aggregation_path), "exec"), aggregation.__dict__)
sys.modules[aggregation.__name__] = aggregation
adapters = load_module(ALIAS + ".datasets.input_adapters", VALIDATION / "datasets" / "input_adapters.py")
validator_module = load_module(ALIAS + ".validator", VALIDATION / "validator.py")
MetricField = validator_module.MetricField
Validator = validator_module.Validator


def load_interpolated_buoy_wrapper():
    """Use production interpolation and KDTree without the ESMF package import."""
    namespace = dict(np=np, KDTree=KDTree, BallTree=BallTree, Mapping=Mapping)
    for path, names in (
        (VALIDATION / "inv_dist_interp.py", {"InvDistTree_np", "gauss_function_np"}),
        (VALIDATION / "datasets" / "utils.py", {"InterpolatedOverBuoysDataset", "lat_lon_from_grid"}),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert {node.name for node in nodes} == names
        selected = ast.Module(body=nodes, type_ignores=[])

        class PrivateMetricFieldImport(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module == "libs.validation.validator":
                    node.module = ALIAS + ".validator"
                return node

        selected = PrivateMetricFieldImport().visit(selected)
        exec(compile(selected, str(path), "exec"), namespace)
    return namespace["InterpolatedOverBuoysDataset"]


InterpolatedOverBuoysDataset = load_interpolated_buoy_wrapper()


def tearDownModule():
    for name in tuple(sys.modules):
        if name == ALIAS or name.startswith(ALIAS + "."):
            sys.modules.pop(name)
    sys.modules.update(_previous_alias_modules)


def times(*hours):
    return np.datetime64("2023-12-31", "ns") + np.asarray(hours).astype("timedelta64[h]")


def batch(values, *, bids=None, datetimes=None, var_names=None, units=None, coords=None):
    values = np.asarray(values, dtype=np.float32)
    n, t, v = values.shape
    bids = bids if bids is not None else [f"simba:{i}" for i in range(n)]
    datetimes = times(*range(t)) if datetimes is None else np.asarray(datetimes, dtype="datetime64[ns]")
    var_names = tuple(var_names or ("ice_thickness", "snow_thickness")[:v])
    units = tuple(units or ["m"] * v)
    coords = np.broadcast_to([80.0, 120.0], (n, t, 2)).copy() if coords is None else np.asarray(coords)
    valid = np.isfinite(values)
    coord_valid = np.isfinite(coords).all(axis=-1)
    return buoy_types.BuoyBatch(
        bids=np.asarray(bids), coords=coords, datetimes=datetimes,
        variables=values, var_names=var_names, units=units, valid=valid,
        coord_valid=coord_valid,
        coord_times=np.where(coord_valid, datetimes[None, :], np.datetime64("NaT", "ns")),
        value_times=np.where(valid, datetimes[None, :, None], np.datetime64("NaT", "ns")),
        uncertainty=np.full(values.shape, np.nan),
        metadata={bid: {"source": bid.split(":")[0]} for bid in bids},
    )


def prediction(values, reference, **metadata):
    meta = dict(
        dims=("buoy", "time", "variable"), bids=reference.bids,
        datetimes=reference.datetimes, var_names=reference.var_names, units=reference.units,
    )
    meta.update(metadata)
    return MetricField(values, **meta)


def error_field(prepared, metric):
    return adapters.make_metric_field(
        metric.compute(*prepared.arrays), metric=metric,
        context=prepared.context, input_dims=prepared.input_dims,
    )


class ArrayInputAdapterTests(unittest.TestCase):
    def test_identity_preserves_subclasses_metadata_and_broadcasting(self):
        left = MetricField(np.arange(6.0).reshape(2, 3), dataset="left", left_only=True)
        right = MetricField(np.array([1.0, 2.0, 3.0]), dataset="right")
        metric = metrics.Difference()
        prepared = adapters.ArrayInputAdapter().prepare([left, right], metric=metric)
        self.assertIs(prepared.arrays[0], left)
        self.assertIs(prepared.arrays[1], right)
        self.assertTrue(prepared.has_valid_samples)
        field = error_field(prepared, metric)
        np.testing.assert_array_equal(field, [[-1, -1, -1], [2, 2, 2]])
        self.assertEqual(field.meta["dataset"], "right")
        self.assertTrue(field.meta["left_only"])
        self.assertNotIn("dims", field.meta)

    def test_all_nan_retains_legacy_metric_behavior_and_empty_skips(self):
        metric = metrics.Difference()
        prepared = adapters.ArrayInputAdapter().prepare([np.array([np.nan]), np.array([np.nan])], metric=metric)
        self.assertTrue(prepared.has_valid_samples)
        empty = adapters.ArrayInputAdapter().prepare([np.empty(0), np.empty(0)], metric=metric)
        self.assertFalse(empty.has_valid_samples)

    def test_three_inputs_remain_in_original_order(self):
        values = (np.array([0.75]), np.array([0.5]), np.array([1.0]))
        metric = metrics.ReversedSkillScore()
        prepared = adapters.ArrayInputAdapter().prepare(values, metric=metric)
        self.assertEqual(len(prepared.arrays), 3)
        for original, prepared_array in zip(values, prepared.arrays):
            self.assertIs(prepared_array, original)
        np.testing.assert_allclose(error_field(prepared, metric), [0.5])


class BuoyInputAdapterTests(unittest.TestCase):
    def test_permuted_ids_times_variables_and_dimensions_align_by_labels(self):
        reference = batch(
            np.arange(12.0).reshape(2, 3, 2), bids=["simba:a", "crrel:b"],
            datetimes=times(23, 24, 25),
        )
        model_values = (reference.variables + 1)[[1, 0]][:, [2, 0, 1]][:, :, [1, 0]]
        model = prediction(
            model_values.transpose(1, 2, 0), reference,
            dims=("time", "variable", "buoy"), bids=reference.bids[[1, 0]],
            datetimes=reference.datetimes[[2, 0, 1]],
            var_names=reference.var_names[::-1], units=reference.units[::-1],
        )
        metric = metrics.Difference()
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metric)
        self.assertEqual(prepared.input_dims, ("buoy", "time", "variable"))
        np.testing.assert_array_equal(prepared.arrays[0], reference.variables + 1)
        np.testing.assert_array_equal(prepared.arrays[1], reference.variables)
        field = error_field(prepared, metric)
        np.testing.assert_array_equal(field, np.ones((2, 3, 2)))
        np.testing.assert_array_equal(field.meta["bids"], reference.bids)
        np.testing.assert_array_equal(field.meta["datetimes"], reference.datetimes)
        self.assertEqual(field.meta["dims"], ("buoy", "time", "variable"))

    def test_missing_model_buoys_and_times_remain_on_reference_axes(self):
        reference = batch(np.ones((2, 3, 1)), bids=["simba:a", "crrel:b"])
        model = prediction(
            np.array([[[4.0]]]), reference,
            bids=np.array(["crrel:b"]), datetimes=times(1),
        )
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metrics.MSE())
        expected = np.zeros((2, 3, 1), dtype=bool)
        expected[1, 1, 0] = True
        np.testing.assert_array_equal(prepared.valid, expected)
        self.assertTrue(np.isnan(prepared.arrays[0][~expected]).all())
        self.assertTrue(np.isnan(prepared.arrays[1][~expected]).all())
        self.assertEqual(prepared.arrays[0][1, 1, 0], 4.0)
        np.testing.assert_array_equal(prepared.context.metadata["bids"], reference.bids)

    def test_masks_intersect_finite_measurements_positions_and_model_qc(self):
        reference = batch(
            [[[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]],
            coords=[[[80, 120], [81, 121], [np.nan, np.nan]]],
        )
        model = prediction(np.full((1, 3, 2), 10.0), reference,
                           valid=np.array([[[True, False], [True, True], [True, True]]]))
        originals = (reference.variables.copy(), model.copy())
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metrics.MSE())
        expected = np.array([[[True, False], [True, False], [False, False]]])
        np.testing.assert_array_equal(prepared.valid, expected)
        for arr in prepared.arrays:
            self.assertTrue(np.isnan(arr[~expected]).all())
        np.testing.assert_array_equal(reference.variables, originals[0])
        np.testing.assert_array_equal(model, originals[1])

    def test_bare_arrays_require_explicit_alignment_opt_in(self):
        reference = batch(np.ones((1, 2, 1)))
        model = np.ones((1, 2, 1))
        with self.assertRaises(ValueError):
            adapters.BuoyInputAdapter().prepare([model, reference], metric=metrics.MSE())
        prepared = adapters.BuoyInputAdapter(assume_aligned=True).prepare([model, reference], metric=metrics.MSE())
        np.testing.assert_array_equal(prepared.arrays[0], model)

    def test_explicit_raw_array_axes_are_used_even_for_equal_dimension_sizes(self):
        reference = batch(np.arange(8.0).reshape(2, 2, 2))
        model = (reference.variables + 3.0).transpose(1, 2, 0)
        prepared = adapters.BuoyInputAdapter(
            assume_aligned=True, array_dims=("time", "variable", "buoy"),
        ).prepare([model, reference], metric=metrics.Difference())
        np.testing.assert_array_equal(prepared.arrays[0], reference.variables + 3.0)
        np.testing.assert_array_equal(prepared.arrays[1], reference.variables)

    def test_assumed_alignment_preserves_nonidentity_metadata_but_rejects_partial_identity(self):
        reference = batch(np.arange(12.0).reshape(2, 3, 2))
        raw = (reference.variables + 3.0).transpose(1, 2, 0)
        adapter = adapters.BuoyInputAdapter(
            assume_aligned=True, array_dims=("time", "variable", "buoy"),
        )
        lead_h = np.array([0, 6, 12])
        prepared = adapter.prepare([MetricField(raw, lead_h=lead_h), reference], metric=metrics.Difference())
        np.testing.assert_array_equal(prepared.arrays[0], reference.variables + 3.0)
        np.testing.assert_array_equal(prepared.context.metadata["inputs"][0]["metadata"]["lead_h"], lead_h)
        partial_metadata = (
            {"dims": ("time", "variable", "buoy")},
            {"bids": reference.bids},
            {"datetimes": reference.datetimes},
            {"var_names": reference.var_names},
            {"units": reference.units},
        )
        for metadata in partial_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                adapter.prepare([MetricField(raw, **metadata), reference], metric=metrics.Difference())

    def test_complex_point_values_are_rejected_before_float_coercion(self):
        reference = batch(np.ones((1, 2, 1)))
        model = prediction(np.ones((1, 2, 1), dtype=complex) + 2j, reference)
        with self.assertRaises(TypeError):
            adapters.BuoyInputAdapter().prepare([model, reference], metric=metrics.MSE())

    def test_missing_array_axis_metadata_is_not_inferred_from_shape(self):
        reference = batch(np.ones((2, 2, 2)))
        model = prediction(reference.variables, reference)
        model.meta.pop("dims")
        with self.assertRaises(ValueError):
            adapters.BuoyInputAdapter().prepare([model, reference], metric=metrics.MSE())

    def test_duplicate_labels_unknown_units_and_mismatched_coordinates_reject(self):
        reference = batch(np.ones((2, 2, 1)))
        invalid_metadata = (
            {"bids": ["simba:0", "simba:0"]},
            {"datetimes": times(0, 0)},
            {"units": ("kelvin",)},
            {"coords": reference.coords + np.array([0.0, 1.0])},
            {"dims": ("time", "time", "variable")},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                adapters.BuoyInputAdapter().prepare(
                    [prediction(reference.variables, reference, **metadata), reference], metric=metrics.MSE(),
                )

    def test_nested_identifier_and_variable_labels_are_rejected(self):
        reference = batch(np.ones((2, 3, 2)))
        invalid_metadata = (
            {"bids": reference.bids.reshape(-1, 1)},
            {"var_names": np.asarray(reference.var_names).reshape(-1, 1)},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                adapters.BuoyInputAdapter().prepare(
                    [prediction(reference.variables, reference, **metadata), reference], metric=metrics.MSE(),
                )

    def test_variable_selection_mapping_and_length_conversion(self):
        reference = batch([[[1.5, 0.3], [2.0, 0.4]]])
        model = prediction(reference.variables * 100.0, reference,
                           var_names=("hi", "hs"), units=("cm", "cm"))
        prepared = adapters.BuoyInputAdapter(
            variables=("snow_thickness",),
            variable_maps={0: {"ice_thickness": "hi", "snow_thickness": "hs"}},
        ).prepare([model, reference], metric=metrics.MSE())
        self.assertEqual(prepared.arrays[0].shape, (1, 2, 1))
        np.testing.assert_allclose(prepared.arrays[0], reference.variables[:, :, 1:2])
        field = error_field(prepared, metrics.MSE())
        np.testing.assert_allclose(field, 0, atol=1e-12)
        self.assertEqual(field.meta["units"], ("m^2",))

    def test_each_input_retains_aligned_actual_times_uncertainty_and_provenance(self):
        reference = batch(np.arange(12.0).reshape(2, 3, 2))
        reference = replace(
            reference,
            value_times=reference.value_times - np.timedelta64(5, "m"),
            coord_times=reference.coord_times - np.timedelta64(2, "m"),
        )
        def permute(values):
            return values[[1, 0]][:, [2, 0, 1]][:, :, [1, 0]].transpose(1, 2, 0)

        actual_times = reference.value_times - np.timedelta64(30, "m")
        coord_times = reference.coord_times - np.timedelta64(10, "m")
        model = prediction(
            permute(reference.variables * 100), reference,
            dims=("time", "variable", "buoy"), bids=reference.bids[[1, 0]],
            datetimes=reference.datetimes[[2, 0, 1]],
            var_names=reference.var_names[::-1], units=("cm", "cm"),
            value_times=permute(actual_times),
            coord_times=coord_times[[1, 0]][:, [2, 0, 1]],
            uncertainty=permute(np.full(reference.variables.shape, 20.0)),
            init_time=times(-12)[0],
        )
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metrics.MSE())
        contexts = prepared.context.metadata["inputs"]
        np.testing.assert_array_equal(contexts[0]["value_times"], actual_times)
        np.testing.assert_array_equal(contexts[0]["coord_times"], coord_times)
        np.testing.assert_allclose(contexts[0]["uncertainty"], .2)
        np.testing.assert_array_equal(contexts[1]["value_times"], reference.value_times)
        np.testing.assert_array_equal(contexts[1]["coord_times"], reference.coord_times)
        self.assertEqual(contexts[0]["metadata"]["init_time"], times(-12)[0])
        self.assertEqual(prepared.context.metadata["reference_index"], 1)

    def test_vector_requires_all_components_and_preserves_result_geometry(self):
        reference = batch([[[1.0, 0.0], [1.0, 0.0]]], var_names=("u", "v"), units=("m/s", "m/s"))
        model = prediction(np.array([[[0.0, 1.0], [1.0, np.nan]]]), reference)
        metric = metrics.AngleError(var_axis=-1)
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metric)
        self.assertTrue(np.isnan(prepared.arrays[0][0, 1]).all())
        self.assertTrue(np.isnan(prepared.arrays[1][0, 1]).all())
        field = error_field(prepared, metric)
        self.assertEqual(field.shape, (1, 2))
        self.assertAlmostEqual(field[0, 0], 90)
        self.assertTrue(np.isnan(field[0, 1]))
        self.assertEqual(field.meta["dims"], ("buoy", "time"))
        np.testing.assert_array_equal(field.meta["coords"], reference.coords)

    def test_default_vector_axis_rejects_instead_of_reducing_buoys(self):
        reference = batch(np.ones((2, 2, 2)), var_names=("u", "v"), units=("m/s", "m/s"))
        with self.assertRaises(ValueError):
            adapters.BuoyInputAdapter().prepare(
                [prediction(reference.variables, reference), reference], metric=metrics.AngleError(),
            )

    def test_sequential_vector_metric_removes_only_the_variable_axis(self):
        reference = batch([[[1.0, 2.0], [3.0, 4.0]]], var_names=("u", "v"), units=("m/s", "m/s"))
        model = prediction(reference.variables + np.array([3.0, 4.0]), reference)
        metric = metrics.SequentialMetric(metrics.Difference(), metrics.VectorNorm(var_axis=-1))
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metric)
        field = error_field(prepared, metric)
        np.testing.assert_allclose(field, [[5.0, 5.0]])
        self.assertEqual(field.meta["dims"], ("buoy", "time"))

    def test_repeated_squaring_retains_unambiguous_unit_exponents(self):
        reference = batch([[[1.0], [2.0]]])
        model = prediction(np.array([[[3.0], [5.0]]]), reference)
        metric = metrics.SequentialMetric(metrics.MSE(), metrics.StatSquared())
        prepared = adapters.BuoyInputAdapter().prepare([model, reference], metric=metric)
        field = error_field(prepared, metric)
        np.testing.assert_allclose(field, [[[16.0], [81.0]]])
        self.assertEqual(field.meta["units"], ("(m^2)^2",))

    def test_all_invalid_and_empty_buoy_inputs_have_no_samples(self):
        reference = batch(np.ones((1, 2, 1)))
        prepared = adapters.BuoyInputAdapter().prepare(
            [prediction(np.full((1, 2, 1), np.nan), reference), reference], metric=metrics.MSE(),
        )
        self.assertFalse(prepared.has_valid_samples)
        empty = buoy_types.BuoyBatch.empty(times(0, 1), ("ice_thickness",), ("m",))
        prepared = adapters.BuoyInputAdapter().prepare([empty, empty], metric=metrics.MSE())
        self.assertFalse(prepared.has_valid_samples)


class MemoryDataset:
    def __init__(self, name, field):
        self.name, self.field = name, field

    def __getitem__(self, when):
        return self.field


class CaptureAggregator:
    def init_accumulator(self, shape):
        return {"shape": shape, "entries": []}

    def accumulate(self, accumulator, field, when):
        accumulator["entries"].append((np.asarray(field).copy(), dict(field.meta), when))

    def finalize(self, accumulator):
        return accumulator


class InterpolatedBuoyValidationTests(unittest.TestCase):
    def setUp(self):
        self.progress = patch.object(validator_module, "tqdm", lambda values, **kwargs: values)
        self.progress.start()
        self.addCleanup(self.progress.stop)
        self.when = date(2024, 1, 1)

    def case(self, *, n=3, t=1, v=1, base_time_axis=0, **wrapper_options):
        lat = np.array([[80.0, 80.0], [81.0, 81.0]])
        lon = np.array([[100.0, 101.0], [100.0, 101.0]])
        grid_values = (
            100.0 * np.arange(t)[:, None, None, None]
            + 10.0 * np.arange(v)[None, :, None, None]
            + np.arange(4.0).reshape(1, 1, 2, 2)
        )
        point_indices = (np.arange(n)[:, None] + np.arange(t)[None, :]) % 4
        coords = np.stack((lat.ravel()[point_indices], lon.ravel()[point_indices]), axis=-1)
        expected = np.empty((n, t, v))
        for i in range(n):
            for j in range(t):
                expected[i, j] = grid_values[j].reshape(v, -1)[:, point_indices[i, j]]
        reference = batch(expected + 0.5, coords=coords)
        if base_time_axis == 1:
            grid_values = grid_values.transpose(1, 0, 2, 3)
        model_ds = MemoryDataset("model", MetricField(grid_values, lead_h=np.arange(t) * 6))
        model_ds.grid = dict(lat=lat, lon=lon)
        observation_ds = MemoryDataset("observations", reference)
        wrapper = InterpolatedOverBuoysDataset(
            model_ds, observation_ds, base_time_axis=base_time_axis, **wrapper_options,
        )
        return wrapper, observation_ds, expected

    def test_real_interpolation_identity_and_paired_metrics_with_unequal_axes(self):
        for n, t, v, base_time_axis in ((3, 1, 1, 0), (3, 2, 2, 1)):
            with self.subTest(n=n, t=t, v=v, base_time_axis=base_time_axis):
                wrapper, observations, expected = self.case(n=n, t=t, v=v, base_time_axis=base_time_axis)
                output = StringIO()
                with redirect_stdout(output):
                    sampled = wrapper[self.when]
                    validator = Validator(
                        [wrapper, observations], [metrics.IdentityStat(), metrics.MAE(), metrics.Difference()],
                        [CaptureAggregator(), aggregation.AverageAggregator(), aggregation.RawFieldAggregator()],
                        input_adapter=adapters.BuoyInputAdapter(
                            reference_index=-1, assume_aligned=True,
                        ),
                    )
                    validator.run([self.when], show_errors=True)
                self.assertEqual(output.getvalue(), "")
                expected_dims = (("time", "variable", "buoy") if base_time_axis == 0
                                 else ("variable", "time", "buoy"))
                self.assertEqual(sampled.meta["dims"], expected_dims)
                np.testing.assert_array_equal(sampled.meta["bids"], observations.field.bids)
                np.testing.assert_array_equal(sampled.meta["coords"], observations.field.coords)
                results = validator.summarize()
                model_identity = results["identity"][(wrapper.name,)]["CaptureAggregator"]["entries"][0]
                observed_identity = results["identity"][(observations.name,)]["CaptureAggregator"]["entries"][0]
                np.testing.assert_allclose(model_identity[0], expected)
                np.testing.assert_allclose(observed_identity[0], expected + 0.5)
                np.testing.assert_array_equal(model_identity[1]["inputs"][0]["metadata"]["lead_h"], np.arange(t) * 6)
                pair = (wrapper.name, observations.name)
                np.testing.assert_allclose(results["mae"][pair]["CaptureAggregator"]["entries"][0][0], 0.5)
                np.testing.assert_allclose(results["difference"][pair]["CaptureAggregator"]["entries"][0][0], -0.5)
                self.assertAlmostEqual(results["mae"][pair]["AverageAggregator"], 0.5)
                self.assertAlmostEqual(results["difference"][pair]["AverageAggregator"], -0.5)
                self.assertAlmostEqual(results["identity"][(wrapper.name,)]["AverageAggregator"], float(expected.mean()))
                raw_model = results["identity"][(wrapper.name,)]["RawFieldAggregator"][self.when]
                np.testing.assert_allclose(raw_model, expected)
                np.testing.assert_array_equal(raw_model.meta["bids"], observations.field.bids)
                self.assertEqual(raw_model.meta["dims"], ("buoy", "time", "variable"))

    def test_model_unary_identity_does_not_inherit_missing_observation_measurements(self):
        wrapper, observations, expected = self.case()
        values = observations.field.variables.copy()
        values[0, 0, 0] = np.nan
        observations.field = batch(values, coords=observations.field.coords)
        validator = Validator(
            [wrapper, observations], [metrics.IdentityStat(), metrics.MAE()],
            [aggregation.RawFieldAggregator()],
            input_adapter=adapters.BuoyInputAdapter(reference_index=-1, assume_aligned=True),
        )
        validator.run([self.when])
        results = validator.summarize()
        model = results["identity"][(wrapper.name,)]["RawFieldAggregator"][self.when]
        observed = results["identity"][(observations.name,)]["RawFieldAggregator"][self.when]
        paired = results["mae"][(wrapper.name, observations.name)]["RawFieldAggregator"][self.when]
        np.testing.assert_allclose(model, expected)
        self.assertTrue(np.isnan(observed[0, 0, 0]))
        self.assertTrue(np.isnan(paired[0, 0, 0]))
        np.testing.assert_allclose(paired[1:], 0.5)

    def test_declared_model_names_and_units_support_strict_adapter(self):
        wrapper, observations, expected = self.case(var_names=("ice_thickness",), units=("cm",))
        wrapper.base_ds.field = wrapper.base_ds.field * 100
        sampled = wrapper[self.when]
        prepared = adapters.BuoyInputAdapter(reference_index=-1).prepare(
            [sampled, observations.field], metric=metrics.Difference(),
        )
        np.testing.assert_allclose(prepared.arrays[0], expected)
        self.assertEqual(sampled.meta["units"], ("cm",))
        field = error_field(prepared, metrics.Difference())
        np.testing.assert_allclose(field, -0.5)

    def test_unknown_model_semantics_require_explicit_assumption(self):
        wrapper, observations, _ = self.case()
        sampled = wrapper[self.when]
        self.assertNotIn("var_names", sampled.meta)
        self.assertNotIn("units", sampled.meta)
        with self.assertRaises(ValueError):
            adapters.BuoyInputAdapter(reference_index=-1).prepare(
                [sampled, observations.field], metric=metrics.Difference(),
            )

    def test_partial_invalid_coordinates_retain_other_buoys_and_missing_values(self):
        wrapper, observations, expected = self.case(t=2)
        coords = observations.field.coords.copy()
        coords[1, 0] = np.nan
        observations.field = batch(observations.field.variables, coords=coords)
        sampled = wrapper[self.when]
        prepared = adapters.BuoyInputAdapter(reference_index=-1, assume_aligned=True).prepare(
            [sampled, observations.field], metric=metrics.Difference(),
        )
        expected[1, 0] = np.nan
        np.testing.assert_allclose(prepared.arrays[0], expected, equal_nan=True)
        self.assertFalse(prepared.valid[1, 0, 0])
        self.assertEqual(int(prepared.valid.sum()), 5)

    def test_all_invalid_and_empty_coords_do_not_query_tree(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                wrapper, observations, _ = self.case()
                reference = observations.field
                if empty:
                    observations.field = buoy_types.BuoyBatch.empty(reference.datetimes, reference.var_names, reference.units)
                else:
                    observations.field = batch(reference.variables, coords=np.full(reference.coords.shape, np.nan))
                with patch.object(wrapper.interpolator, "set_queries", wraps=wrapper.interpolator.set_queries) as query:
                    sampled = wrapper[self.when]
                query.assert_not_called()
                self.assertEqual(sampled.shape, (1, 1, 0 if empty else 3))
                self.assertTrue(np.isnan(sampled).all())
                prepared = adapters.BuoyInputAdapter(reference_index=-1, assume_aligned=True).prepare(
                    [sampled, observations.field], metric=metrics.MAE(),
                )
                self.assertFalse(prepared.has_valid_samples)

    def test_legacy_coords_only_observations_keep_layout_and_grid_metadata(self):
        wrapper, observations, expected = self.case(t=2)
        observations.field = SimpleNamespace(coords=observations.field.coords.transpose(1, 0, 2))
        sampled = wrapper[self.when]
        np.testing.assert_allclose(sampled, expected.transpose(1, 2, 0))
        np.testing.assert_array_equal(sampled.meta["lead_h"], [0, 6])
        self.assertNotIn("bids", sampled.meta)

    def test_explicit_model_times_must_match_observation_query_times(self):
        wrapper, _, _ = self.case(t=2)
        wrapper.base_ds.field.meta["datetimes"] = times(0, 2)
        with self.assertRaises(ValueError):
            wrapper[self.when]


class ValidatorInputAdapterTests(unittest.TestCase):
    def setUp(self):
        self.progress = patch.object(validator_module, "tqdm", lambda values, **kwargs: values)
        self.progress.start()
        self.addCleanup(self.progress.stop)
        self.when = date(2024, 1, 1)

    def test_legacy_positional_constructor_broadcasting_and_metadata(self):
        datasets = [
            MemoryDataset("model", MetricField(np.arange(6.0).reshape(2, 3), source="model")),
            MemoryDataset("reference", MetricField(np.ones(3), source="reference")),
        ]
        # The original eight positional constructor arguments stay valid.
        validator = Validator(datasets, [metrics.Difference()], [CaptureAggregator()],
                              self.when, self.when, "D", None, None)
        validator.run([self.when])
        summary = validator.summarize()["difference"][("model", "reference")]["CaptureAggregator"]
        np.testing.assert_array_equal(summary["entries"][0][0], [[-1, 0, 1], [2, 3, 4]])
        self.assertEqual(summary["entries"][0][1]["source"], "reference")
        validator.run([self.when])
        self.assertEqual(len(summary["entries"]), 1)
        self.assertEqual(validator.processed_dates, {self.when})

    def test_buoy_metric_receives_arrays_and_aggregator_receives_point_context(self):
        reference = batch([[[1.0], [2.0]]])
        model = prediction(np.array([[[3.0], [5.0]]]), reference)
        validator = Validator(
            [MemoryDataset("model", model), MemoryDataset("reference", reference)],
            [metrics.MSE()], [CaptureAggregator()], input_adapter=adapters.BuoyInputAdapter(),
        )
        validator.run([self.when])
        result = validator.summarize()["mse"][("model", "reference")]["CaptureAggregator"]
        np.testing.assert_array_equal(result["entries"][0][0], [[[4.0], [9.0]]])
        np.testing.assert_array_equal(result["entries"][0][1]["bids"], reference.bids)

    def test_no_joint_observations_skips_aggregation_and_marks_date_processed(self):
        reference = batch(np.ones((1, 2, 1)))
        model = prediction(np.full((1, 2, 1), np.nan), reference)
        metric = metrics.MSE()
        validator = Validator(
            [MemoryDataset("model", model), MemoryDataset("reference", reference)],
            [metric], [CaptureAggregator()], input_adapter=adapters.BuoyInputAdapter(),
        )
        with patch.object(metric, "compute", wraps=metric.compute) as compute:
            validator.run([self.when])
        compute.assert_not_called()
        self.assertIsNone(validator.results["mse"][("model", "reference")]["CaptureAggregator"])
        self.assertEqual(validator.processed_dates, {self.when})

    def test_save_load_resumes_without_reprocessing(self):
        datasets = [MemoryDataset("a", np.ones((2, 2))), MemoryDataset("b", np.zeros((2, 2)))]
        validator = Validator(datasets, [metrics.MSE()], [CaptureAggregator()])
        validator.run([self.when])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "validation.pkl")
            validator.save(path)
            restored = Validator(datasets, [metrics.MSE()], [CaptureAggregator()], load_path=path)
            restored.run([self.when])
        self.assertEqual(restored.processed_dates, {self.when})
        result = restored.summarize()["mse"][("a", "b")]["CaptureAggregator"]
        self.assertEqual(len(result["entries"]), 1)
        np.testing.assert_array_equal(result["entries"][0][0], np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
