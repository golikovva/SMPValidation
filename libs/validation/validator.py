import datetime
import itertools
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from .aggregators import Aggregator
from .metrics import Metric


class Validator:
    """
    Validator orchestrates computation of metrics over datasets and aggregation.

    ``input_adapter`` prepares each metric's inputs and their shared context.
    It defaults to ArrayInputAdapter for existing grid-based validations.
    """

    def __init__(
        self,
        datasets: List[Any] = None,
        metrics: List[Metric] = None,
        aggregators: List[Aggregator] = None,
        start_date: datetime.date = None,
        end_date: datetime.date = None,
        date_step: str = 'D',
        combinator: object = None,
        load_path: Union[str, None] = None,
        *,
        input_adapter=None,
    ):
        if input_adapter is None:
            # Import lazily: dataset utilities also use MetricField from here.
            from .datasets.input_adapters import ArrayInputAdapter

            input_adapter = ArrayInputAdapter()
        self.datasets = datasets
        self.metrics = metrics
        self.aggregators = aggregators
        self.start_date = start_date
        self.end_date = end_date
        self.date_step = date_step
        self.processed_dates = set()
        self.combinator = combinator or itertools.combinations
        self.input_adapter = input_adapter
        if load_path:
            self._load_result(load_path)
        else:
            self._init_result()

    def _make_date_list(self) -> List[datetime.date]:
        dates = np.arange(self.start_date, self.end_date + np.timedelta64(1, self.date_step), np.timedelta64(1, self.date_step))
        return dates.astype('O').tolist()
        days = (self.end_date - self.start_date).days
        return [self.start_date + datetime.timedelta(days=i) for i in range(days)]

    def _init_result(self) -> None:
        """
        Initialize result storage:
        self.results[metric_name][metric_key][aggregator_name] -> accumulator
        where metric_key is a tuple of dataset names of length arity
        """
        self.results: Dict[str, Dict[Tuple[str,...], Dict[str, Any]]] = {}
        names = [ds.name for ds in self.datasets]

        for metric in self.metrics:
            mname = metric.name
            self.results[mname] = {}
            # combinations/permutations depending on if order matters
            combos = self.combinator(names, metric.arity)
            for combo in combos:
                key = tuple(name for name in combo)
                # initialize aggregator accumulators
                self.results[mname][key] = {}
                for agg in self.aggregators:
                    agg_name = agg.__class__.__name__
                    self.results[mname][key][agg_name] = None  # agg.init_accumulator(shape)

    def run(self, dates: List[datetime.date] = None, show_errors: bool = False) -> None:
        """
        Run validation over the specified date range (or provided list of dates).
        """
        from .datasets.input_adapters import make_metric_field

        if dates is None:
            dates = self._make_date_list()

        new_dates = [d for d in dates if d not in self.processed_dates]
        if not new_dates:
            return

        for date in tqdm(new_dates, desc="Running Validator"):
            # extract raw fields
            fields = {}
            for ds in self.datasets:
                arr = None
                try:
                    arr = ds[date]
                except Exception as e:
                    if show_errors:
                        print(f"Error getting {date} from {ds.name}: {e}")
                if arr is None:
                    if show_errors:
                        print(ds.name, 'is none for the', date)
                    continue
                fields[ds.name] = arr
            # compute each metric and aggregate
            for metric in self.metrics:
                combos = self.combinator(fields.keys(), metric.arity)
                for combo in combos:
                    inputs = [fields[i] for i in combo]
                    prepared = self.input_adapter.prepare(inputs, metric=metric)
                    if not prepared.has_valid_samples:
                        continue

                    errors = metric.compute(*prepared.arrays)
                    err_field = make_metric_field(
                        errors,
                        metric=metric,
                        context=prepared.context,
                        input_dims=prepared.input_dims,
                    )
                    
                    for agg in self.aggregators:
                        agg_name = agg.__class__.__name__
                        acc = self.results[metric.name][combo][agg_name]
                        if acc is None:
                            acc = agg.init_accumulator(err_field.shape)
                            self.results[metric.name][combo][agg_name] = acc
                        agg.accumulate(acc, err_field, date)

            self.processed_dates.update([date])

    def summarize(self) -> Dict[str, Dict[Tuple[str,...], Dict[str, Any]]]:
        """
        Finalize all aggregators and return summarized results.
        Returns a nested dict keyed by metric, metric_key, aggregator_name.
        Entries without any valid samples remain None.
        """
        summary: Dict[str, Dict[Tuple[str,...], Dict[str, Any]]] = {}
        for mname, combos in self.results.items():
            summary[mname] = {}
            for key, aggs in combos.items():
                summary[mname][key] = {}
                for agg_name, acc in aggs.items():
                    if acc is None:
                        summary[mname][key][agg_name] = None
                        continue
                    # find aggregator class by name
                    agg = next(a for a in self.aggregators if a.__class__.__name__ == agg_name)
                    summary[mname][key][agg_name] = agg.finalize(acc)
        return summary

    def save(self, path: str) -> None:
        """Save results and processed dates to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'processed_dates': self.processed_dates
            }, f)

    def _load_result(self, path: str) -> None:
        """Load results and processed dates from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.results = data['results']
        self.processed_dates = data['processed_dates']

class MetricField(np.ndarray):
    """
    ndarray with attached metadata.
    Ordinary aggregators can use it as a normal array.
    Special aggregators can read .meta.
    """

    def __new__(cls, input_array, **meta):
        obj = np.asarray(input_array).view(cls)
        obj.meta = dict(meta)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.meta = getattr(obj, "meta", {})

    def get_meta(self, key=None, default=None):
        if key is None:
            return self.meta
        return self.meta.get(key, default)
