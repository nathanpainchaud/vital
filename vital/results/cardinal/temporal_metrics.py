import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from vital import get_vital_root
from vital.data.cardinal.utils.data_struct import View
from vital.data.cardinal.utils.itertools import Views
from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric
from vital.results.metrics import Metrics

logger = logging.getLogger(__name__)


class TemporalMetrics(Metrics):
    """Class that computes temporal coherence metrics on sequences of attributes' values."""

    desc = "temporal_metrics"
    ResultsCollection = Views
    input_choices = []
    default_attribute_statistics_cfg = get_vital_root() / "data/camus/statistics/image_attr_stats.yaml"

    def __init__(
        self,
        attribute_statistics_cfg: Union[str, Path],
        thresholds_cfg: Union[str, Path],
        inconsistent_frames_only: bool = False,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            attribute_statistics_cfg: File containing pre-computed statistics for each attribute, used to normalize
                their values.
            thresholds_cfg: File containing pre-computed thresholds on the acceptable temporal consistency metrics'
                values for each attribute.
            inconsistent_frames_only: For metrics apart from the proportion of inconsistent frames, only compute these
                metrics on the inconsistent frames. Only when `measure_thresholds` is `False`.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)

        # Load statistics on the attributes' distributions thresholds from the config files
        with open(attribute_statistics_cfg) as f:
            self._attrs_bounds = yaml.safe_load(f)
        with open(thresholds_cfg) as f:
            self._thresholds = yaml.safe_load(f)

        self._inconsistent_frames_only = inconsistent_frames_only

    def _extract_attributes_from_result(self, result: View, item_tag: str) -> Dict[str, np.ndarray]:
        attrs_data = result.attrs[item_tag]
        missing_attrs = [attr for attr in self._thresholds if attr not in attrs_data]
        if missing_attrs:
            logger.warning(
                f"Requested attributes {missing_attrs} were not available for '{result.id}'. The attributes listed "
                f"were thus ignored."
            )
        return {attr: attrs_data[attr] for attr in self._thresholds if attr not in missing_attrs}

    def process_result(self, result: View) -> Optional[Tuple[str, "TemporalMetrics.ProcessingOutput"]]:
        """Computes temporal coherence metrics on sequences of attributes' values.

        Args:
            result: Data structure holding all the sequence`s data.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Mapping between the metrics and their value for the sequence.
        """
        attrs = self._extract_attributes_from_result(result, self.input_tag)
        metrics = {}
        inconsistent_frames_by_attr, err_thresh_ratios_by_attr = {}, {}
        for attr, attr_vals in attrs.items():
            temporal_consistency_abs_err = np.abs(
                compute_temporal_consistency_metric(attr_vals, bounds=self._attrs_bounds[attr])
            )
            threshold = self._thresholds[attr]
            # Identifies the frames that are temporally inconsistent
            inconsistent_frames_by_attr[attr] = temporal_consistency_abs_err > threshold
            # Computes the ratios between the error and the tolerated threshold
            err_thresh_ratios = temporal_consistency_abs_err / threshold
            if self._inconsistent_frames_only:
                err_thresh_ratios = err_thresh_ratios[temporal_consistency_abs_err > threshold]
            err_thresh_ratios_by_attr[f"{attr}_error_to_threshold_ratio"] = err_thresh_ratios

        # Computes the ratio of frames that are temporally inconsistent, for each of the measured attributes
        metrics.update(
            {
                f"{attr}_inconsistent_frames_ratio": inconsistent_frames.mean()
                for attr, inconsistent_frames in inconsistent_frames_by_attr.items()
            }
        )
        # Identifies the frames that are temporally inconsistent w.r.t. any of the measured attributes
        metrics["temporally_inconsistent_frames_ratio"] = (
            np.array(list(inconsistent_frames_by_attr.values())).any(axis=0).mean()
        )
        # Compute the ratio of error to threshold for inconsistent frames, for each of the measured attributes
        metrics.update(
            {attr: err_thresh_ratios.mean() for attr, err_thresh_ratios in err_thresh_ratios_by_attr.items()}
        )
        # Compute the ratio of error to threshold for inconsistent frames, across all the measured attributes
        metrics["error_to_threshold_ratio"] = np.hstack(list(err_thresh_ratios_by_attr.values())).mean()
        # Compute whether the sequence as any temporal inconsistencies, between any instants and for any attributes
        metrics["temporal_consistency_errors"] = bool(metrics["temporally_inconsistent_frames_ratio"])

        return result.id, metrics

    def aggregate_outputs(
        self, outputs: Mapping[View.Id, "TemporalMetrics.ProcessingOutput"], output_path: Path
    ) -> None:
        """Override of the parent method to save aggregate and individual results separately (for index formatting)."""
        metrics = pd.DataFrame.from_dict(outputs, orient="index").rename_axis(index=["patient", "view"])
        agg_metrics = self._aggregate_metrics(metrics)

        # Save the individual and aggregated scores in two steps
        agg_metrics.to_csv(output_path, na_rep="Nan")
        with output_path.open("a") as f:
            f.write("\n")
        metrics.to_csv(output_path, mode="a", na_rep="Nan")

    def _aggregate_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Computes global statistics for the metrics computed over each result.

        Args:
            metrics: Metrics computed over each result.

        Returns:
            Global statistics on the metrics computed over each result.
        """
        # Define groups of columns that have to be aggregated differently
        frames_ratio_cols = metrics.columns[metrics.columns.str.contains("frames_ratio")]
        err_thresh_ratio_cols = metrics.columns[metrics.columns.str.contains("error_to_threshold_ratio")]
        temporally_inconsistent_indices = metrics["temporal_consistency_errors"]

        # Aggregate metrics with results reported on all sequences
        frames_ratio_agg_metrics = metrics.loc[temporally_inconsistent_indices, frames_ratio_cols].agg(
            ["mean", "std", "max", "min"]
        )

        # Aggregate metrics with results reported only on sequences w/ temporal inconsistencies
        err_thresh_ratio_agg_metrics = metrics[err_thresh_ratio_cols].agg(["mean", "std", "max", "min"])
        temporal_consistency_agg_metrics = metrics[["temporal_consistency_errors"]].agg(["sum"])

        # Merge aggregations, filling the join with w/ NaNs
        return pd.concat(
            [frames_ratio_agg_metrics, err_thresh_ratio_agg_metrics, temporal_consistency_agg_metrics],
            axis="columns",
        )

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for temporal metrics processor.

        Returns:
            Parser object for temporal metrics processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--attribute_statistics_cfg",
            type=Path,
            default=cls.default_attribute_statistics_cfg,
            help="File containing pre-computed statistics for each attribute, used to normalize their values",
        )
        parser.add_argument(
            "--thresholds_cfg",
            type=Path,
            default=get_vital_root() / "data/camus/statistics/attr_thresholds.yaml",
            help="File containing pre-computed thresholds on the acceptable temporal consistency metrics' values for "
            "each attribute",
        )
        parser.add_argument(
            "--inconsistent_frames_only",
            action="store_true",
            help="For metrics apart from the proportion of inconsistent frames, only compute these metrics on the "
            "inconsistent frames",
        )
        return parser


def main():
    """Run the script."""
    TemporalMetrics.main()


if __name__ == "__main__":
    main()
