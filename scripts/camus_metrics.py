import itertools
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from medpy import metric

from vital.data.camus.config import Label
from vital.metrics.evaluate.clinical.heart_us import compute_left_ventricle_volumes
from vital.utils.image.io import sitk_load

PATIENT_ID_REGEX = r"patient\d{4}"
IMG_FILENAME_PATTERN = "{patient_id}_{view}_{frame}{tag}.mhd"
VIEWS_MAPPING = {"a4c": "4CH", "a2c": "2CH"}
LABELS = {"endo": Label.LV, "epi": [Label.LV, Label.MYO]}
SCORES = {"dsc": metric.dc}
DISTANCES = {"hd": metric.hd, "mad": metric.assd}


def _compute_segmentation_metrics(
    prediction: np.ndarray, gt: np.ndarray, voxelspacing: Tuple[float, float]
) -> Dict[str, float]:
    metrics = {}
    for label_tag, label in LABELS.items():
        pred_mask, gt_mask = np.isin(prediction, label), np.isin(gt, label)

        # Compute the segmentation scores
        metrics.update({f"{label_tag}_{score}": score_fn(pred_mask, gt_mask) for score, score_fn in SCORES.items()})

        # Compute the segmentation distances (that require the voxelspacing)
        metrics.update(
            {
                f"{label_tag}_{dist}": dist_fn(pred_mask, gt_mask, voxelspacing=voxelspacing)
                for dist, dist_fn in DISTANCES.items()
            }
        )

    return metrics


def main():
    """Run the script."""
    import re
    from argparse import ArgumentParser

    import pandas as pd
    from tqdm.auto import tqdm

    parser = ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Root directory for the input masks to evaluate")
    parser.add_argument(
        "--input_tag",
        type=str,
        default="",
        help="Tag at the end of the input filenames identifying which files to select",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        help="Root directory for the reference masks used to measure segmentation accuracy metrics",
    )
    parser.add_argument(
        "--target_tag",
        type=str,
        default="_gt",
        help="Tag at the end of the target filenames identifying which files to select",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory under which to save the different metrics measured on the input",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add at the beginning of the names of the logged files to differentiate between runs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(
        patient_id_candidate
        for patient_id_candidate in set(file.stem.split("_")[0] for file in args.input_dir.rglob("*"))
        if re.match(PATIENT_ID_REGEX, patient_id_candidate)
    )

    clinical_metrics, segmentation_metrics = [], []
    for patient_id in tqdm(patient_ids, desc="Computing metrics for patients", unit="patient"):
        a2c_ed, a2c_info = sitk_load(
            list(
                args.input_dir.rglob(
                    IMG_FILENAME_PATTERN.format(patient_id=patient_id, view="2CH", frame="ED", tag=args.input_tag)
                )
            )[0]
        )
        a2c_es, _ = sitk_load(
            list(
                args.input_dir.rglob(
                    IMG_FILENAME_PATTERN.format(patient_id=patient_id, view="2CH", frame="ES", tag=args.input_tag)
                )
            )[0]
        )
        a2c_voxelspacing = a2c_info["spacing"][:2][::-1]
        a4c_ed, a4c_info = sitk_load(
            list(
                args.input_dir.rglob(
                    IMG_FILENAME_PATTERN.format(patient_id=patient_id, view="4CH", frame="ED", tag=args.input_tag)
                )
            )[0]
        )
        a4c_es, _ = sitk_load(
            list(
                args.input_dir.rglob(
                    IMG_FILENAME_PATTERN.format(patient_id=patient_id, view="4CH", frame="ES", tag=args.input_tag)
                )
            )[0]
        )
        a4c_voxelspacing = a4c_info["spacing"][:2][::-1]

        edv, esv = compute_left_ventricle_volumes(
            np.isin(a2c_ed, Label.LV),
            np.isin(a2c_es, Label.LV),
            a2c_voxelspacing,
            np.isin(a4c_ed, Label.LV),
            np.isin(a4c_es, Label.LV),
            a4c_voxelspacing,
        )
        ef = round(100 * (edv - esv) / edv, 1)

        clinical_metrics.append({"patient": patient_id, "ef": ef, "edv": edv, "esv": esv})

        if args.target_dir:
            for view, frame in itertools.product(["a2c", "a4c"], ["ed", "es"]):
                prediction = locals()[f"{view}_{frame}"]
                voxelspacing = locals()[f"{view}_voxelspacing"]

                gt, _ = sitk_load(
                    list(
                        args.target_dir.rglob(
                            IMG_FILENAME_PATTERN.format(
                                patient_id=patient_id,
                                view=VIEWS_MAPPING[view],
                                frame=frame.upper(),
                                tag=args.target_tag,
                            )
                        )
                    )[0]
                )

                frame_metrics = {
                    "patient": patient_id,
                    "view": view,
                    "frame": frame,
                    **_compute_segmentation_metrics(prediction, gt, voxelspacing),
                }
                segmentation_metrics.append(frame_metrics)

    clinical_metrics = pd.DataFrame.from_records(clinical_metrics, index="patient")
    clinical_metrics.to_csv(args.output_dir / f"{args.prefix}clinical_metrics.csv")

    if segmentation_metrics:
        segmentation_metrics = pd.DataFrame.from_records(segmentation_metrics, index=["patient", "view", "frame"])
        segmentation_metrics.to_csv(args.output_dir / f"{args.prefix}segmentation_metrics.csv")


if __name__ == "__main__":
    main()
