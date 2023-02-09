import numpy as np
from scipy import ndimage
from skimage.morphology import disk

from vital.utils.image.us.measure import EchoMeasure


def cut_myo_along_lv_base_line(mask: np.ndarray, lv_label: int, myo_label: int, bg_label: int = 0) -> np.ndarray:
    """Cuts the myocardium mask (replacing pixels w/ background) so that it reaches no further than the LV base's line.

    Args:
        mask: (H, W), Segmentation map of the left ventricle and myocardium in an apical view of the heart.
        lv_label: Label of the left ventricle.
        myo_label: Label of the myocardium.
        bg_label: Label of the background.

    Returns:
        (H, W), Segmentation mask where the myocardium reaches no further than the LV base's line.
    """
    # Identify the points at the left/right corner of the LV's base
    lv_control_points = EchoMeasure.endo_epi_control_points(mask, lv_label, myo_label, "endo", 3)
    left_point, right_point = lv_control_points[0], lv_control_points[-1]

    # Compute y = ax + b form of the equation of LV base's line
    a = (right_point[0] - left_point[0]) / (right_point[1] - left_point[1])
    b = -(a * left_point[1] - left_point[0])

    def _is_below_lv_base_line(y: int, x: int) -> bool:
        """Whether the y,x coordinates provided fall below the LV base's line."""
        return y > a * x + b

    # Create a binary for the whole image that is positive above the LV base's line
    below_lv_base_line_mask = np.fromfunction(_is_below_lv_base_line, mask.shape)

    # Assign the pixels of the myocardium that fall below the LV base's line to the background
    myo_mask = np.isin(mask, myo_label)
    mask[myo_mask * below_lv_base_line_mask] = bg_label

    return mask


def generate_left_atrium(
    mask: np.ndarray, lv_label: int, myo_label: int, la_label: int, bg_label: int = 0
) -> np.ndarray:
    """Generates a left atrium mask based on the anatomical prior of a disk centered around the LV base's midpoint.

    Args:
        mask: (H, W), Segmentation map of the left ventricle and myocardium in an apical view of the heart.
        lv_label: Label of the left ventricle.
        myo_label: Label of the myocardium.
        la_label: Label of the left atrium.
        bg_label: Label of the background.

    Returns:
        (H, W), Segmentation mask where the atrium was generated based on the anatomical prior of a disk centered around
        the LV base's midpoint.
    """
    # Remove the already existing left atrium, if present
    mask[np.isin(mask, la_label)] = bg_label

    # Identify the points at the left/right corner of the LV's base
    lv_control_points = EchoMeasure.endo_epi_control_points(mask, lv_label, myo_label, "endo", 3)
    left_point, right_point = lv_control_points[0], lv_control_points[-1]

    # Compute the properties of the circle for which the two points represent the diameter
    mid_point = tuple((left_point + right_point) // 2)
    radius = int(np.linalg.norm(left_point - right_point) / 2)

    # Generate a mask of a disk centered around the LV's base
    disk_mask = np.zeros_like(mask)
    disk_mask[mid_point] = 1
    disk_mask = ndimage.binary_dilation(disk_mask, structure=disk(radius))

    # Compute the left atrium mask by taking the intersection between the disk and the background
    la_mask = np.isin(mask, bg_label) * disk_mask

    # Assign the pixels to the left atrium
    mask[la_mask] = la_label

    return mask


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    from tqdm.auto import tqdm

    from vital.data.camus.config import Label
    from vital.utils.image.io import sitk_load, sitk_save
    from vital.utils.path import remove_suffixes

    parser = ArgumentParser()
    parser.add_argument("--input_mask", type=Path, nargs="+", help="Segmentation mask(s) to process")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory under which to save the processed segmentation masks",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_cut",
        help="Suffix to add at the end of the name of the original files to generate the name of each output file",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for in_file in tqdm(args.input_mask, desc="Processing input masks", unit="mask"):
        filename = remove_suffixes(in_file).name
        suffixes = "".join(in_file.suffixes)
        out_file = args.output_dir / (filename + args.suffix + suffixes)

        mask, metadata = sitk_load(in_file)
        mask = cut_myo_along_lv_base_line(mask, Label.LV, Label.MYO, bg_label=Label.BG)
        mask = generate_left_atrium(mask, Label.LV, Label.MYO, Label.ATRIUM, bg_label=Label.BG)

        sitk_save(mask, out_file, origin=metadata.get("origin"), spacing=metadata.get("spacing"), dtype=np.uint8)


if __name__ == "__main__":
    main()
