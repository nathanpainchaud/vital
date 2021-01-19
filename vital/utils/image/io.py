from numbers import Number
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK


def load_mhd(filepath: Path) -> Tuple[np.ndarray, Tuple[Tuple[Number, ...], ...]]:
    """Loads a mhd image and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # load image and save info
    image = SimpleITK.ReadImage(str(filepath))
    info = (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # create numpy array from the .mhd file and corresponding image
    im_array = np.squeeze(SimpleITK.GetArrayFromImage(image))

    return im_array, info