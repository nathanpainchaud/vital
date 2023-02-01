import logging

logger = logging.getLogger(__name__)


def main():
    """Run the script."""
    from argparse import ArgumentParser
    from pathlib import Path

    from tqdm.auto import tqdm

    from vital.data.cardinal.utils.itertools import Views
    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser = Views.add_args(parser)
    parser.add_argument(
        "--data_tag",
        type=str,
        required=True,
        help="Tag of the sequence to use to count the number of frames in each view",
    )
    parser.add_argument(
        "--subset_file",
        type=Path,
        help="Path to a file listing the IDs of a subset of patients on which to count frames",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    data_tag, subset_file = kwargs.pop("data_tag"), kwargs.pop("subset_file", None)

    patients_to_include = None
    if subset_file:
        patients_to_include = subset_file.read_text().splitlines()
    kwargs["include_patients"] = patients_to_include

    views = Views(**kwargs)
    nb_frames = sum(
        len(view.data[data_tag])
        for view in tqdm(views.values(), desc="Count number of frames in views", unit=Views.desc)
    )
    intro = f"Number of frames across {len(views)} sequences"
    if subset_file:
        intro += f" from subset '{subset_file.stem}'"
    logger.info(intro + f": {nb_frames}")


if __name__ == "__main__":
    main()
