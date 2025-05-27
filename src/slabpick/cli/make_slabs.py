from argparse import ArgumentParser

import numpy as np

from slabpick.slab import Slab


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Generate slabs by projecting tomogram slices along the z-axis.",
    )
    # basic input/output arguments
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory containing input tomograms in MRC format",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--zthick",
        type=float,
        required=True,
        help="Thickness of each slab projection in Angstrom",
    )
    parser.add_argument(
        "--zslide",
        type=float,
        required=False,
        default=0,
        help="Thickness of the sliding window in Angstrom; 0 results in no overlap",
    )
    parser.add_argument(
        "--include_tags",
        type=str,
        nargs="+",
        required=False,
        default=["Vol"],
        help="List of substrings for volume inclusion",
    )
    parser.add_argument(
        "--exclude_tags",
        type=str,
        nargs="+",
        required=False,
        default=["EVN", "ODD"],
        help="List of substrings for volume exclusion",
    )
    parser.add_argument(
        "--filenames",
        type=str,
        required=False,
        help="Text file containing filename prefixes; suffix should be _Vol.mrc",
    )

    return parser.parse_args()


def main():
    config = parse_args()

    filenames = None
    if config.filenames:
        filenames = np.loadtxt(config.filenames, dtype=str)

    slabber = Slab(
        config.zthick,
        config.zslide,
    )
    slabber.generate_slabs(
        config.in_dir,
        config.out_dir,
        config.include_tags,
        config.exclude_tags,
        filenames,
    )


if __name__ == "__main__":
    main()
