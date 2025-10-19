import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import slabpick.dataio as dataio
import slabpick.stacker as stacker


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Normalize and contrast-invert a particle stack for Relion intake.",
    )
    parser.add_argument(
        "--in_stack",
        type=str,
        required=True,
        help="Particle stack .mrcs file",
    )
    parser.add_argument(
        "--out_stack",
        type=str,
        required=True,
        help="Normalized, contrast-inverted output stack",
    )
    parser.add_argument(
        "--apix",
        type=float,
        required=False,
        default=1,
        help="Tilt-series pixel size",
    )
    parser.add_argument(
        "--voxel_spacing",
        type=float,
        required=False,
        help="Stack voxel spacing",
    )
    parser.add_argument(
        "--keep_contrast",
        action="store_true",
        help="Do not invert contrast",
    )

    return parser.parse_args()


def main():
    
    config = parse_args()

    contrast = -1
    if config.keep_contrast:
        contrast = 1

    stack = dataio.load_mrc(config.in_stack).copy()
    if config.voxel_spacing is None:
        config.voxel_spacing = dataio.get_voxel_size(config.in_stack)
    
    stack = stacker.normalize_stack(stack)
    dataio.save_mrc(
        contrast*stack, 
        config.out_stack, 
        apix=config.voxel_spacing,
    )
    dataio.make_stack_starfile(
        config.out_stack,
        config.out_stack.replace(".mrcs", ".star"),
        apix=config.apix,
    )

    
if __name__ == "__main__":
    main()
