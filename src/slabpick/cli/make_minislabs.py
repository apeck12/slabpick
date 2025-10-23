import os
from argparse import ArgumentParser

import slabpick.minislab as minislab
from slabpick.settings import ProcessingConfigMakeMinislabs


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Generate minislabs, or per-particle 2D projections.",
    )
    
    parser.add_argument(
        "--in_coords",
        type=str,
        required=True,
        nargs="+",
        help="Coordinate file(s) as one or multiple starfiles or a copick config file",
    )
    parser.add_argument(
        "--in_vol",
        type=str,
        required=False,
        nargs="*",
        help="Directory containing volumes or a copick config file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for galleries and/or particle stack",
    )
    parser.add_argument(
        "--extract_shape",
        type=int,
        nargs=3,
        required=True,
        help="Subvolume extraction shape in (x,y,z) in Angstrom",
    )
    parser.add_argument(
        "--voxel_spacing",
        type=float,
        required=True,
        help="Pixel size of tomograms to extract minislabs from in Angstrom",
    )
    parser.add_argument(
        "--extension",
        type=str,
        required=False,
        default="mrc",
        help="Tomogram extension, either zarr or mrc",
    )
    parser.add_argument(
        "--tomo_type",
        type=str,
        required=False,
        help="Tomogram type if extracting from copick",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=False,
        help="User ID if coordinates from copick",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        required=False,
        help="Session ID if coordinates from copick",
    )
    parser.add_argument(
        "--particle_name",
        type=str,
        required=False,
        help="Particle name, required if coordinates from copick",
    )
    parser.add_argument(
        "--angles",
        type=float,
        required=False,
        nargs="+",
        default=[0],
        help="Tilt angles to apply to each particle",
    )
    parser.add_argument(
        "--gallery_shape",
        type=int,
        nargs=2,
        required=False,
        default=[16, 15],
        help="Number of gallery particles in (row,col) format",
    )
    parser.add_argument(
        "--make_stack",
        action="store_true",
        help="Make particle stack instead of galleries",
    )
    parser.add_argument(
        "--invert_contrast",
        action="store_true",
        help="Invert default contrast",
    )
    parser.add_argument(
        "--border_factor",
        type=float,
        default=1.1,
        required=False,
        help="Multiplicative factor times mean for gallery border value",
    )
    parser.add_argument(
        "--border_width",
        type=int,
        default=2,
        required=False,
        help="Number of border pixels for around each tile",
    )

    return parser.parse_args()


def generate_config(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)

    reconfig = {}
    reconfig["software"] = {"name": "slabpick", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in ["in_coords", "in_vol"]}
    reconfig["output"] = {k: d_config[k] for k in ["out_dir"]}

    used_keys = [list(reconfig[key].keys()) for key in reconfig]
    used_keys = [p for param in used_keys for p in param]
    param_keys = [key for key in d_config if key not in used_keys]
    reconfig["parameters"] = {k: d_config[k] for k in param_keys}

    reconfig = ProcessingConfigMakeMinislabs(**reconfig)

    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "make_minislabs.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))


def main():

    config = parse_args()
    generate_config(config)

    minislab.make_minislabs_multisession(
        config.in_coords,
        config.in_vol,
        config.out_dir,
        config.extract_shape,
        config.voxel_spacing,
        extension=config.extension,
        tomo_type=config.tomo_type,
        particle_name=config.particle_name,
        user_id=config.user_id,
        session_id=config.session_id,
        angles=config.angles,
        gshape=tuple(config.gallery_shape),
        make_stack=config.make_stack,
        invert_contrast=config.invert_contrast,
        border_factor=config.border_factor,
        border_width=config.border_width,
    )
    
    
if __name__ == "__main__":
    main()
