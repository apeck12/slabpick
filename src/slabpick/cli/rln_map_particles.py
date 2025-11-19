import os
import json
import numpy as np
import pandas as pd
import starfile
from argparse import ArgumentParser
import slabpick.dataio as dataio
from copick.impl.filesystem import CopickRootFSSpec


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Generate starfile based on cryosparc-curated picks.",
    )
    parser.add_argument(
        "--rln_file",
        type=str,
        required=True,
        help="Relion class averaging starfile",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        required=True,
        help="Bookkeeping file mapping particles to particle stack",
    )
    parser.add_argument(
        "--coords_file",
        type=str,
        required=True,
        help="Copick json file specifying coordinates",
    )
    parser.add_argument(
        "--particle_name",
        type=str,
        required=False,
        help="Copick particle name",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        required=False,
        help="Copick session ID for input coordinates",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=False,
        help="Copick user ID for input coordinates",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=False,
        help="Output copick json or Relion-4 starfile",
    )
    parser.add_argument(
        "--particle_name_out",
        type=str,
        required=False,
        help="Copick particle name for output if different from input",
    )
    parser.add_argument(
        "--session_id_out",
        type=str,
        required=False,
        help="Copick session ID for output",
    )
    parser.add_argument(
        "--user_id_out",
        type=str,
        required=False,
        help="Copick user ID for output",
    )
    parser.add_argument(
        "--apix",
        type=float,
        required=False,
        help="Tilt-series pixel size, inverse of this will be applied if writing out a starfile",
    )
    parser.add_argument(
        "--rejected_set",
        action="store_true",
        help="Extract coordinates of the rejected particles in the star file",
    )
    
    return parser.parse_args()


def generate_config(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)

    input_list = ["rln_file", "map_file", "coords_file"]
    output_list = ["out_file"]
    parameter_list = [
        "particle_name",
        "session_id",
        "user_id",
        "apix",
        "particle_name_out",
        "session_id_out",
        "user_id_out",
        "rejected_set",
    ]

    reconfig = {}
    reconfig["software"] = {"name": "slabpick", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in input_list}
    reconfig["output"] = {k: d_config[k] for k in output_list}
    reconfig["parameters"] = {k: d_config[k] for k in parameter_list}

    out_dir = os.path.dirname(os.path.abspath(config.out_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rln_map_particles.json"), "w") as f:
        json.dump(reconfig, f, indent=4)


def main():

    config = parse_args()
    if config.out_file is None:
        config.out_file = config.coords_file
    if config.particle_name_out is None:
        config.particle_name_out = config.particle_name
    generate_config(config)
    
    # extract particle coordinates
    cp_interface = dataio.CopickInterface(config.coords_file)
    d_coords = cp_interface.get_all_coords(
        config.particle_name,
        user_id=config.user_id,
        session_id=config.session_id,
    )
    ini_particle_count = np.sum(
        np.array([d_coords[tomo].shape[0] for tomo in d_coords]),
    )
    
    # map retained particles from Relion back to gallery tiles
    rln_particles = starfile.read(config.rln_file)["particles"]
    indices = np.array([fn.split("@")[0] for fn in rln_particles.rlnImageName.values]).astype(int)
    particles_map = pd.read_csv(config.map_file)
    if config.rejected_set:
        print("Selecting the rejected particles")
        indices = np.setdiff1d(np.arange(len(particles_map)), indices)
    curated_map = particles_map.iloc[indices]
    
    # curate particles, retaining a particle if any of its tilts is selected
    d_coords_sel = {}
    tomo_list = np.unique(curated_map.tomogram.values)
    for _i, tomo in enumerate(tomo_list):
        tomo_indices = np.where(curated_map.tomogram.values == tomo)[0]
        particle_indices = curated_map.iloc[tomo_indices].particle.values
        d_coords_sel[tomo] = d_coords[tomo][np.unique(particle_indices)] 
        final_particle_count = np.sum(np.array([d_coords_sel[tomo].shape[0] for tomo in d_coords_sel]))
    print(f"Retained {final_particle_count} of {ini_particle_count}",)

    if os.path.splitext(config.out_file)[1] == '.json':
        print("Writing retained coordinates to a copick experiment")
        root = CopickRootFSSpec.from_file(config.out_file)
        dataio.coords_to_copick(root, d_coords_sel, config.particle_name_out, config.session_id_out, config.user_id_out)
    elif os.path.splitext(config.out_file)[1] == '.star':
        print("Writing retained coordinates to a Relion-4 star file")
        dataio.make_starfile(d_coords_sel, config.out_file, coords_scale=1.0/config.apix)
    else:
        raise ValueError(f"Unrecognized output argument, should be copick json or starfile")

    
if __name__ == "__main__":
    main()
