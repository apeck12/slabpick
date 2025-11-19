import os
import json
import numpy as np
import pandas as pd
import starfile
from tqdm import tqdm
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
        required=False,
        help="Copick json file specifying coordinates for backwards compatibilty for single data session cases",
    )
    parser.add_argument(
        "--particle_name",
        type=str,
        required=True,
        help="Copick particle name",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        required=True,
        help="Copick session ID for input coordinates",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=True,
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
        required=True,
        help="Copick session ID for output",
    )
    parser.add_argument(
        "--user_id_out",
        type=str,
        required=True,
        help="Copick user ID for output",
    )
    parser.add_argument(
        "--rejected_set",
        action="store_true",
        help="Extract coordinates of the rejected particles in the star file",
    )
    
    return parser.parse_args()


def generate_config(config):
    """
    Store command line arguments in a json file written to same
    directory that contains the particle_map.csv file.
    """
    d_config = vars(config)

    input_list = ["rln_file", "map_file", "coords_file"]
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
    reconfig["parameters"] = {k: d_config[k] for k in parameter_list}

    out_dir = os.path.dirname(os.path.abspath(config.map_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rln_map_particles.json"), "w") as f:
        json.dump(reconfig, f, indent=4)


def curate_data_session(d_coords: dict, curated_map_session: pd.DataFrame) -> dict:
    """
    Retain the selected particles from a data session. The curated_map_
    session should contain only selected particles for one data session.
    
    Parameters
    ----------
    d_coords: dictionary mapping tomograms to coords
    curated_map_session: curated particle map for the session
    
    Returns
    -------
    d_coords_sel: dict mapping tomograms to selected coords
    """
    tomo_list = np.unique(curated_map_session.tomogram.values)
    d_coords_sel = {}
    for _i, tomo in enumerate(tomo_list):    
        tomo_indices = np.where(curated_map_session.tomogram.values == tomo)[0]
        particle_indices = curated_map_session.iloc[tomo_indices].particle.values
        d_coords_sel[tomo] = d_coords[tomo][np.unique(particle_indices)] 
    return d_coords_sel

        
def main():

    config = parse_args()

    if config.out_file is None:
        config.out_file = config.coords_file
    if config.particle_name_out is None:
        config.particle_name_out = config.particle_name
    generate_config(config)
    
    # map retained particles from Relion back to gallery tiles
    rln_particles = starfile.read(config.rln_file)["particles"]
    indices = np.array([fn.split("@")[0] for fn in rln_particles.rlnImageName.values]).astype(int)
    particles_map = pd.read_csv(config.map_file)
    if config.rejected_set:
        print("Selecting the rejected particles")
        indices = np.setdiff1d(np.arange(len(particles_map)), indices)
    curated_map = particles_map.iloc[indices]


    # retrieve sessions that will be used and write parameter config
    if 'session' not in particles_map.columns:
        if config.coords_file is None:
            raise ValueError("The --coords_file must be specified")
        session_list = [config.coords_file]
    else:
        session_list = np.unique(curated_map.session.values)
    config.coords_file = list(session_list)
    generate_config(config)

    # loop over data sessions
    counts = 0
    for sconfig in tqdm(session_list):
        # get original coordinates
        cp_interface = dataio.CopickInterface(sconfig)
        d_coords = cp_interface.get_all_coords(
            config.particle_name,
            user_id=config.user_id,
            session_id=config.session_id,
        )
        ini_particle_count = np.sum(np.array([d_coords[tomo].shape[0] for tomo in d_coords]),)

        # curate coordinates, retaining only the selected particles
        if 'session' not in particles_map.columns:
            curated_map_session = curated_map
        else:
            curated_map_session = curated_map[curated_map.session==sconfig]
        d_coords_sel = curate_data_session(d_coords, curated_map_session)
        final_particle_count = np.sum(np.array([d_coords_sel[tomo].shape[0] for tomo in d_coords_sel]))
        print(f"Retained {final_particle_count} of {ini_particle_count} from {sconfig}",)
        counts += final_particle_count

        # write retained coordinates to copick project
        root = CopickRootFSSpec.from_file(sconfig)
        dataio.coords_to_copick(root, d_coords_sel, config.particle_name_out, config.session_id_out, config.user_id_out)

    assert counts == len(rln_particles)

    
if __name__ == "__main__":
    main()
