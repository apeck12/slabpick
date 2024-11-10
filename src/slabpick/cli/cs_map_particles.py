import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import slabpick.csedit as csedit
import slabpick.dataio as dataio
from slabpick.settings import ProcessingConfigCsMapParticles


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Generate starfile based on cryosparc-curated picks.",
    )
    parser.add_argument(
        "--copick_json",
        type=str,
        required=False,
        help="Copick json file",
    )
    parser.add_argument(
        "--in_star",
        type=str,
        required=False,
        help="Starfile containing coordinates associated with map_file",
    )
    parser.add_argument(
        "--in_star_multiple",
        type=str,
        required=False,
        nargs="+",
        help="List of starfiles containing coordinates associated with map_file",
    )
    parser.add_argument(
        "--col_name",
        type=str,
        required=False,
        default="rlnTomoName",
        help="Tomogram column name in starfile(s)",
    )
    parser.add_argument(
        "--cs_file",
        type=str,
        required=True,
        help="Cryosparc extraction job, e.g. topaz_picked_particles.cs",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        required=True,
        help="Bookkeeping file mapping particles to gallery tiles",
    )
    parser.add_argument(
        "--particle_name",
        type=str,
        required=False,
        help="Copick particle name, required if using copick for coordinates",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        required=False,
        help="Copick session ID, required if using copick for coordinates",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=False,
        help="Copick user ID, required if using copick for coordinates",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output starfile",
    )
    parser.add_argument(
        "--coords_scale",
        type=float,
        required=False,
        default=1.0,
        help="Multiplicative factor to convert input coords in starfile(s) to Angstrom",
    )
    parser.add_argument(
        "--apix",
        type=float,
        required=True,
        help="Tilt-series pixel size, inverse of this will be applied when writing out starfile",
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

    input_list = ["cs_file", "map_file", "copick_json", "in_star", "in_star_multiple"]
    parameter_list = [
        "col_name",
        "particle_name",
        "session_id",
        "user_id",
        "coords_scale",
        "apix",
        "rejected_set",
    ]

    reconfig = {}
    reconfig["software"] = {"name": "slabpick", "version": "0.1.0"}
    reconfig["input"] = {k: d_config[k] for k in input_list}
    reconfig["output"] = {k: d_config[k] for k in ["out_file"]}
    reconfig["parameters"] = {k: d_config[k] for k in parameter_list}

    reconfig = ProcessingConfigCsMapParticles(**reconfig)

    out_dir = os.path.dirname(os.path.abspath(config.out_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cs_map_particles.json"), "w") as f:
        f.write(reconfig.model_dump_json(indent=4))


def main():
    config = parse_args()
    generate_config(config)

    # extract all particle coordinates
    if config.in_star:
        d_coords = dataio.read_starfile(
            config.in_star,
            coords_scale=config.coords_scale,
            col_name=config.col_name,
        )
    elif config.in_star_multiple:
        d_coords = dataio.combine_star_files(
            config.in_star_multiple,
            coords_scale=config.coords_scale,
            col_name=config.col_name,
        )
    elif config.copick_json:
        cp_interface = dataio.CopickInterface(config.copick_json)
        d_coords = cp_interface.get_all_coords(
            config.particle_name,
            user_id=config.user_id,
            session_id=config.session_id,
        )
    else:
        raise ValueError("Either a copick config or a starfile must be provided.")
    ini_particle_count = np.sum(
        np.array([d_coords[tomo].shape[0] for tomo in d_coords]),
    )

    # map retained particles in cryosparc to gallery tiles
    cs_extract = np.load(config.cs_file)
    particles_map = pd.read_csv(config.map_file)
    if len(particles_map) > 1e6:
        curated_map = csedit.curate_particles_map_iterative(
            cs_extract,
            particles_map,
            rejected_set=config.rejected_set,
        )
    else:
        curated_map = csedit.curate_particles_map(
            cs_extract,
            particles_map,
            rejected_set=config.rejected_set,
        )

    # curate particles -- currently a particle is kept if any of its tilts is selected
    d_coords_sel = {}
    tomo_list = np.unique(curated_map.tomogram.values)
    for _i, tomo in enumerate(tomo_list):
        tomo_indices = np.where(curated_map.tomogram.values == tomo)[0]
        particle_indices = curated_map.iloc[tomo_indices].particle.values
        if config.copick_json is not None:
            d_coords_sel[tomo] = d_coords[str(tomo)][np.unique(particle_indices)]
        else:
            d_coords_sel[tomo] = d_coords[tomo][np.unique(particle_indices)]
        final_particle_count = np.sum(
            np.array([d_coords_sel[tomo].shape[0] for tomo in d_coords_sel]),
        )
    print(
        f"Cryosparc reduced particle set size from {ini_particle_count} to {final_particle_count}",
    )

    # generate Relion 4-compatible starfile
    dataio.make_starfile(d_coords_sel, config.out_file, coords_scale=1.0 / config.apix)


if __name__ == "__main__":
    main()
