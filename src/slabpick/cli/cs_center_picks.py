import os
import shutil
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def parse_args():
    """Parser for command line arguments."""
    parser = ArgumentParser(
        description="Modify a picked_particles.cs file to center picks on tile centers (fast).",
    )
    parser.add_argument(
        "--cs_file",
        type=str,
        required=True,
        help="CryoSPARC picked_particles.cs file",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        required=True,
        help="Bookkeeping file mapping particles to gallery tiles",
    )
    parser.add_argument(
        "--gallery_shape",
        type=int,
        nargs=2,
        required=False,
        default=[16, 15],
        help="Number of gallery particles in (row,col) format",
    )
    return parser.parse_args()


def generate_config(config):
    """
    Store command line arguments in a json file.
    """
    d_config = vars(config)
    reconfig = {
        "software": {"name": "slabpick", "version": "0.1.0"},
        "input": {k: d_config[k] for k in ["cs_file", "map_file"]},
        "output": {k: d_config[k] for k in ["cs_file"]},
        "parameters": {k: d_config[k] for k in ["gallery_shape"]},
    }
    out_dir = os.path.dirname(os.path.abspath(os.path.join(config.map_file, "../")))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cs_center_picks.json")
    with open(out_path, "w") as f:
        json.dump(reconfig, f, indent=4)
    return out_dir


def build_retain_idx_fast_nofeed(mgraph_id_bytes, particles_map, gallery_shape):
    """
    FAST
    - Compute nominal per-gallery particles from particles_map.
    - Verify each gallery has >= these picks. If not, fail with ValueError.
    - Select earliest `limit[g]` rows per gallery using a single sort + rank pass..
    """
    mgraph_id_strs = np.char.decode(mgraph_id_bytes, encoding="utf-8")

    df_idx = pd.DataFrame({
        "path": mgraph_id_strs,
        "idx": np.arange(len(mgraph_id_strs), dtype=np.int64),
    })
    df_idx["gallery"] = (
        df_idx["path"]
        .str.extract(r'particles_(\d+)\.mrc$', expand=False)
        .astype(int)
    )

    tile_cap = int(np.prod(gallery_shape))
    total_needed = len(particles_map)
    gnums = np.unique(particles_map.gallery.values)

    n_full_galleries = total_needed // tile_cap
    remainder = total_needed % tile_cap
    if remainder == 0:
        remainder = tile_cap

    limits = np.zeros(len(gnums), dtype=np.int64)
    if n_full_galleries > 0:
        limits[:n_full_galleries] = tile_cap
    if n_full_galleries < len(limits):
        limits[n_full_galleries] = remainder

    counts = (
        df_idx["gallery"]
        .value_counts(sort=False)
        .reindex(gnums, fill_value=0)
        .astype(np.int64)
        .to_numpy()
    )

    allocated = np.minimum(counts, limits).sum()

    if allocated != total_needed:
        raise ValueError(
            f"Could not allocate exactly {total_needed} rows; allocated {allocated}."
        )

    df_idx_sorted = df_idx.sort_values(["gallery", "idx"], kind="mergesort")
    df_idx_sorted["rank_in_gallery"] = df_idx_sorted.groupby("gallery").cumcount()

    per_gallery_limit = dict(zip(gnums, limits))
    lim_series = (
        df_idx_sorted["gallery"]
        .map(per_gallery_limit)
        .fillna(0)
        .astype(np.int64)
    )

    mask_keep = df_idx_sorted["rank_in_gallery"] < lim_series
    kept_rows = df_idx_sorted.loc[mask_keep, "idx"]

    retain_idx = np.sort(kept_rows.to_numpy(dtype=np.int64))

    if len(retain_idx) != total_needed:
        raise ValueError(
            f"retain_idx length mismatch: {len(retain_idx)} vs particles_map {total_needed}"
        )

    return retain_idx


def main():
    config = parse_args()
    out_dir = generate_config(config)

    print(f"[info] reading particles map: {config.map_file}")
    particles_map = pd.read_csv(config.map_file)

    print(f"[info] loading picks: {config.cs_file}")
    cs_picks = np.load(config.cs_file)

    gallery_shape = tuple(config.gallery_shape)[::-1]

    print("[info] computing retain indices (fast path, no feed-forward)...")
    retain_idx = build_retain_idx_fast_nofeed(
        cs_picks["location/micrograph_path"],
        particles_map,
        gallery_shape,
    )

    print("[info] slicing cs_picks down to kept indices...")
    cs_picks = cs_picks[retain_idx]
    assert len(cs_picks) == len(particles_map)

    print("[info] assigning centered fractional coordinates...")
    fxpos = np.arange(gallery_shape[0]) / gallery_shape[0] + 0.5 / gallery_shape[0]
    fypos = np.arange(gallery_shape[1]) / gallery_shape[1] + 0.5 / gallery_shape[1]
    xpos, ypos = np.meshgrid(fxpos, fypos)
    xpos, ypos = xpos.flatten(), ypos.flatten()

    gnums = np.unique(particles_map.gallery.values)
    tile_cap = np.prod(gallery_shape)

    new_xpos = np.tile(xpos, len(gnums) - 1)
    new_ypos = np.tile(ypos, len(gnums) - 1)

    remainder = len(particles_map) % tile_cap
    if remainder == 0:
        remainder = tile_cap

    new_xpos = np.concatenate((new_xpos, xpos[:remainder]))
    new_ypos = np.concatenate((new_ypos, ypos[:remainder]))

    cs_picks["location/center_x_frac"] = new_xpos
    cs_picks["location/center_y_frac"] = new_ypos

    backup_path = config.cs_file.split(".cs")[0] + "_original.cs"
    print(f"[info] backing up original picks to {backup_path}")
    shutil.copy2(config.cs_file, backup_path)

    print(f"[info] writing updated picks to {config.cs_file}")
    with open(config.cs_file, "wb") as f:
        np.save(f, cs_picks)

    print(f"[done] wrote updated picks and config to {out_dir}")


if __name__ == "__main__":
    main()
