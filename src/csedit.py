import pandas as pd
import numpy as np

def curate_particles_map(cs_extract: np.ndarray, 
                         particles_map: pd.DataFrame, 
                         max_distance: float=0.2) -> pd.DataFrame:
    """
    Curate a bookkeeping file that maps entries to gallery tiles
    based on particles retained in a cryosparc extraction job.
    
    Parameters
    ----------
    cs_extract: np.recarray, cryosparc topaz_picked_particles.cs 
    particles_map: pd.DataFrame, gallery bookkeeping file
    max_distance: float, fractional distance allowed for miscentering
    
    Returns
    -------
    pd.DataFrame, reduced gallery bookkeeping file of retained particles
    """
    # extract x,y positions of cryosparc-extracted particles
    g_shape = (particles_map.row.max()+1, particles_map.col.max()+1)[::-1]
    cs_xpos = np.array(cs_extract['location/center_x_frac']*g_shape[0] - 0.5)
    cs_ypos = np.array(cs_extract['location/center_y_frac']*g_shape[1] - 0.5)
    
    # exclude particles too far away from tile centers
    remainder_x, remainder_y = cs_xpos%1, cs_ypos%1
    remainder_x = np.min([remainder_x, 1 - remainder_x], axis=0)
    remainder_y = np.min([remainder_y, 1 - remainder_y], axis=0)
    residual = np.sqrt(np.sum(np.square([remainder_x, remainder_y]), axis=0))
    n_excluded = len(residual)-len(residual>max_distance)
    print(f"{n_excluded} particles of {len(residual)} total excluded based on distance threshold")
    cs_xpos = np.around(cs_xpos[np.where(residual<max_distance)[0]]).astype(int)
    cs_ypos = np.around(cs_ypos[np.where(residual<max_distance)[0]]).astype(int)
    
    # extract gallery index of cryosparc-extracted particles
    cs_mgraph_id = cs_extract['location/micrograph_path']
    cs_mgraph_id = np.array([int(fn.decode("utf-8").split("_")[-1].split(".mrc")[0]) for fn in cs_mgraph_id]) 
    assert np.all(cs_mgraph_id[:-1] <= cs_mgraph_id[1:])
    cs_mgraph_id = cs_mgraph_id[np.where(residual<max_distance)[0]]
    cs_map = np.array([cs_mgraph_id, cs_xpos, cs_ypos]).T
    
    # map back to gallery bookkeeping file
    ini_map = np.array([particles_map.gallery.values, particles_map.col.values, particles_map.row.values]).T
    indices = np.where(np.prod(np.swapaxes(ini_map[:,:,None],1,2) == cs_map, axis=2).astype(bool))
    assert np.sum(np.abs(ini_map[indices[0]] - cs_map[indices[1]]))==0
    
    return particles_map.iloc[indices[0]]
