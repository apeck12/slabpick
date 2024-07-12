import numpy as np
import sys
import os
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_dir)
from dataio import *

def test_mrc_functions():
    """
    Test mrc functions: loading, retrieving voxel size and saving.
    """
    apix = 5
    out_name = "test.mrc"

    vol = np.random.randn(5,3).astype(np.float32)
    save_mrc(vol, out_name, apix=apix)
    assert get_voxel_size(out_name) == apix
    
    same_vol = load_mrc(out_name)
    np.testing.assert_array_equal(vol, same_vol)
    os.remove(out_name)
    
def test_star_functions():
    """
    Test writing and loading starfiles from/to a dict mapping 
    run names to coordinates.
    """
    coords_scale = 2
    out_name = "test.star"
    
    d_coords = {}
    for run_name in ['TS_1_1', 'TS_1_2', 'TS_1_3']:
        d_coords[run_name] = np.random.randint(0, high=100, 
                                               size=(np.random.randint(3,7),3)).astype(np.float32)
    make_starfile(d_coords, out_name, coords_scale=coords_scale)
    
    same_d_coords = read_starfile(out_name, coords_scale=1.0/coords_scale)
    for run_name in d_coords.keys():
        np.testing.assert_array_equal(d_coords[run_name], 
                                      same_d_coords[run_name])
    os.remove(out_name)
