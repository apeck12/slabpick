import os
import time
import glob
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import zarr

import slabpick.dataio as dataio

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_subvolume(
    coord: np.ndarray,
    volume: np.ndarray | zarr.core.Array,
    shape: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    """
    Retrieve a subvolume centered on the specified coordinates
    of the given extraction shape. If the

    Parameters
    ----------
    coord: coordinates center of subvolume in pixels
    volume: volume to extract from
    shape: subvolume shape in pixels

    Returns
    -------
    extracted subvolume
    """
    c = coord.astype(int)
    xstart, xend = c[2] - int(shape[2] / 2), c[2] + int(shape[2] / 2)
    ystart, yend = c[1] - int(shape[1] / 2), c[1] + int(shape[1] / 2)
    zstart, zend = c[0] - int(shape[0] / 2), c[0] + int(shape[0] / 2)

    xstart = max(0, xstart)
    ystart = max(0, ystart)
    zstart = max(0, zstart)

    xend = min(xend, volume.shape[0])
    yend = min(yend, volume.shape[1])
    zend = min(zend, volume.shape[2])

    subvolume = volume[xstart:xend, ystart:yend, zstart:zend]

    delta = shape[::-1] - np.array(subvolume.shape)
    if np.sum(np.abs(delta)) == 0:
        return subvolume

    else:
        fill_volume = np.random.normal(
            loc=np.abs(subvolume.mean()),
            scale=10 * subvolume.std(),
            size=shape[::-1],
        )

        xs, ys, zs = 0, 0, 0
        ze, ye, xe = shape

        if delta[0] != 0:
            if xstart == 0:
                xs = delta[0]
            else:
                xe -= delta[0]

        if delta[1] != 0:
            if ystart == 0:
                ys = delta[1]
            else:
                ye -= delta[1]

        if delta[2] != 0:
            if zstart == 0:
                zs = delta[2]
            else:
                ze -= delta[2]

        fill_volume[xs:xe, ys:ye, zs:ze] = subvolume
        fill_volume = scipy.ndimage.gaussian_filter(
            fill_volume,
            sigma=2,
        )
        fill_volume[xs:xe, ys:ye, zs:ze] = subvolume

        return fill_volume


def tilt_subvolume(
    subvolume: np.ndarray,
    angle: float,
    extract_shape: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    """
    Tilt a subvolume along the tilt axis by the specified
    angle. Crop the tilted subvolume along the x-axis to
    the target shape. Vaccuum regions that enter the field
    of view are set to zero.

    Parameters
    ----------
    subvolume: subvolume to tilt
    angle: tilt angle in degrees
    extract_shape: target subvolume shape

    Returns
    -------
    tilted_subvolume of correct xy dimensions
    """
    subvolume_rot = scipy.ndimage.rotate(
        subvolume,
        angle,
        axes=(0, 2),
        reshape=True,
        order=1,
        mode="constant",
        cval=0,
    )

    mdpt = int(subvolume_rot.shape[2] / 2)
    hdim = int(extract_shape[0] / 2)

    return subvolume_rot[:, :, mdpt - hdim : mdpt + hdim]


def render_even(shape: np.ndarray) -> np.ndarray:
    """
    Force all even dimensions for a volume's shape
    array by reducing any odd dimensions by 1 pixel.

    Parameters
    ----------
    shape: volume shape in pixels

    Returns
    -------
    shape: shape of all even dimensions
    """
    odd_dims = np.where(shape.astype(int) % 2)[0]
    if len(odd_dims) > 0:
        shape[odd_dims] -= 1
    return shape


def generate_minislabs(
    coords: np.ndarray,
    volume: np.ndarray,
    extract_shape: np.ndarray | tuple,
    angles: list = None,
    buffered_shape: np.ndarray | tuple = None,
) -> dict:
    """
    Generate minislabs by extracting subvolumes around the specified
    coordinates, optionally tilting, and projecting along the z-axis.
    If applying tilts, the extraction subvolume is increased along the
    x-axis by default to avoid projecting vacuum along the edges.

    Parameters
    ----------
    coords: (X,Y,Z) coordinates of particle centers in pixels
    volume: particle-containing tomogram
    extract_shape: target (X,Y,Z) dimensions of extraction subvolumes
    angles: tilt angles to apply to each subvolume
    buffered_shape: actual (X,Y,Z) dimensions of subvolumes to extract

    Returns
    -------
    projs: dict of per-particle minislabs
    """
    if angles is None:
        angles = [0]
    counter = 0
    projs = {}

    if angles == [0]:
        buffered_shape = extract_shape
    else:
        if buffered_shape is None:
            buffered_shape = np.array([1.5, 1, 1]) * np.array(extract_shape)
            buffered_shape = render_even(buffered_shape.astype(int))

    for i, c in enumerate(coords):
        print(f"processing particle {i}")
        subvolume = get_subvolume(c, volume, buffered_shape)
        if subvolume.shape != tuple(buffered_shape)[::-1]:
            raise ValueError("Shapes do not match")

        if angles == [0]:
            projs[counter] = np.sum(subvolume, axis=0)
            counter += 1

        else:
            for angle in angles:
                tilted_subvolume = tilt_subvolume(subvolume, angle, extract_shape)
                projs[counter] = np.sum(tilted_subvolume, axis=0)
                counter += 1

    return projs


class Minislab:
    """
    Generate minislabs, or per-particle subvolume projections. For
    downstream processing either pile them up into particle stacks
    or tile them into 2d gallery views.
    """

    def __init__(
        self,
        extract_shape: tuple[int, int, int],
        angles: list = None,
    ):
        """
        Parameters
        ----------
        extract_shape: subvolume shape in pixels along (X,Y,Z)
        angles: tilt angles if generating tilted minislabs
        """
        if angles is None:
            angles = [0]
        self.minislabs = {}
        self.angles = angles
        self.shape = extract_shape
        buffered_shape = np.array([1.5, 1, 1]) * np.array(extract_shape)
        self.buffered_shape = render_even(buffered_shape.astype(int))

        # for bookkeeping purposes
        self.num_particles = 0
        self.num_galleries = 0
        self.particle_index = []
        self.particle_tilt = []
        self.tomogram_id = []
        self.row_idx, self.col_idx, self.gallery_idx = [], [], []

    def make_minislabs(
        self,
        coords: np.ndarray,
        volume: np.ndarray,
        tomo_id: str,
    ) -> None:
        """
        Generate minislabs from the specified volume and coordinates.

        Parameters
        ----------
        coords: (X,Y,Z) coordinates of particle centers in pixels
        volume: particle-containing tomogram
        tomo_id: tomogram name
        """
        projs = generate_minislabs(
            coords,
            volume,
            self.shape,
            self.angles,
            self.buffered_shape,
        )

        for i in range(len(projs)):
            self.minislabs[self.num_particles] = projs[i]
            self.num_particles += 1
        self.particle_tilt.extend(coords.shape[0] * list(self.angles))
        self.particle_index.extend(
            list(np.repeat(np.arange(coords.shape[0]), len(self.angles))),
        )
        self.tomogram_id.extend(len(projs) * [tomo_id])

    def make_one_gallery(
        self,
        gshape: tuple[int, int],
        key_list: list,
    ) -> np.ndarray:
        """
        Generate one gallery by tiling the minislabs specified in key_list.

        Parameters
        ----------
        gshape: gallery shape in (nrows, ncols)
        key_list: keys to pull from self.minislabs dict

        Returns
        -------
        gallery: gallery of tiled minislabs
        """
        if len(key_list) > np.prod(gshape):
            raise IndexError("Number of minislabs exceeds number of gallery tiles.")

        pshape = self.minislabs[key_list[0]].shape  # particle shape
        gallery = np.zeros((gshape[0] * pshape[0], gshape[1] * pshape[1])).astype(
            np.float32,
        )
        stacked = np.array([self.minislabs[i] for i in self.minislabs])

        counter = 0
        for i in range(gshape[0]):
            for j in range(gshape[1]):
                if counter > len(key_list) - 1:
                    fill_volume = np.random.normal(
                        loc=np.abs(stacked.mean()),
                        scale=stacked.std(),
                        size=pshape,
                    )
                    fill_volume = scipy.ndimage.gaussian_filter(
                        fill_volume,
                        sigma=1.4,
                    )
                    gallery[
                        i * pshape[0] : i * pshape[0] + pshape[0],
                        j * pshape[1] : j * pshape[1] + pshape[1],
                    ] = fill_volume
                else:
                    gallery[
                        i * pshape[0] : i * pshape[0] + pshape[0],
                        j * pshape[1] : j * pshape[1] + pshape[1],
                    ] = self.minislabs[key_list[counter]]
                    self.row_idx.append(i)
                    self.col_idx.append(j)
                    self.gallery_idx.append(self.num_galleries)
                counter += 1

        self.num_galleries += 1

        return gallery

    def make_galleries(
        self,
        gshape: tuple[int, int],
        out_dir: str,
        apix: float,
    ) -> None:
        """
        Generate galleries from all stored minislabs.

        Parameters
        ----------
        gshape: gallery shape in (nrows, ncols)
        out_dir: output directory
        apix: pixel size in Angstrom
        """
        # generate galleries
        key_list_all = list(self.minislabs.keys())
        n_mgraphs = len(key_list_all) // np.prod(gshape) + 1
        if len(key_list_all) % np.prod(gshape) == 0:
            n_mgraphs -= 1

        os.makedirs(out_dir, exist_ok=True)
        for nm in range(self.num_galleries, n_mgraphs):
            print(f"Generating gallery {nm}")
            end_key = np.min([np.prod(gshape) * (nm + 1), max(key_list_all) + 1])
            key_list = list(np.arange(nm * np.prod(gshape), end_key).astype(int))
            gallery = self.make_one_gallery(gshape, key_list)
            dataio.save_mrc(
                gallery, os.path.join(out_dir, f"particles_{nm:03d}.mrc"), apix=apix,
            )

        self.make_gallery_bookkeeper(out_dir)

    def make_galleries_live(
        self,
        gshape: tuple[int, int],
        out_dir: str,
        apix: float,
        final: bool=False,
    ) -> None:
        """
        Generate as many full galleries as possible from stored
        minislabs, unless in final mode, in which case generate
        a gallery from whatever is left over.

        Parameters
        ----------
        gshape: gallery shape in (nrows, ncols)
        out_dir: output directory
        apix: pixel size in Angstrom
        """
        os.makedirs(out_dir, exist_ok=True)
        
        if final:
            self.make_galleries(gshape, out_dir, apix)
            
        else:
            key_list_all = list(self.minislabs.keys())
            n_mgraphs = len(key_list_all) // np.prod(gshape) 

            for nm in range(self.num_galleries, n_mgraphs):
                print(f"Generating gallery {nm}")
                end_key = np.prod(gshape) * (nm + 1)
                key_list = list(np.arange(nm * np.prod(gshape), end_key).astype(int))
                gallery = self.make_one_gallery(gshape, key_list)
                dataio.save_mrc(
                    gallery, os.path.join(out_dir, f"particles_{nm:03d}.mrc"), apix=apix,
                )
        
    def make_gallery_bookkeeper(
        self,
        out_dir: str,
    ) -> pd.DataFrame:
        """
        Generate a gallery bookkeeping file in csv format that
        tracks the provenance of each minislab (gallery tile).

        Parameters
        ----------
        gshape: gallery shape in (nrows, ncols)
        out_dir: output directory
        apix: pixel size in Angstrom

        Returns
        -------
        df: mapping between gallery tiles and particle origins
        """
        df = pd.DataFrame(
            {
                "tomogram": self.tomogram_id,
                "particle": self.particle_index,
                "tilt": self.particle_tilt,
                "gallery": self.gallery_idx,
                "row": self.row_idx,
                "col": self.col_idx,
            },
        )
        df.to_csv(os.path.join(out_dir, "particle_map.csv"), index=False)
        return df

    def make_stack(
        self,
        out_dir: str,
        apix: float,
    ) -> np.ndarray:
        """
        Generate a particle stack from all stored minislabs.

        Parameters
        ----------
        out_dir: output directory
        apix: pixel size in Angstrom

        Returns
        -------
        stack: particle stack
        """
        # generate particle stack
        os.makedirs(out_dir, exist_ok=True)
        stack = np.array([self.minislabs[i] for i in self.minislabs])
        dataio.save_mrc(stack, os.path.join(out_dir, "particles.mrcs"), apix=apix)

        # generate bookkeeping file
        df = pd.DataFrame(
            {
                "tomogram": self.tomogram_id,
                "particle": self.particle_index,
                "tilt": self.particle_tilt,
            },
        )
        df.to_csv(os.path.join(out_dir, "particle_map.csv"), index=False)

        return stack


def make_minislabs_multi_entry(
    in_coords: str,
    in_vol: str,
    out_dir: str,
    extract_shape: tuple[int,int,int],
    voxel_spacing: float,
    extension: str="mrc",
    tomo_type: str=None,
    particle_name: str=None,
    user_id: str=None,
    session_id: str=None,
    coords_scale: float=1,
    col_name: str='rlnMicrographName',
    angles: list=[0],
    gshape: tuple[int,int]=(16,15),
) -> None:
    """
    Copick entry point for generating minislabs. Both coordinates and
    volumes are specified in a copick configuration file. Minislabs are 
    output both as galleries and a particle stack.
    
    Parameters
    ----------
    in_coords: copick configuration file or starfile 
    in_vol: directory containing tomograms or None if tomos from copick
    out_dir: output directory
    extract_shape: extraction shape in Angstrom along (X,Y,Z)
    voxel_spacing: tomogram voxel spacing
    extension: mrc or zarr if not loading from copick
    tomo_type: tomogram type 
    particle_name: particle name
    user_id: copick user ID 
    session_id: copick session ID 
    coords_scale: factor to convert starfile coords to Angstrom
    col_name: column name cooresponding to tomogram name
    angles: angles if generating tilted minislabs
    gshape: gallery shape (nrows, ncols)
    """
    # load coordinates -- starfile entry
    if os.path.splitext(in_coords)[-1] == ".star":
        cp_interface = None
        d_coords = dataio.read_starfile(
            in_coords,
            coords_scale=coords_scale,
            col_name=col_name,
        )
    # load coordinates -- copick entry
    elif os.path.splitext(in_coords)[-1] == ".json":
        cp_interface = dataio.CopickInterface(in_coords)
        d_coords = cp_interface.get_all_coords(
            particle_name,
            user_id=user_id,
            session_id=session_id,
        )
    else:
        raise ValueError("in_coords argument not recognized")

    if cp_interface is None and os.path.splitext(in_vol)[-1] == ".json":
        cp_interface = dataio.CopickInterface(in_vol)
    
    extract_shape = (np.array(extract_shape)/voxel_spacing).astype(int)
    extract_shape = render_even(extract_shape)
    
    montage = Minislab(extract_shape, angles)
    for run_name in d_coords:
        print(f"Processing volume {run_name}")
        # load volume -- directory entry 
        if in_vol is not None and os.path.isdir(in_vol):
            vol_name = os.path.join(in_vol, f"{run_name}.{extension}")
            if extension == "mrc":
                volume = dataio.load_mrc(vol_name)
            elif extension == "zarr":
                volume = np.array(zarr.open(vol_name, "r"))
        # load volume -- copick entry
        else:
            volume = cp_interface.get_run_tomogram(str(run_name), voxel_spacing, tomo_type)
        coords_pixels = d_coords[run_name] / voxel_spacing
        montage.make_minislabs(coords_pixels, volume, run_name)
    montage.make_galleries(gshape, os.path.join(out_dir, "gallery"), voxel_spacing)
    montage.make_stack(os.path.join(out_dir, "stack"), voxel_spacing)

    
def make_minislabs_live(
    in_star: str,
    in_vol: str,
    out_dir: str,
    extract_shape: tuple[int,int,int],
    voxel_spacing: float,
    coords_scale: float,
    col_name: str='rlnMicrographName',
    angles: list=[0],
    gshape: tuple[int,int]=(16,15),
    t_interval: float=300,
    t_exit: float=1800,
) -> None:
    """
    Live mode for generating minislabs. Volumes should be in a single
    directory, while coordinates are derived from starfiles that are 
    being generated live. Minislabs are output both as galleries and 
    a particle stack.
    
    Parameters
    ----------
    in_star: glob-expandable path to star files
    in_vol: directory containing tomograms in mrc format
    out_dir: output directory
    extract_shape: extraction shape in Angstrom along (X,Y,Z)
    voxel_spacing: tomogram voxel spacing
    coords_scale: factor to convert starfile coords to Angstrom
    col_name: column name cooresponding to tomogram name
    angles: angles if generating tilted minislabs
    gshape: gallery shape (nrows, ncols)
    t_interval: interval in seconds before checking for new files
    t_exit: interval in seconds after which to exit if no new files found
    """
    start = time.time()
    processed = []
    
    extract_shape = (np.array(extract_shape)/voxel_spacing).astype(int)
    extract_shape = render_even(extract_shape)
    
    montage = Minislab(extract_shape, angles)
    while True:
        fnames = glob.glob(in_star)
        fnames = [fn for fn in fnames if fn not in processed]
        print(time.strftime("%X %x %Z"), f": Found {len(fnames)} new files to process")
        
        if len(fnames) > 0:
            d_coords = dataio.combine_star_files(
                fnames,
                coords_scale=coords_scale,
                col_name=col_name,
            )
            
            for run_name in d_coords:
                print(f"Processing volume {run_name}")
                vol_name = os.path.join(in_vol, f"{run_name}.mrc")
                volume = dataio.load_mrc(vol_name)
                coords_pixels = d_coords[run_name] / voxel_spacing
                montage.make_minislabs(coords_pixels, volume, run_name)
                
            montage.make_galleries_live(gshape, os.path.join(out_dir, "gallery"), voxel_spacing)
            
            processed.extend(fnames)
            start_time = time.time()
            
        time.sleep(t_interval)
        t_elapsed = time.time() - start_time
        if t_elapsed > t_exit:
            break
            
    montage.make_galleries_live(gshape, os.path.join(out_dir, "gallery"), voxel_spacing, final=True)
    montage.make_stack(os.path.join(out_dir, "stack"), voxel_spacing)
