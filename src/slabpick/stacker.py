import pandas as pd
import numpy as np
import random
import starfile
import mrcfile


def normalize_stack(particles: np.ndarray, radius: float = 0.9) -> np.ndarray:
    """
    Normalize a stack on a per-particle basis. The central
    region inside the supplied fractional radius is masked
    to compute the per-particle mean and standard deviation.

    Parameters
    ----------
    particles: stack of particles
    radius: boundary for computing standard deviation

    Returns
    -------
    particles: normalized particle stack
    """
    pshape = particles[0].shape
    y, x = np.indices(pshape)
    center = pshape[1] / 2 - 0.5, pshape[0] / 2 - 0.5
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = np.broadcast_to(r < np.min(center) * radius, particles.shape).astype(int)
    particles_masked = np.ma.masked_array(particles, mask=mask)

    mu_background = np.mean(particles_masked, axis=(1,2)).data
    sigma_background = np.std(particles_masked, axis=(1,2)).data
    particles -= mu_background[:,np.newaxis,np.newaxis]
    particles /= sigma_background[:,np.newaxis,np.newaxis]
    return particles


def invert_contrast(particles: np.ndarray) -> np.ndarray:
    """
    Invert contrast of a particle stack.

    Parameters
    ----------
    particles: stack of particles

    Returns
    -------
    particle stack with inverted contrast
    """
    return -1.0 * particles


def combine_particle_stacks(
    stacks: list, 
    particles_starfile: str,
    output: str,
    randomize: bool=True,
    counts: list=None
)->None:
    """
    Generate a starfile corresponding to a combined particle stack.
    Optionally randomize the order (default) with an equal mix from
    the input stacks.
    
    Parameters
    ----------
    stacks: list of particle stack mrcs files
    particles_starfile: particles starfile that optics group will be taken from
    output: output particles starfile
    randomize: if True, randomize order of particle images
    counts: number of particles to take from each 
    """
    shapes = [mrcfile.mmap(filename).data.shape for filename in stacks]
    # check that particle image dimensions match across stacks
    assert np.ptp(np.array([shapes[i][1] for i in range(len(shapes))])) == 0
    assert np.ptp(np.array([shapes[i][2] for i in range(len(shapes))])) == 0
    # check that counts argument is valid
    if counts is not None:
        assert len(counts) == len(stacks)
        assert all([counts[i]<=shapes[i][0] for i in range(len(shapes))])
    
    rlnImageName = []
    for i,fname in enumerate(stacks):
        images = [f"{num}@{fname}" for num in range(shapes[i][0])]
        if counts is not None:
            images = [images[i] for i in sorted(random.sample(range(len(images)), counts[i]))]
        rlnImageName.extend(images)
    if randomize:
        random.shuffle(rlnImageName)
    
    grp_particles = {
        'rlnImageName': rlnImageName,
        'rlnOpticsGroup': len(rlnImageName) * [1],
        'rlnGroupNumber': len(rlnImageName) * [1],
    }
    
    combined = starfile.read(particles_starfile)
    combined['particles'] = pd.DataFrame.from_dict(grp_particles)
    starfile.write(combined, output)
