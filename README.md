# slabpick
Tools to facilitate particle picking and curation from 2D projections of tomography data
from either slabs (projected slices through the full tomogram) or minislabs (projections
of per-particle subvolumes). Minslabs can be formatted as a particle stack for cryoDRGN
and Relion compatibility or a gallery of tiled particle projections for use in CryoSPARC.
Functions to generate Relion-4 starfiles or copick-formatted annotations from particles
curated in CryoSPARC or by Relion's 2D class averaging routine are also included.

## Instructions for use

The `make_minislabs` command will generate either galleries (per-particle projections tiled
into mock micrographs) for use in CryoSPARC or particle stacks for use in Relion.

For use in CryoSPARC, an example workflow would be:
1. Import the galleries as micrographs with the "Output Constant CTF" option turned on.
2. Perform a round of blob picking, with the settings set to ensure that each gallery has
more picks than the number of tiles per gallery.
3. Run `cs_center_picks` to center the mock picks from blob-picking in each tile and remove
excess picks, e.g.
```
cs_center_picks —cs_file /path/to/picked_particles.cs —map_file path/to/gallery/particle_map.csv
```
4. After 2D classification, particles can be mapped back to a copick project or a Relion-4 starfile
using the `cs_map_particles` command.

Details for use in Relion to follow.