# EPU_foilhole_analysis

Still a work in progress so sorry it's not too user friendly... --MGI

USAGE: EPU_foilhole_analysis.py <Path to EPU metadata> <relion starfile>

THe EPU metadata must have the original directory structure as written by EPU
The starfile should be a run_itxxx_data.star from a relion reconstruction.

Later I will add features to also examine the raw picked particles vs the ones that actually went into the reconstruction. 
There may be some very interesting information hidden there. The start of that is in compare_recon_vs_extract:

USAGE: compare_recon_vs_extract <reconstruction data starfile> <original particles starfile>

reconstruction star file is run_itxxx_data.star from a Refine3D job
original is from the particles.star from an Extrat job
