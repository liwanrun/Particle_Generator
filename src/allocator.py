import shapely
import numpy as np
from particle import Particle
from bkggrid import BkgGrid

class Allocator:
    def __init__(self, doi:tuple, spacing, min_gap) -> None:
        self.doi = doi
        self.spacing = spacing
        self.min_gap = min_gap
        self.particles = []

        nrows = int(np.ceil(doi[0] / spacing))
        ncols = int(np.ceil(doi[1] / spacing))
        self.msh = BkgGrid((nrows, ncols), spacing/np.sqrt(2))

    def allocate_particle_by_particle(self):
        pass