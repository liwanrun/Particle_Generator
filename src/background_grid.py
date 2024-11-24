from particle import *

class BackgroundGrid:
    def __init__(self, doi, spacing=1.0) -> None:
        xmin, ymin, xmax, ymax = doi
        m = int(np.ceil((ymax - ymin) / spacing))
        n = int(np.ceil((xmax - xmin) / spacing))
        self.shape = (m, n)
        self.spacing = spacing
        self.allocParticleIds = [[[] for _ in range(m)] for _ in range(n)]
        self.doi = doi

    def __repr__(self) -> str:
        print(f'Backgroud grid shape = {self.shape}.')

    def get_incident_cells(self, particle:Particle):
        minx, miny, maxx, maxy = particle.boundingBox()
        rmin = max(int(np.floor(miny / self.spacing)), 0)
        rmax = min(int(np.ceil(maxy / self.spacing)), self.shape[0])
        cmin = max(int(np.floor(minx / self.spacing)), 0)
        cmax = min(int(np.ceil(maxx / self.spacing)), self.shape[1])
        cellIds = np.dstack(np.mgrid[rmin:rmax, cmin:cmax])
        cellIds = cellIds.reshape(cellIds.size//2, 2).tolist()
        return cellIds    
    
    def assign_particle_to_cells(self, particle:Particle):
        covered_cells = self.get_incident_cells(particle)
        for cell in covered_cells:
            i, j = cell
            self.allocParticleIds[i][j].append(particle.pid)


## Debug
if __name__ == '__main__':
    doi = (0.0, 0.0, 10.0, 10.0)
    grid = BackgroundGrid(doi = doi)
    grid.create_seed_points(spacing = 0.5)
    # print(grid.seedPoints)
    seed = grid.sample_random_seed()
    print(seed)
  
    # Visualization
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax.plot(grid.seedPoints[:,:,0], grid.seedPoints[:, :, 1], 'bo')
    ax.plot(seed[0], seed[1], 'ro')
    ax.set_aspect(1.0)
    plt.show()