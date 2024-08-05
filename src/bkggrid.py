from particle import *

class BkgGrid:
    def __init__(self, shape=(10, 10), space=1.0) -> None:
        self.shape = shape
        self.space = space
        self.particleIds = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    def __repr__(self) -> str:
        print(f'Backgroud grid with shape = {self.shape} and spcae = {self.space}')

    def get_incident_cells(self, particle:Particle):
        minx, miny, maxx, maxy = particle.boundingBox()
        rid = max(int(np.floor(miny / self.space)), 0)
        rjd = min(int(np.ceil(maxy / self.space)), self.shape[0])
        cid = max(int(np.floor(minx / self.space)), 0)
        cjd = min(int(np.ceil(maxx / self.space)), self.shape[1])
        cellIds = np.dstack(np.mgrid[rid:rjd, cid:cjd])
        cellIds = cellIds.reshape(cellIds.size//2, 2).tolist()
        return cellIds
    
    def assign_particle_to_cells(self, particle:Particle):
        cellIds = self.get_incident_cells(particle)
        for cid in cellIds:
            self.particleIds[cid[0]][cid[1]].append(particle.pid)


## Debug
if __name__ == '__main__':
    pass