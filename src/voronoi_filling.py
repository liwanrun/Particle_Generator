import numpy as np
import shapely as sp
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class Cropped_voronoi():
    '''Voronoi diagram cropped by a rectangular boundary'''
    def __init__(self, doi:np.array) -> None:
        self.doi = doi

    def create_voronoi_diagram(self, npts):
        xmin, ymin, xmax, ymax = self.doi
        xx = np.random.uniform(xmin, xmax, npts)
        yy = np.random.uniform(ymin, ymax, npts)
        points = np.vstack((xx, yy)).T
        print(xx); print(yy); print(points)
        self.voi = Voronoi(points)
    
    def boundary_seed_points(self) -> np.array:
        boundary_seeds = []
        for p_id, r_id in enumerate(self.voi.point_region):
            region = self.voi.regions[r_id]
            if np.any(np.array(region) == -1):
                boundary_seeds.append(self.voi.points[p_id])
            else:
                xmin, ymin, xmax, ymax = self.doi
                doi_poly = sp.Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                rgn_poly = sp.Polygon(self.voi.vertices[region])
                if not doi_poly.contains_properly(rgn_poly):
                    boundary_seeds.append(self.voi.points[p_id])

        return np.array(boundary_seeds)


if __name__ == '__main__':
    np.random.seed(0)
    bounds = np.array([0.0, 0.0, 10.0, 10.0])
    crop_voi = Cropped_voronoi(bounds)
    crop_voi.create_voronoi_diagram(npts=100)
    boundary_seeds = crop_voi.boundary_seed_points()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), layout='constrained')
    xx = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    yy = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ax.plot(xx, yy, 'r--')
    voronoi_plot_2d(crop_voi.voi, ax, show_vertices=False)
    ax.set_aspect(1.0)
    ax.set_xlim(-2.0, 12.0)
    ax.set_ylim(-2.0, 12.0)

    for i, point in enumerate(boundary_seeds):
        ax.text(point[0], point[1], s=f'{i}', color='r')

    plt.show()
