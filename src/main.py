import rich
import shapely
import numpy as np
import matplotlib.pyplot as plt
from rich import print as rprint

from particle_manager import ParticleManager
from background_grid import BackgroundGrid
from fft_generator import FFTGenerator


if __name__ == '__main__':
    rprint("==================== S T A R T ====================")
    ## User Inputs ##
    doi = (0.0, 0.0, 10.0, 10.0)
    xmin, ymin, xmax, ymax = doi
    bkgGrid = BackgroundGrid(doi, spacing=0.05*(xmax - xmin))
    manager = ParticleManager(doi)
    manager.background_grid = bkgGrid
    manager.minimum_gap = 0.005*(xmax - xmin)
    manager.grading_limits = [0.5, 1.0]
    manager.grading_content = 0.6

    ## Group particle ##
    manager.set_particle_amplitude([0.15, 0.075, 0.005, 0.001])
    manager.set_diameter_interval([0.5, 1.0])
    manager.set_dipAngle_parameter([10.0, 10.0])
    manager.generate_particle_group(max_iters=1000, max_times=100)

    ## Information ##
    count = manager.get_number_of_particles()
    ratio = manager.get_particle_volume_ratio()
    rprint(f'[ PROMPT ] {count} particles account for {ratio*100}%.')
    rprint("===================== E  N  D =====================")

    ## Visiualization ##
    fig, axs = plt.subplots(figsize=(6.5, 6.5))
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    axs.set_aspect('equal')
    axs.set_title(f'Number of particles: ({manager.get_number_of_particles()})')
    nrows, ncols = bkgGrid.shape
    axs.set_xticks(np.linspace(xmin, xmax, ncols+1))
    axs.set_yticks(np.linspace(ymin, ymax, nrows+1))
    axs.grid(True, ls='--')

    for particle in manager.particleCollection:
        # print(particle.calc_area())
        origin = particle.centroid()[0]
        axs.text(origin[0], origin[1], f'{particle.pid}')
        particle.render(axs, add_bbox=False, add_rect=False)
    plt.show()


