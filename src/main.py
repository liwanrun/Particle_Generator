import rich
import shapely
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rich import print as rprint
from particle_manager import ParticleManager
from background_grid import BackgroundGrid
from fft_generator import FFTGenerator

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.size'] = 10.5

if __name__ == '__main__':
    rprint("==================== S T A R T ====================")
    ## User Inputs ##
    doi = (0.0, 0.0, 10.0, 10.0)
    xmin, ymin, xmax, ymax = doi
    bkgGrid = BackgroundGrid(doi, spacing=0.05*(xmax - xmin))
    # bkgGrid.create_seed_points()
    manager = ParticleManager(doi)
    manager.background_grid = bkgGrid
    manager.minimum_gap = 0.01*(xmax - xmin)
    manager.grading_limits = [0.4, 1.0]
    manager.grading_content = 1.0
    manager.boundary_periodic = True
    # np.random.seed(1)

    ## Group 1 ##
    manager.set_particle_amplitude([0.3, 0.00, 0.001, 0.0000])
    manager.set_diameter_interval([0.6, 1.0])
    manager.set_dipAngle_parameter([45.0, 0.0])
    manager.generate_particle_group(gid = 0, max_iters=1000, max_times=100)
    ## Group 2 ##
    # manager.set_particle_amplitude([0.3, 0.05, 0.001, 0.000])
    # manager.set_diameter_interval([0.4, 0.6])
    # manager.set_dipAngle_parameter([30.0, 10.0])
    # manager.generate_particle_group(gid = 1, max_iters=1000, max_times=500)
    ## Group 3 ##
    # manager.set_particle_amplitude([0.45, 0.075, 0.005, 0.001])
    # manager.set_diameter_interval([0.7, 0.8])
    # manager.set_dipAngle_parameter([10.0, 10.0])
    # manager.generate_particle_group(gid = 2, max_iters=1000, max_times=100)

    ## Information ##
    count = manager.get_number_of_particles()
    ratio = manager.particleContent / ((xmax - xmin) * (ymax - ymin))
    rprint(f'[ PROMPT ] {count} particles account for {ratio*100}%.')
    rprint("===================== E  N  D =====================")
    manager.write_gmsh_model('particles.geo')

    ## Visiualization ##
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, BoundaryNorm
    nbin = manager.get_number_of_particles()
    #nbin = 3
    cmap = mpl.colormaps['coolwarm'].resampled(nbin)
    # norm = Normalize(vmin=0, vmax=manager.get_number_of_particles())
    norm = BoundaryNorm(np.arange(cmap.N+1).tolist(), ncolors=cmap.N)
    mapper = ScalarMappable(norm, cmap)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0), layout='constrained')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect(1.0)
    ax.set_box_aspect(1.0)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Number of particles: {manager.get_number_of_particles()}')
    nrows, ncols = bkgGrid.shape
    ax.set_xticks(np.linspace(xmin, xmax, 11))
    ax.set_yticks(np.linspace(ymin, ymax, 11))
    #ax.set_clip_on(False)
    fig.colorbar(mapper, ax=ax, extend='neither', label='Particle Id')

    for particle in manager.particleCollection:
        # print(particle.calc_area())
        origin = particle.centroid()[0]
        ax.text(origin[0], origin[1], f'{particle.pid}',color='m')
        color = mapper.to_rgba(particle.pid)
        particle.render(ax, color, add_bbox=False, add_rect=False)

    ax.grid(True, lw=0.5, ls='--', zorder=0)
    plt.savefig('img\irregular_particles.svg', dpi=330, transparent=True)
    plt.show()


