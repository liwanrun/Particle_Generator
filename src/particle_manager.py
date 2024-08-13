import os
import numpy as np
from particle import *
from background_grid import BackgroundGrid
from fft_generator import FFTGenerator
from rich import print as rprint
from rich.progress import track

class ParticleManager:
    def __init__(self, doi) -> None:
        self.doi = doi
        self.particleCollection = []

        self._grading_limits = None
        self._grading_content = 0.0
        self._minimum_gap = 0.0
        self._background_grid = None
        self._boundary_periodic = False
        self.group_particle_sizes = [0]

    @property
    def grading_limits(self):
        return self._grading_limits
    
    @grading_limits.setter
    def grading_limits(self, value):
        self._grading_limits = value

    @property
    def grading_content(self):
        return self._grading_content
    
    @grading_content.setter
    def grading_content(self, value):
         if not isinstance(value, float) or value < 0:
            raise ValueError('Positive value expected!')
         self._grading_content = value

    @property
    def minimum_gap(self):
        return self._minimum_gap
    
    @minimum_gap.setter
    def minimum_gap(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError('Positive value expected!')
        self._minimum_gap = value
    
    @property
    def background_grid(self):
        return self._bkgGrid
    
    @background_grid.setter
    def background_grid(self, value):
        if not isinstance(value, BackgroundGrid):
            raise ValueError('Type is not correct!')
        self._bkgGrid = value

    @property
    def boundary_periodic(self):
        return self._boundary_periodic
    
    @boundary_periodic.setter
    def boundary_periodic(self, value):
        if not isinstance(value, bool):
            raise ValueError('Bool value expected!')
        self._boundary_periodic = value

    def set_particle_amplitude(self, spectrum):
        self.spectrum = spectrum

    def set_diameter_interval(self, interval):
        self.diameter = interval

    def set_dipAngle_parameter(self, interval):
        self.dipAngle = interval

    def get_number_of_particles(self) -> int:
        return len(self.particleCollection)

    def get_particle_total_volume(self) -> float:
        vol = 0.0
        for p in self.particleCollection:
            vol += p.calc_area()
        return vol
    
    def get_particle_volume_ratio(self) -> float:
        xmin, ymin, xmax, ymax = self.doi
        total_vol = (xmax - xmin)*(ymax - ymin)
        cover_vol = self.get_particle_total_volume()
        return cover_vol / total_vol

    def get_particle_group_content(self, di, dj, factor=0.5):
        dmin, dmax = self._grading_limits
        Pmax = 100 * (dmax / dmax)**(factor)
        Pmin = 100 * (dmin / dmax)**(factor)
        Pi = 100 * (di / dmax)**(factor)
        Pj = 100 * (dj / dmax)**(factor)
        xmin, ymin, xmax, ymax = self.doi
        vol = (xmax - xmin) * (ymax - ymin) * self._grading_content
        return (Pj - Pi) / (Pmax - Pmin) * vol 
    
    def has_collision(self, particle, gap=0.0) -> bool:
        covered_cells = self._bkgGrid.get_incident_cells(particle) 
        handled_flags = np.zeros(len(self.particleCollection), dtype='bool')
        for cell in covered_cells:
            i, j = cell
            for id in self._bkgGrid.allocParticleIds[i][j]:
                if handled_flags[id]: continue
                other = self.particleCollection[id]
                if particle.is_AABB_intersect(other, gap):
                    if particle.is_exact_intersect(other, gap):
                        return True
                handled_flags[id] = True
        return False

    def assign_domain_vertex_particles(self, particle, doi) -> float:
        polygon = shapely.Polygon(particle.points)
        coords = shapely.box(*doi).exterior.coords[:-1]
        which_vertex = 0; 
        for i, xyz in enumerate(coords):
            if polygon.contains_properly(shapely.Point(xyz)): 
                which_vertex = i; break
            
        offsets = np.array([(doi[2] - doi[0]), (doi[3] - doi[1])])
        directions = [[[0, 0], [0, 1], [-1, 1], [-1, 0]], 
                      [[0, -1], [0, 0], [-1, 0], [-1, -1]], 
                      [[1, -1], [1, 0], [0, 0], [0, -1]], 
                      [[1, 0], [1, 1], [0, 1], [0, 0]]]
        ghost_particles = []
        for j, xyz in enumerate(coords):
            if polygon.contains_properly(shapely.Point(xyz)): continue
            ghost = PolygonParticle(particle.points.copy())
            ghost.translate(directions[which_vertex][j] * offsets)
            if self.has_collision(ghost, self._minimum_gap): return 0.0
            ghost_particles.append(ghost)
            
        self._bkgGrid.assign_particle_to_cells(particle)
        self.particleCollection.append(particle)
        for ghost in ghost_particles:
            ghost.pid = len(self.particleCollection)
            ghost.gid = particle.gid
            self._bkgGrid.assign_particle_to_cells(ghost)
            self.particleCollection.append(ghost) 
        return particle.calc_area()

    def assign_domain_edge_particles(self, particle, doi) -> float:
        polygon = shapely.Polygon(particle.points)
        coords = shapely.box(*doi).exterior.coords
        lines = [shapely.LineString([coords[i], coords[i+1]]) for i in range(len(coords) - 1)]
        which_edges = []
        for i, line in enumerate(lines):
            if polygon.crosses(line):
                which_edges.append(i)
                
        offsets = np.array([(doi[2] - doi[0]), (doi[3] - doi[1])])
        directions = [[[0, 0], [0, 0], [-1, 0], [0, 0]], 
                      [[0, 0], [0, 0], [0, 0], [0, -1]], 
                      [[1, 0], [0, 0], [0, 0], [0, 0]], 
                      [[0, 0], [0, 1], [0, 0], [0, 0]]]
        ghost_particles = []
        for which_edge in which_edges:
            j = (which_edge + 2) % len(lines)
            ghost = PolygonParticle(particle.points.copy())
            ghost.translate(directions[which_edge][j] * offsets)
            if self.has_collision(ghost, self._minimum_gap): return 0.0
            ghost_particles.append(ghost)

        self._bkgGrid.assign_particle_to_cells(particle)
        self.particleCollection.append(particle)
        for ghost in ghost_particles:
            ghost.pid = len(self.particleCollection)
            ghost.gid = particle.gid
            self._bkgGrid.assign_particle_to_cells(ghost)
            self.particleCollection.append(ghost)     
        return particle.calc_area()

    def generate_particle_group(self, gid, max_iters=1000, max_times=100):
        dmin, dmax = self.diameter
        mean, var = self.dipAngle
        A1, A3, A16, A37 = self.spectrum
        target_vol = self.get_particle_group_content(dmin, dmax)
        remain_vol = target_vol
        factory = FFTGenerator(nfreq=128)
        for _ in track(range(max_times)):
            # Generate particle
            particle = factory.generate_by_amplitude(A1, A3, A16, A37)
            particle.scale(np.random.uniform(dmin, dmax) / particle.calc_diameter() )
            particle.rotate(np.random.normal(mean, var) - particle.calc_dipAngle())
            particle.pid = len(self.particleCollection)
            particle.gid = gid
            if particle.calc_area() > remain_vol: continue
            # Allocate particle
            for _ in range(max_iters):
                xmin, ymin, xmax, ymax = self.doi
                x = np.random.uniform(xmin, xmax)
                y = np.random.uniform(ymin, ymax)
                particle.moveTo(np.array([x, y]))
                if self._boundary_periodic:
                     if not self.has_collision(particle, self._minimum_gap):
                        if particle.contain_domain_vertex(self.doi, 2*self._minimum_gap):
                            occupy_vol = self.assign_domain_vertex_particles(particle, self.doi)
                            remain_vol = remain_vol - occupy_vol
                            break
                        elif particle.intersect_domain_edge(self.doi, 2*self._minimum_gap):
                            occupy_vol = self.assign_domain_edge_particles(particle, self.doi)
                            remain_vol = remain_vol - occupy_vol
                            break
                        elif particle.is_within_domain(self.doi, -self._minimum_gap):
                            self._bkgGrid.assign_particle_to_cells(particle)
                            self.particleCollection.append(particle)
                            remain_vol = remain_vol - particle.calc_area()
                            break 
                elif particle.is_within_domain(self.doi, -self._minimum_gap):
                        if not self.has_collision(particle, self._minimum_gap):
                            self._bkgGrid.assign_particle_to_cells(particle)
                            self.particleCollection.append(particle)
                            remain_vol = remain_vol - particle.calc_area()
                            break 

        finish_vol = target_vol - remain_vol
        self.group_particle_sizes.append(len(self.particleCollection))                 
        rprint(f'[ GROUP{gid} ] {finish_vol} of {target_vol} is finished.')

    ## Output
    def write_gmsh_model(self, fname):
        suffix = os.path.splitext(os.path.basename(fname))[1]
        if suffix != '.geo':
            print('Invalid file format!')
            return
        
        with open(fname, 'wt') as file:
            file.write('SetFactory("OpenCASCADE");\n')
            file.write(f'lc = {self._minimum_gap};\n')
            point_off = 1
            curve_off = 1
            surface_off = 1
            # write particles
            file.write('/* Add irregular particles */\n')
            for i, particle in enumerate(self.particleCollection):
                for j, point in enumerate(particle.points):
                    file.write(f'Point({j + point_off}) = {{{point[0]}, {point[1]}, 0.0, lc}};\n')
                file.write(f'BSpline({i + curve_off}) = {{{point_off}:{point_off + len(particle.points) - 1}, {point_off}}};\n')
                file.write(f'Curve Loop({i + surface_off}) = {{{i + curve_off}}};\n')
                file.write(f'Plane Surface({i + surface_off}) = {{{i + surface_off}}};\n')
                point_off = point_off + len(particle.points) + 1
            curve_off = curve_off + len(self.particleCollection)
            surface_off = surface_off + len(self.particleCollection)
            # write injection
            # file.write('/* Add injection borehole */\n')
            # file.write('p_1 = newp;\n')
            # file.write(f'Point(p_1) = {{5.0, 4.9, 0.0, 1.0}};\n')
            # file.write('p_2 = newp;\n')
            # file.write(f'Point(p_2) = {{5.0, 5.1, 0.0, 1.0}};\n')
            # file.write('l_0 = newc;\n')
            # file.write(f'Line(l_0) = {{p_1, p_2}};\n')
            # write domain
            file.write('/* Add domain of interest (DOI) */\n')
            width = self.doi[2] - self.doi[0]
            height = self.doi[3] - self.doi[1]
            file.write(f'Rectangle({surface_off}) = {{{self.doi[0]}, {self.doi[1]}, 0.0, {width}, {height}, 0.0}};\n')
            # group by grading
            physical_off = 1
            for i in range(len(self.group_particle_sizes) - 1):
                beg = self.group_particle_sizes[i] + 1
                end = self.group_particle_sizes[i + 1]
                file.write(f'Physical Surface("grain-{i + 1}-1", {physical_off + i}) = {{{beg}:{end}}};\n')
                file.write(f'BooleanIntersection {{ Surface{{{beg}:{end}}}; Delete; }}{{ Surface{{{surface_off}}}; }}\n')
            physical_off = physical_off + len(self.group_particle_sizes)
            file.write(f'grains() = BooleanFragments {{ Surface{{:}}; Delete; }}{{}};\n')
            # physical group
            file.write('/* Add physical groups */\n')
            file.write(f'Physical Surface("grain-{physical_off - 1}-1", {physical_off - 1}) = {{ grains(#grains()-1) }};\n')
            file.write(f'Physical Surface("block-1", {physical_off}) = {{ Surface{{:}} }};\n')
            file.write(f'Physical Curve("constraint-1", newreg) = {{ Curve{{:}} }};\n')
            # scaling
            # file.write('/* Scale */\n')
            # file.write(f'Dilate {{{{0, 0, 0}}, {{10, 10, 1}}}} {{ Point{{:}}; Curve{{:}}; Surface{{:}}; }}\n')
            # mesh size

if __name__ == '__main__':
    doi = (0.0, 0.0, 10.0, 10.0)
    xmin, ymin, xmax, ymax = doi
    bkgGrid = BackgroundGrid(doi, spacing=0.05*(xmax - xmin))
    manager = ParticleManager(doi)
    manager.background_grid = bkgGrid
    manager.minimum_gap = 0.005*(xmax - xmin)
    manager.boundary_periodic = True

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), layout='none')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_box_aspect(1.0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    # ax.set_title(f'Number of particles: {manager.get_number_of_particles()}')
    nrows, ncols = bkgGrid.shape
    ax.set_xticks(np.linspace(xmin, xmax, 11))
    ax.set_yticks(np.linspace(ymin, ymax, 11))
    ax.grid(True, ls='--')

      # np.random.seed(7)
    particle_factory = FFTGenerator(nfreq=128)
    particle = particle_factory.generate_by_amplitude()
    particle.render(ax, 'r')
    left = PolygonParticle(particle.points.copy())
    left.translate(np.array([doi[0] - doi[2], 0.0]))
    right = PolygonParticle(particle.points.copy())
    right.moveTo(np.array([doi[2] - doi[0], 0.0]))
    up = PolygonParticle(particle.points.copy())
    up.moveTo(np.array([0.0, doi[3] - doi[1]]))
    down = PolygonParticle(particle.points.copy())
    down.moveTo(np.array([0.0, doi[1] - doi[3]]))
    diag = PolygonParticle(particle.points.copy())
    diag.moveTo(np.array([doi[2] - doi[0], doi[3] - doi[1]]))
    # right.translate()
    # visualization
    left.render(ax, 'r')
    right.render(ax, 'c')
    up.render(ax, 'c')
    down.render(ax, 'c')
    diag.render(ax, 'c')
    plt.show()
    fig.savefig('img\corner_boundary.svg', dpi=330, transparent=True)




