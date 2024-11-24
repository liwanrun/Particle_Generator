import os
import random
import gmsh
import shapely
import numpy as np
from particle import PolygonParticle
# from background_grid import BackgroundGrid
from fft_generator import FFTGenerator
from rich import print as rprint
from rich.progress import track

# Library code
class ParticleAllocator:
    '''Particle allocator using candidate seed points'''
    def __init__(self, doi) -> None:
        self.doi = doi                      # [xmin, ymin, xmax, ymax]
        self.particleArr = []               # Allocated particles
        self.particleVol = 0.0
        self.particle_size_of_group = [0]
        self._periodic_boundary = False
        self._minimum_gap = 0.0
        self._grading_ratio = 0.0           # Specified grading ratio
        self._grading_limits = None         # Specified grading diameters

    def create_seed_points(self, spacing):
        '''Generate seed points for placing particles'''
        x1, y1, x2, y2 = self.doi
        m = int(np.ceil((y2 - y1) / spacing))
        n = int(np.ceil((x2 - x1) / spacing))
        x = np.linspace(x1, x2, (n + 1))
        y = np.linspace(y1, y2, (m + 1))
        xx, yy = np.meshgrid(x, y)
        self.seedPoints = np.stack((xx, yy), axis=-1, dtype = float)
        self.seedStates = np.zeros((m+1, n+1), dtype=bool)
        self.seed_spacing = spacing

    def sample_seed_point(self, idx:int, candidates:np.array) -> np.array:
        if self._periodic_boundary:
            any_corner = np.array([10.0, 0.0])
            seed_point = any_corner if not len(self.particleArr) else candidates[idx]
            return seed_point
        else:
            return candidates[idx]
    
    # Group-specific functions
    def set_group_sizes(self, dmin, dmax):
        '''Uniform sampling'''
        self.group_sizes = np.array([dmin, dmax])

    def set_group_dips(self, mean, var):
        '''Normal sampling'''
        self.group_dips = np.array([mean, var])

    def set_group_spectrum(self, A1, A3, A16, A37):
        '''Fourier descriptors'''
        self.group_spectrums = np.array([A1, A3, A16, A37])

    def set_group_particle(self, particle:PolygonParticle):
        '''Allocate particles with same shapes'''
        self.group_particle = particle

    # Private functions
    def get_group_volume(self, d_l, d_r, factor=0.5) -> float:
        '''P(d) = 100*(d/d_max)^n'''
        dmin, dmax = self._grading_limits
        Pmax = 100 * (dmax / dmax)**(factor)
        Pmin = 100 * (dmin / dmax)**(factor)
        Pl = 100 * (d_l / dmax)**(factor)
        Pr = 100 * (d_r / dmax)**(factor)
        w = self.doi[2] - self.doi[0]
        h = self.doi[3] - self.doi[1]
        vol = w * h * self._grading_ratio
        return (Pr - Pl) / (Pmax - Pmin) * vol 
    
    ## Important subroutine
    def get_particle_indices(self, p:PolygonParticle) -> np.array:
        #buffer = np.sqrt(2.0) * self.seed_spacing
        buffer = 0.5 * self._minimum_gap
        nrows, ncols = self.seedPoints.shape[:-1]
        x1, y1, x2, y2 = p.boundingBox(buffer)
        r1 = max(int(np.ceil(y1 / self.seed_spacing)), 0) 
        r2 = min(int(np.ceil(y2 / self.seed_spacing)), nrows)
        c1 = max(int(np.ceil(x1 / self.seed_spacing)), 0)
        c2 = min(int(np.ceil(x2 / self.seed_spacing)), ncols)
        # indices = np.dstack(np.mgrid[r1:r2, c1:c2])
        return np.array([r1, r2, c1, c2])

    def has_collision(self, p:PolygonParticle) -> bool:
        r1, r2, c1, c2 = self.get_particle_indices(p)
        points = self.seedPoints[r1:r2, c1:c2]
        states = self.seedStates[r1:r2, c1:c2]
        buffer = 0.5 * self._minimum_gap
        seed_points = points[states == True]
        for seed in seed_points:
            if p.contain_point(seed, tol=buffer):
                return True
        return False
    
    def update_particle_seeds(self, p:PolygonParticle) -> None:
        r1, r2, c1, c2 = self.get_particle_indices(p)
        points = self.seedPoints[r1:r2, c1:c2]
        states = self.seedStates[r1:r2, c1:c2]
        buffer = 0.5 * self._minimum_gap
        # buffer = np.sqrt(2.0) * self.seed_spacing
        for idx, val in np.ndenumerate(states):
            if (val == True): continue
            i, j = idx; pt = points[i, j]
            if p.contain_point(pt, tol=buffer):
                states[i, j] = True
        self.seedStates[r1:r2, c1:c2] = states
    
    def is_particle_on_corner(self, p:PolygonParticle, tol) -> bool:
        polygon = shapely.Polygon(p.points).buffer(-tol)
        coords = shapely.box(*self.doi).exterior.coords[:-1]
        for xyz in coords:
            if polygon.contains_properly(shapely.Point(xyz)):
                return True
        return False
        
    def get_corner_index(self, p:PolygonParticle) -> int:
        polygon = shapely.Polygon(p.points)
        corners = shapely.box(*doi).exterior.coords[:-1]
        for index, point in enumerate(corners):
            if polygon.contains_properly(shapely.Point(point)): 
                return index
        return -1
    
    def alloc_corner_particle(self, index:int, particle:PolygonParticle) -> float:
        if -1 == index: return 0.0
        x1, y1, x2, y2 = self.doi
        period_points = np.array([[[x2, y2], [x1, y2], [x1, y1]],
                                  [[x1, y2], [x1, y1], [x2, y1]],
                                  [[x1, y1], [x2, y1], [x2, y2]],
                                  [[x2, y1], [x2, y2], [x1, y2]]], dtype=float)
        ghost_particles = []
        for point in period_points[index]:
            ghost = PolygonParticle(particle.points.copy())
            ghost.moveTo(point)
            if self.has_collision(ghost): return 0.0
            ghost_particles.append(ghost)
        # Allocate corner particles
        self.update_particle_seeds(particle)
        self.particleArr.append(particle)
        for ghost in ghost_particles:
            ghost.pid = len(self.particleArr)
            ghost.gid = particle.gid
            self.update_particle_seeds(ghost)
            self.particleArr.append(ghost)
        return particle.calc_area()
    
    def is_particle_on_boundary(self, p:PolygonParticle) -> bool:
        particle = shapely.Polygon(p.points).exterior
        boundary = shapely.box(*self.doi).exterior
        return particle.intersects(boundary)

    def is_particle_within_boundary(self, p:PolygonParticle, tol:float) -> bool:
        particle = shapely.Polygon(p.points).buffer(tol)
        boundary = shapely.box(*self.doi)
        return particle.within(boundary)

    def get_boundary_indexes(self, p:PolygonParticle) -> np.array:
        polygon = shapely.Polygon(p.points)
        coords = shapely.box(*doi).exterior.coords
        lines = [shapely.LineString([coords[i], coords[i+1]]) for i in range(len(coords) - 1)]
        indexes = []
        for index, line in enumerate(lines):
            if polygon.crosses(line):
                indexes.append(index)
        return np.array(indexes)

    def alloc_boundary_particle(self, indexes:np.array, particle:PolygonParticle) -> float:
        if indexes.size == 0: return 0.0
        x1, y1, x2, y2 = self.doi
        delta_x = x2 - x1
        delta_y = y2 - y1
        offsets = np.array([[-delta_x, 0.0], [0.0, -delta_y], 
                            [delta_x, 0.0], [0.0, delta_y]])
        ghost_particles = []
        for index in indexes:
            ghost = PolygonParticle(particle.points.copy())
            ghost.translate(offsets[index])
            if self.has_collision(ghost): return 0.0
            ghost_particles.append(ghost)
        # Allocate corner particles
        self.update_particle_seeds(particle)
        self.particleArr.append(particle)
        for ghost in ghost_particles:
            ghost.pid = len(self.particleArr)
            ghost.gid = particle.gid
            self.update_particle_seeds(ghost)
            self.particleArr.append(ghost)
        return particle.calc_area()

    ## Interface functions
    def allocate_particle_group(self, gid:int, max_iters:int=1000, max_times:int=100):
        A1, A3, A16, A37 = self.group_spectrums
        d1, d2 = self.group_sizes
        mu, sig = self.group_dips
        remain_vol = target_vol = self.get_group_volume(d1, d2)
        factory = FFTGenerator(nfreq=128)
        for _ in track(range(max_times)):
            # Generate particle
            particle = factory.generate_by_amplitude(A1, A3, A16, A37)
            corner_vol = particle.calc_area()
            dia = particle.calc_diameter()
            dip = particle.calc_dipAngle()
            rng = np.random.default_rng(0)
            particle.scale(rng.uniform(d1, d2) / dia)
            particle.rotate(rng.normal(mu, sig) - dip)
            particle.pid = len(self.particleArr)
            particle.gid = gid 
            if corner_vol > remain_vol: continue
            candidate_seeds = self.seedPoints[self.seedStates == False]
            np.random.shuffle(candidate_seeds)
            for i, _ in enumerate(candidate_seeds):
                seed_point = self.sample_seed_point(i, candidate_seeds)
                particle.moveTo(seed_point)
                if self._periodic_boundary:
                    if self.has_collision(particle): continue
                    # Seed types of corner and boundary points
                    if not self.is_particle_on_boundary(particle):
                        '''Interior particle'''
                        if particle.is_closely_within_domain(self.doi, self._minimum_gap):
                            self.update_particle_seeds(particle)
                            self.particleArr.append(particle)
                            remain_vol = remain_vol - particle.calc_area()
                            break
                    elif self.is_particle_on_corner(particle, self._minimum_gap):
                        '''Corner particle'''
                        index = self.get_corner_index(particle)
                        corner_vol = self.alloc_corner_particle(index, particle)
                        remain_vol = remain_vol - corner_vol
                        break
                    else:
                        '''Edge particle'''
                        if particle.is_closely_cross_boundary(self.doi, self._minimum_gap):
                            indexes = self.get_boundary_indexes(particle)
                            margin_vol = self.alloc_boundary_particle(indexes, particle)
                            remain_vol = remain_vol - margin_vol
                            break
                elif particle.is_closely_within_domain(self.doi, self._minimum_gap):
                    if not self.has_collision(particle):
                        self.update_particle_seeds(particle)
                        self.particleArr.append(particle)
                        remain_vol = remain_vol - particle.calc_area()
                        break

        finish_vol = target_vol - remain_vol
        self.particleVol += finish_vol
        self.particle_size_of_group.append(len(self.particleArr))                 
        rprint(f'[ GROUP{gid} ] {finish_vol} of {target_vol} is finished.')

    def get_number_of_particles(self) -> int:
        return len(self.particleArr)
    
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
            for i, particle in enumerate(self.particleArr):
                for j, point in enumerate(particle.points):
                    file.write(f'Point({j + point_off}) = {{{point[0]}, {point[1]}, 0.0, lc}};\n')
                file.write(f'BSpline({i + curve_off}) = {{{point_off}:{point_off + len(particle.points) - 1}, {point_off}}};\n')
                file.write(f'Curve Loop({i + surface_off}) = {{{i + curve_off}}};\n')
                file.write(f'Plane Surface({i + surface_off}) = {{{i + surface_off}}};\n')
                point_off = point_off + len(particle.points) + 1
            curve_off = curve_off + len(self.particleArr)
            surface_off = surface_off + len(self.particleArr)
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
            for i in range(len(self.particle_size_of_group) - 1):
                beg = self.particle_size_of_group[i] + 1
                end = self.particle_size_of_group[i + 1]
                file.write(f'Physical Surface("grain-{i + 1}-1", {physical_off + i}) = {{{beg}:{end}}};\n')
                file.write(f'BooleanIntersection {{ Surface{{{beg}:{end}}}; Delete; }}{{ Surface{{{surface_off}}}; }}\n')
            physical_off = physical_off + len(self.particle_size_of_group)
            file.write(f'grains() = BooleanFragments {{ Surface{{:}}; Delete; }}{{}};\n')
            # delete points outside boundary
            file.write('/* Delete points beyond boundary */\n')
            dist_tol = 1.0e-06
            file.write(f'outPts[] = Point {{:}};\n')
            file.write(f'inPts[] = Point In BoundingBox {{                    \
                       {self.doi[0]-dist_tol}, {self.doi[1]-dist_tol}, 0.0,   \
                       {self.doi[2]+dist_tol}, {self.doi[3]+dist_tol}, 0.0}};\n')
            file.write(f'outPts -= inPts[];\n')
            file.write(f'Recursive Delete {{ Point {{outPts[]}};}}\n')
            file.write(f'MeshSize {{inPts[]}} = lc;\n')
            # physical group
            file.write('/* Add physical groups */\n')
            file.write(f'Physical Surface("grain-{physical_off - 1}-1", {physical_off - 1}) = {{ grains(#grains()-1) }};\n')
            file.write(f'Physical Surface("block-1", {physical_off}) = {{ Surface{{:}} }};\n')
            file.write(f'Physical Curve("constraint-1", newreg) = {{ Curve{{:}} }};\n')
            # scaling
            # file.write('/* Scale */\n')
            # file.write(f'Dilate {{{{0, 0, 0}}, {{10, 10, 1}}}} {{ Point{{:}}; Curve{{:}}; Surface{{:}}; }}\n')
            # mesh size

    def get_contour_points(self, p:PolygonParticle) -> np.array:
        try:
            boundary = shapely.box(*self.doi)
            particle = shapely.Polygon(p.points)
            profile = shapely.intersection(boundary, particle.exterior)
            contour = shapely.line_merge(profile, directed=False)
            return np.array(contour.coords)
        except NotImplementedError as e:
            print(f"Line merging failed: {e}")


    def export_gmsh_geo_file(self, fname:str):
        suffix = os.path.splitext(os.path.basename(fname))[1]
        if (suffix != '.geo'): print('Invalid file format!'); return
        with open(fname, 'wt') as file:
            file.write('SetFactory("OpenCASCADE");\n')
            file.write(f'lc = {self._minimum_gap};\n')
            point_off = 1
            curve_off = 1
            # write particle contour
            file.write('/* Conglomerate aggregate */\n')
            for _, particle in enumerate(self.particleArr):
                if self.is_particle_on_boundary(particle):
                    points = self.get_contour_points(particle)
                    for j, point in enumerate(points):
                        file.write(f'Point({j + point_off}) = {{{point[0]}, {point[1]}, 0.0, lc}};\n')
                    file.write(f'BSpline({curve_off}) = {{{point_off}:{point_off + len(points) - 1}}};\n')
                    point_off += len(points)
                    curve_off += 1
                else:
                    points = particle.points
                    for j, point in enumerate(points):
                        file.write(f'Point({j + point_off}) = {{{point[0]}, {point[1]}, 0.0, lc}};\n')
                    file.write(f'BSpline({curve_off}) = {{{point_off}:{point_off + len(points) - 1}, {point_off}}};\n')
                    point_off += len(points) + 1
                    curve_off += 1
              # write domain
            file.write('/* Add domain of interest (DOI) */\n')
            x1, y1, x2, y2 = self.doi
            file.write(f'Rectangle(1) = {{{x1}, {y1}, 0.0, {x2 - x1}, {y2 - y1}, 0.0}};\n')
            file.write(f'Physical Surface("domain", 1) = {{1}};\n')
            file.write(f'frag() = BooleanFragments {{ Surface{{:}}; Delete; }} {{ Curve{{:}}; Delete; }};\n')
            file.write(f'sand_phases() = Physical Surface{{:}};\n')
            file.write(f'Printf("Sand phases: ", sand_phases());\n')
            file.write(f'grain_phases() = Surface{{:}};\n')
            file.write(f'grain_phases() -= sand_phases();\n')
            file.write(f'Printf("Grain phases: ", grain_phases());\n')
            num_groups = len(self.particle_size_of_group)
            for gid in np.arange(num_groups - 1):
                m = self.particle_size_of_group[gid]
                n = self.particle_size_of_group[gid + 1]
                file.write(f'Physical Surface("grain-{gid + 1}-1", newreg) = grain_phases({m}):grain_phases({n - 1});\n')
            file.write(f'Physical Surface("grain-{num_groups}-1", newreg) = sand_phases();\n')
            # file.write(f'Physical Surface ("Grain", {physical_surface_off+1}) = Grain_phases(:);\n')
            file.write(f'Physical Surface("block-1", newreg) = {{ Surface{{:}} }};\n')
            file.write(f'Physical Curve("constraint-1", newreg) = {{ Curve{{:}} }};\n')
            file.write(f'mesh_points() = Point{{:}};\n')
            file.write(f'MeshSize {{mesh_points()}} = lc;')  

    def mesh_with_gmsh(self, fname:str):
        '''Generate mesh file with high-quality mesh and group info.'''
        lc = self._minimum_gap*0.8
        gmsh.initialize() 

        # Particles represented by Spline curve
        point_off = 1
        curve_off = 1
        splines = []
        for _, particle in enumerate(self.particleArr):
            pointTags = []
            for id, pt in enumerate(particle.points):
                idx = point_off + id; pointTags.append(idx)
                gmsh.model.occ.add_point(pt[0], pt[1], 0.0, lc, tag=idx)
            pointTags.append(point_off)
            point_off += len(pointTags)
            gmsh.model.occ.add_bspline(pointTags, tag=curve_off)
            splines.append((1, curve_off))
            curve_off += 1

        # Rectangular domain
        x1, y1, x2, y2 = self.doi
        doi = gmsh.model.occ.add_rectangle(x1, y1, 0.0, x2 - x1, y2 - y1)
        gmsh.model.occ.synchronize()
        _, outDimTagsMap = gmsh.model.occ.intersect([(2, doi)], splines)
        gmsh.model.occ.synchronize()
        # Union broken spline contours avoiding silver element
        contours = outDimTagsMap[1:]#.copy()
        for i, dimTags in enumerate(contours):
            if len(dimTags) > 1: 
                tags, _ = gmsh.model.occ.fuse([dimTags[0]], dimTags)
                contours[i] = tags
        gmsh.model.occ.synchronize() #; print(contours)
        # Create particle surfaces
        profiles = [item[0] for item in contours] #; print(profiles)
        doi = gmsh.model.occ.add_rectangle(x1, y1, 0.0, x2 - x1, y2 - y1)
        gmsh.model.occ.fragment([(2, doi)], profiles)
        gmsh.model.occ.synchronize() #; print(outDimTagsMap)
        # Find matrix surface tag
        curve_1, curve_2 = random.choices(profiles, k=2)
        ups_1, _ = gmsh.model.getAdjacencies(curve_1[0], curve_1[1])
        ups_2, _ = gmsh.model.getAdjacencies(curve_2[0], curve_2[1])
        matrix_surface = list(filter(lambda x: x in ups_1, ups_2)) #; print(matrix_surface)
        # Find particle group surfaces
        grain_surfaces = []
        for curve in profiles:
            upwards, _ = gmsh.model.getAdjacencies(curve[0], curve[1])
            surface = [x for x in list(upwards) if x not in matrix_surface]
            # print(upwards); print(surface)
            grain_surfaces += surface

        # Define physical groups
        physical_off = 1
        num_groups = len(self.particle_size_of_group) - 1
        for gid in np.arange(num_groups):
            beg = self.particle_size_of_group[gid]
            end = self.particle_size_of_group[gid + 1]
            gmsh.model.addPhysicalGroup(dim=2, tags=grain_surfaces[beg:end], tag=(physical_off), name=f'grain-{physical_off}-1')
            physical_off += 1
        gmsh.model.addPhysicalGroup(dim=2, tags=matrix_surface, tag=(physical_off), name=f'grain-{physical_off}-1')
        surfaceTags = np.array(gmsh.model.getEntities(dim=2))[:, 1].tolist(); physical_off += 1 
        gmsh.model.addPhysicalGroup(dim=2, tags=surfaceTags, tag=physical_off, name='block-1')
        curveTags = np.array(gmsh.model.getEntities(dim=1))[:, 1].tolist(); physical_off += 1
        gmsh.model.addPhysicalGroup(dim=1, tags=curveTags, tag=physical_off, name='constraint-1')
        gmsh.model.occ.synchronize()

        # Define mesh size
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        pointTags = gmsh.model.occ.get_entities(dim=0)
        gmsh.model.mesh.set_size(pointTags, lc)
        gmsh.model.mesh.generate(2)
        gmsh.write(fname)

        gmsh.fltk.run()
        gmsh.finalize()

    def test_shapely(self):
        #factory = FFTGenerator(nfreq=128)
        # np.random.seed(1)
        #particle = factory.generate_by_amplitude()
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        particle = PolygonParticle(points)
        particle.scale(1.0)
        particle.rotate(90)
        particle.moveTo(np.array([5.0, 5.0]))
        print(self.get_contour_points(particle))

        polyline = shapely.Polygon(particle.points).exterior; 
        boundary = shapely.box(*self.doi); print(boundary)
        multi_line = shapely.intersection(boundary, polyline)
        line = shapely.line_merge(shapely.intersection(boundary, polyline))
        x, y = line.xy

        colors = ['red', 'blue', 'green', 'orange', 'purple']
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 10.0)
        ax.set_aspect(1.0)
        ax.plot(x, y, colors[0])
        plt.show()

# Client code
if __name__ == '__main__':
    # Global settings
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'   # 设置字体系列为 serif
    plt.rcParams['font.serif'] = ['Times']  # 设置 serif 字体为 Times
    plt.rcParams['font.size'] = 12

    #np.random.seed(2)
    rprint("==================== S T A R T ====================")
    # Global parameters
    doi = [0.0, 0.0, 10.0, 10.0]
    gap = 0.01* (doi[2] - doi[0])
    alloc = ParticleAllocator(doi)
    alloc.create_seed_points(spacing=0.5*gap)
    alloc._minimum_gap = gap
    alloc._grading_ratio = 0.5
    alloc._grading_limits = [0.4, 1.0]
    alloc._periodic_boundary = True
    #alloc.test_shapely()

    # Group parameters
    output_file = r'DD=45.0'
    ## Group 1
    alloc.set_group_spectrum(A1=0.50, A3=0.020, A16=0.001, A37=0.0000)
    alloc.set_group_sizes(dmin=0.6, dmax=1.0)
    alloc.set_group_dips(mean=45.0, var=2)
    alloc.allocate_particle_group(gid=0, max_iters=1000, max_times=100)

    ## Group 2
    alloc.set_group_spectrum(A1=0.50, A3=0.020, A16=0.001, A37=0.0000)
    alloc.set_group_sizes(dmin = 0.4, dmax=0.6)
    alloc.set_group_dips(mean=45.0, var=2)
    alloc.allocate_particle_group(gid=1, max_iters=1000, max_times=100)
    ## Group 3

    ## Information ##
    x1, y1, x2, y2 = doi
    numParticles = alloc.get_number_of_particles()
    percentage = alloc.particleVol / ((x2 - x1) * (y2 - y1))
    rprint(f'[ PROMPT ] {numParticles} particles account for {percentage*100}%.')
    rprint("===================== E  N  D =====================")

    m = int(np.ceil((doi[3] - doi[1]) / gap))
    n = int(np.ceil((doi[2] - doi[0]) / gap))
    x = np.linspace(doi[0], doi[2], (50 + 1))
    y = np.linspace(doi[1], doi[2], (50 + 1))
    xx, yy = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax.set_aspect(1.0)
    ax.set_box_aspect(1.0)
    #ax.set_title(f'Particle Volume: {alloc.particleVol:.2f} ({100*percentage:.2f}$\%$)')
    ax.set_title(f'Particle Number: {len(alloc.particleArr)}')
    ax.set_xlim([doi[0], doi[2]])
    ax.set_ylim([doi[1], doi[3]])
    ax.plot(xx, yy, 'ko', markersize=0.5)
    for particle in alloc.particleArr:
        #origin = particle.centroid()[0]
        #ax.text(origin[0], origin[1], f'{particle.pid}',color='k')
        particle.render(ax, color='C0', add_bbox=False, add_rect=False)
    plt.savefig(f'Orientation\{output_file}.svg', dpi=330, transparent=True)
    plt.show()
    
    ## Output for Gmsh
    #alloc.write_gmsh_model(f'Elongation\{output_file}.geo')
    #alloc.export_gmsh_geo_file(f'Elongation\particles.geo')
    alloc.mesh_with_gmsh(f'Orientation\{output_file}.msh')



