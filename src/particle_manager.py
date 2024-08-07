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

    def generate_particle_group(self, max_iters=1000, max_times=100):
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
            if particle.calc_area() > remain_vol: continue
            # Allocate particle
            for _ in range(max_iters):
                xmin, ymin, xmax, ymax = self.doi
                x = np.random.uniform(xmin, xmax)
                y = np.random.uniform(ymin, ymax)
                particle.moveTo(np.array([x, y]))
                if particle.is_within_domain(self.doi):
                    if not self.has_collision(particle, self._minimum_gap):
                        self._bkgGrid.assign_particle_to_cells(particle)
                        self.particleCollection.append(particle)
                        remain_vol = remain_vol - particle.calc_area()
                        break              
        finish_vol = target_vol - remain_vol
        rprint(f'[ PROMPT ] {finish_vol} of {target_vol} is finished.')

if __name__ == '__main__':
    pass



