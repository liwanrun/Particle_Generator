import shapely
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon 
from abc import abstractmethod

import shapely.plotting

class Particle:
    '''Abstract particle class'''
    def __init__(self) -> None:
        self.pid = -1
        self.gid = -1

    def __repr__(self) -> str:
        pass

    @abstractmethod
    def move_to(self, dest:tuple):
        print('This function should be overwritten!')

    @abstractmethod
    def rotate(self, center:tuple, theta):
        print('This function should be overwritten!')

    @abstractmethod
    def scale(self, factor):
        print('This function should be overwritten!')

    @abstractmethod
    def calc_area(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def calc_perimeter(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def boundingBox(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def calc_elongation(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def render(self, axes):
        print('This function should be overwritten!')


class EllipseParticle(Particle):
    '''Paticles represented by analytical expression'''
    def __init__(self, a, b) -> None:
        super().__init__()
        self.major_rad = a
        self.minor_rad = b

    def __repr__(self) -> str:
        return f'Ellipse particle: a = {self.major_rad}, b = {self.minor_rad}'

    def move_to(self, dest: tuple):
        return super().move_to(dest)
    
    def rotate(self, center: tuple, theta):
        return super().rotate(center, theta)
    
    def scale(self, factor):
        return super().scale(factor)
    
    def calc_area(self) -> float:
        return super().calc_area()
    
    def calc_perimeter(self) -> float:
        return super().calc_perimeter()
    
    def boundingBox(self) -> float:
        return super().boundingBox()
    
    def calc_elongation(self) -> float:
        return super().calc_elongation()
    
    def render(self, axes):
        return super().render(axes)

class PolygonParticle(Particle):
    '''Particles represented by point sets'''
    def __init__(self, pts) -> None:
        super().__init__()
        self.points = pts

    def translate(self, dest: tuple):
        self.points += dest
    
    def rotate(self, theta):
        radian = np.radians(theta)
        matrix = np.array([[np.cos(radian), np.sin(radian)],
                           [-np.sin(radian), np.cos(radian)]])
        self.points = np.dot(self.points, matrix) 
    
    def scale(self, factor):
        self.points *= factor
    
    def calc_area(self) -> float:
        polygon = shapely.Polygon(self.points)
        return polygon.area
    
    def calc_perimeter(self) -> float:
        polygon = shapely.Polygon(self.points)
        return polygon.length
    
    def boundingBox(self, dist=0.0) -> tuple:
        polygon = shapely.Polygon(self.points)
        return polygon.buffer(dist).bounds
    
    def calc_elongation(self) -> float:
        polygon = shapely.Polygon(self.points)
        coords = polygon.minimum_rotated_rectangle.boundary.coords
        a = shapely.LineString([coords[0], coords[1]]).length
        b = shapely.LineString([coords[1], coords[2]]).length
        major_length = np.max([a, b])
        minor_length = np.min([a, b])
        return (major_length / minor_length)

    def calc_roundness(self) -> float:
        A = self.calc_area()
        P = self.calc_perimeter()
        return 2.0*np.sqrt(np.pi * A) / P

    def calc_angularity(self) -> float:
        coords = self.points
        vectors  = np.roll(coords, -1, axis=0) - coords
        radians = np.arctan2(vectors[:, 1], vectors[:, 0])
        radians = np.append(radians, [radians[0]], axis=0)
        radians = np.abs(np.diff(radians, 1, axis=0))
        radians[radians > np.pi] = 2.0 * np.pi - radians[radians > np.pi]
        return np.sum(radians) / (2.0 * np.pi) - 1.0

    def is_valid(self) -> bool:
        polygon = shapely.Polygon(self.points)
        return polygon.is_valid

    def render(self, ax):
       # ax.plot(self.points[:, 0], self.points[:, 1], 'b.')
       polygon = shapely.Polygon(self.points)
       min_box = polygon.minimum_rotated_rectangle
       shapely.plotting.plot_polygon(polygon, ax, add_points=False)
       shapely.plotting.plot_polygon(min_box, ax, add_points=True, fill=False, ls='--')

## Debug ##
if __name__ == '__main__':
    print('Particle class')

    particle = EllipseParticle(2, 4)
    print(particle)