from numpy._core.multiarray import array as array
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
    def moveTo(self, dest:np.array):
        print('This function should be overwritten!')

    def translate(self, vec:np.array):
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
    def calc_diameter(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def calc_dipAngle(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def calc_elongation(self) -> float:
        print('This function should be overwritten!')

    @abstractmethod
    def render(self, axes):
        print('This function should be overwritten!')

    # Predicates
    @abstractmethod
    def is_AABB_intersect(self, other, gap=0.0):
        pass

    @abstractmethod
    def is_RECT_intersect(self, other, gap=0.0):
        pass

    @abstractmethod
    def is_exact_intersect(self, other, gap=0.0):
        pass



class EllipseParticle(Particle):
    '''Paticles represented by analytical expression'''
    def __init__(self, a, b) -> None:
        super().__init__()
        self.major_rad = a
        self.minor_rad = b

    def __repr__(self) -> str:
        return f'Ellipse particle: a = {self.major_rad}, b = {self.minor_rad}'

    def moveTo(self, dest: np.array):
        return super().move_to(dest)
    
    def translate(self, vec: np.array):
        return super().translate(vec)
    
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
    
    def calc_diameter(self) -> float:
        return super().calc_diameter()
    
    def calc_dipAngle(self) -> float:
        return super().calc_dipAngle()
    
    def calc_elongation(self) -> float:
        return super().calc_elongation()
    
    def render(self, axes):
        return super().render(axes)
    
    def is_AABB_intersect(self, other, gap=0.0):
        return super().is_AABB_intersect(other)
    
    def is_RECT_intersect(self, other, gap=0.0):
        return super().is_RECT_intersect(other)
    
    def is_exact_intersect(self, other, gap=0.0):
        return super().is_exact_intersect(other)

class PolygonParticle(Particle):
    '''Particles represented by point sets'''
    def __init__(self, pts) -> None:
        super().__init__()
        self.points = pts

    def centroid(self):
        polygon = shapely.Polygon(self.points)
        return list(polygon.centroid.coords)

    def moveTo(self, dest:list):
        polygon = shapely.Polygon(self.points)
        origin = polygon.centroid.coords
        self.points += (dest - origin)

    def translate(self, vector: np.array):
        self.points += vector
    
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
    
    def calc_diameter(self) -> float:
        polygon = shapely.Polygon(self.points)
        coords = polygon.minimum_rotated_rectangle.boundary.coords
        a = shapely.LineString([coords[0], coords[1]]).length
        b = shapely.LineString([coords[1], coords[2]]).length
        return np.min([a, b])
    
    def calc_dipAngle(self) -> float:
        polygon = shapely.Polygon(self.points)
        coords = polygon.minimum_rotated_rectangle.boundary.coords
        a = shapely.LineString([coords[0], coords[1]]).length
        b = shapely.LineString([coords[1], coords[2]]).length
        v = (np.array(coords[1]) - np.array(coords[0])) if a > b else (np.array(coords[2]) - np.array(coords[1]))
        return np.degrees(np.arctan2(v[1], v[0]))
    
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
    
    # Predicate
    def is_AABB_intersect(self, other, gap=0.0):
        bbox_1 = shapely.Polygon(self.points).buffer(gap).bounds
        bbox_2 = shapely.Polygon(other.points).buffer(gap).bounds
        poly_1 = shapely.geometry.box(*bbox_1)
        poly_2 = shapely.geometry.box(*bbox_2)
        return poly_1.intersects(poly_2)
    
    def is_RECT_intersect(self, other, gap=0.0):
        poly_1 = shapely.Polygon(self.points).minimum_rotated_rectangle.buffer(gap)
        poly_2 = shapely.Polygon(other.points).minimum_rotated_rectangle.buffer(gap)
        return poly_1.intersects(poly_2)
    
    def is_exact_intersect(self, other, gap=0.0):
        poly_1 = shapely.Polygon(self.points).buffer(gap)
        poly_2 = shapely.Polygon(other.points).buffer(gap)
        return poly_1.intersects(poly_2)
    
    def is_within_domain(self, bounds:tuple, tol):
        polygon = shapely.Polygon(self.points)
        boundary = shapely.box(*bounds).buffer(tol)
        return polygon.within(boundary)
    
    def contain_domain_vertex(self, bounds:tuple, tol=0.0) -> bool:
        polygon = shapely.Polygon(self.points).buffer(-tol)
        coords = shapely.box(*bounds).exterior.coords[:-1]
        for xyz in coords:
            if polygon.contains_properly(shapely.Point(xyz)):
                return True
        return False

    def intersect_domain_edge(self, bounds:tuple, tol=0.0) -> bool:
        polygon = shapely.Polygon(self.points).buffer(-tol)
        boundary = shapely.box(*bounds).exterior
        coords = boundary.coords
        num_edges = 0
        for i in range(len(coords[:-1])):
            line = shapely.LineString([coords[i], coords[i + 1]])
            if polygon.intersects(line):
                num_edges += 1
        is_intersect_edge = True if num_edges == 1 else False
        return is_intersect_edge

    def render(self, ax, color, add_bbox=False, add_rect=False):
       #ax.plot(self.points[:, 0], self.points[:, 1], 'b.')
       polygon = shapely.Polygon(self.points)
       shapely.plotting.plot_polygon(polygon, ax, add_points=False, fill=True, clip_on=False, 
                                     fc=color, ec='k', ls='-')
       # AABB
       if add_bbox:
           bbox = shapely.box(*polygon.bounds)
           shapely.plotting.plot_polygon(bbox, ax, add_points=False, fill=False, clip_on=False,
                                         fc=color, ec='k', ls='-.')
       # RECT
       if add_rect: 
           rect = polygon.minimum_rotated_rectangle
           shapely.plotting.plot_polygon(rect, ax, add_points=False, fill=False, clip_on=False,
                                         fc=color, ec='r',ls='-.')
           
        

## Debug ##
if __name__ == '__main__':
    print('Particle class')
    particle = EllipseParticle(2, 4)
    print(particle)