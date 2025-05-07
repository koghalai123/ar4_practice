import trimesh
import os
import csv
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, box


def load_vertices(filepath):
    """Load vertices from CSV file"""
    vertices = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:  
                temp= np.array(row,dtype = float)

                vertices.append(temp)
    return vertices

def load_faces(filepath):
    """Load faces from CSV file"""
    faces = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            temp= np.array(row,dtype = float)
            faces.append(temp) 
    return faces






package_dir = os.path.dirname(os.path.abspath(__file__))
vertices_path = os.path.join(package_dir, 'vertices.csv')
faces_path = os.path.join(package_dir, 'faces.csv')

vertices = load_vertices(vertices_path)
faces = load_faces(faces_path)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
plane_normal = [0, 0, 1]
plane_point = [0, 0, 0.1]

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
section = mesh.section(plane_origin=plane_point, plane_normal=plane_normal)

scene = trimesh.Scene([mesh, section])
#scene.show()
#section.show()


# Compute the intersection (as befor
# Extract vertices of the intersection
intersection_vertices = section.vertices

if np.allclose(plane_normal, [0, 0, 1]):  # XY plane
    points_2d = intersection_vertices[:, :2]
elif np.allclose(plane_normal, [0, 1, 0]):  # XZ plane
    points_2d = intersection_vertices[:, [0, 2]]
else:  # YZ plane
    points_2d = intersection_vertices[:, 1:]

# 2. Initialize the intersector (uses pyembree for acceleration)
intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

# 3. Define rays (origins + directions)

startRayPoint = np.array([
    plane_point,
])+np.array([0,-3,0])
startRayDirection = np.array([[0,1,0]])
startLocations = intersector.intersects_location(
    startRayPoint, 
    startRayDirection)[0]

firstIntersection = startLocations[startLocations[:,1].argsort(),:][0,:]
secondIntersection = startLocations[startLocations[:,1].argsort(),:][1,:]
searchOrigin = (firstIntersection-secondIntersection)/2+secondIntersection

yRange = 0.9*np.sum(np.abs(firstIntersection-secondIntersection))
squarePoints = secondIntersection+np.sign(firstIntersection-secondIntersection).sum()*np.array([[0,0,0],[0,0,0],[0,yRange,0],[0,yRange,0]])+np.array([[-yRange/2,0,0],[yRange/2,0,0],[-yRange/2,0,0],[yRange/2,0,0]])


secondRayPoint = np.array([
    searchOrigin,
])
secondRayDirection = np.array([[1,0,0]])
secondLocations = intersector.intersects_location(
    secondRayPoint, 
    secondRayDirection,multiple_hits = False)[0]

findYdistPoints1 = np.linspace(searchOrigin,secondLocations.flatten())
findYdistDirections1 = np.linspace(np.array([0,1,0]),np.array([0,1,0]))
Y1Intersections = intersector.intersects_location(
    findYdistPoints1, 
    findYdistDirections1,multiple_hits = False)[0]

findYdistPoints2 = findYdistPoints1
findYdistDirections2 = np.linspace(np.array([0,-1,0]),np.array([0,-1,0]))
Y2Intersections = intersector.intersects_location(
    findYdistPoints2, 
    findYdistDirections2,multiple_hits = False)[0]

findXdistPoints = np.linspace(firstIntersection,secondIntersection)
findXdistDirections = np.linspace(np.array([1,0,0]),np.array([1,0,0]))
XIntersections = intersector.intersects_location(
    findXdistPoints, 
    findXdistDirections,multiple_hits = False)[0]


plt.figure(figsize=(8, 6))

# Plot the section as a line

plt.scatter(section.vertices[:, 0], section.vertices[:, 1],
            c='red', marker='o', label='Points Set 1')
# Plot the two point sets
plt.scatter(findYdistPoints1[:, 0], findYdistPoints1[:, 1],
            c='red', marker='o', label='Points Set 1')
plt.scatter(squarePoints[:, 0], squarePoints[:, 1],
            c='green', marker='x', s=100, label='Points Set 2')
plt.scatter(findXdistPoints[:, 0], findXdistPoints[:, 1],
            c='yellow', marker='x', s=100, label='Points Set 3')

# Formatting
plt.axis('equal')  # Equal aspect ratio
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("2D Section with Points")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()



print('done')