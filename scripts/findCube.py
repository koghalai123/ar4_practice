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
vertices_path = os.path.join(package_dir, 'vertices_correct.csv')
faces_path = os.path.join(package_dir, 'faces_correct.csv')

vertices = load_vertices(vertices_path)
faces = load_faces(faces_path)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
plane_normal = [0, 0, 1]
plane_point = [0, 0, 0]

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

yRange = 1*np.sum(np.abs(firstIntersection-secondIntersection))
#squarePoints = secondIntersection+np.sign(firstIntersection-secondIntersection).sum()*np.array([[0,0,0],[0,0,0],[0,yRange,0],[0,yRange,0]])+np.array([[-yRange/2,0,0],[yRange/2,0,0],[-yRange/2,0,0],[yRange/2,0,0]])


secondRayPoint = np.array([
    searchOrigin,
])
secondRayDirection = np.array([[1,0,0]])
secondLocations = intersector.intersects_location(
    secondRayPoint, 
    secondRayDirection,multiple_hits = False)[0]

#Check the y location of the edge of the cube closest to the robot
scanYLoc = np.array([secondIntersection-np.array([0,0.05,0])])
highLim = np.array([firstIntersection])
lowLim = np.array([secondIntersection])
for i in range(100):
    rayPoint = scanYLoc
    rayDirection = np.array([[-1,0,0]])
    intersectLocation = intersector.intersects_location(rayPoint, rayDirection,multiple_hits = False)[0]
    intersectDistance = np.linalg.norm(intersectLocation-rayPoint)
    if intersectDistance > yRange:
        highLim = np.array([[secondIntersection[0],intersectLocation[0,1],secondIntersection[2]]])
        scanYLoc = (secondIntersection+lowLim)/2
    else:
        lowLim = np.array([[secondIntersection[0],intersectLocation[0,1],secondIntersection[2]]])
        scanYLoc = rayPoint = (secondIntersection+highLim)/2
    #print('iteration')
    #print('yRangeTemp',yRangeTemp)
y_edge = intersectLocation[0,1]
#cube_y_edge = 

#Check the limit on the y range, starting from the cube edge
yRangeTemp = yRange
prevyRangeTemp = yRangeTemp
counter = 0
highLim = yRangeTemp
lowLim = 0
for i in range(100):
    rayPoint1 = np.array([[yRangeTemp/2,y_edge,firstIntersection[2]]])
    rayPoint2 = np.array([[yRangeTemp/2,y_edge-yRangeTemp,firstIntersection[2]]])
    rayDirection1 = np.array([[0,-1,0]])
    rayDirection2 = np.array([[0,0,1]])
    intersectLocation1 = intersector.intersects_location(rayPoint1, rayDirection1,multiple_hits = False)[0]
    intersectLocation2 = intersector.intersects_location(rayPoint1, rayDirection2,multiple_hits = False)[0]
    intersectLocation3 = intersector.intersects_location(rayPoint2, rayDirection2,multiple_hits = False)[0]
    intersectDistance1 = np.linalg.norm(intersectLocation1-rayPoint1)
    intersectDistance2 = np.linalg.norm(intersectLocation2-rayPoint1)
    intersectDistance3 = np.linalg.norm(intersectLocation3-rayPoint2)
    yRangeTemp = 0.7*yRangeTemp+0.3*np.min([intersectDistance1,intersectDistance2,intersectDistance3])
    percentDiff = np.abs(prevyRangeTemp-yRangeTemp)/prevyRangeTemp
    if  percentDiff < 0.005:
        print('percentDiff',percentDiff)
        counter += 1
    else:
        counter = 0
    if counter > 5:
        break
    prevyRangeTemp = yRangeTemp
    
    print('yRangeTemp',yRangeTemp)

cubeCorner = np.array([[-yRangeTemp/2,y_edge,firstIntersection[2]]])



x = np.linspace(0, yRangeTemp, 2)
y = np.linspace(0, -yRangeTemp, 2)
z = np.linspace(0, yRangeTemp, 2)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

points_3d = np.column_stack((xx, yy, zz)) + cubeCorner

plane_normal1 = plane_normal
plane_point1 = plane_point
plane_normal2 = [1, 0, 0]
plane_point2 = cubeCorner.flatten()
section1 = mesh.section(plane_origin=plane_point1, plane_normal=plane_normal1)
section2 = mesh.section(plane_origin=plane_point2, plane_normal=plane_normal2)

faces = np.array([
    [0, 2, 1], [1, 2, 3],
    [4, 5, 6], [5, 7, 6],
    [0, 1, 4], [1, 5, 4],
    [2, 6, 3], [3, 6, 7],
    [0, 4, 2], [2, 4, 6],
    [1, 3, 5], [3, 7, 5]
])

# Create the mesh object
cube = trimesh.Trimesh(vertices=points_3d, faces=faces)
mesh.visual.face_colors = [100, 100, 100, 128]

scene = trimesh.Scene([section1, section2,cube,mesh])
scene.show()



#File paths for saving vertices and faces
vertices_csv_path = "workEnvelopeCubeVertices.csv"
faces_csv_path = "workEnvelopeCubeFaces.csv"

# Save vertices to a CSV file
with open(vertices_csv_path, mode='w', newline='') as vertices_file:
    writer = csv.writer(vertices_file)
    writer.writerow(["x", "y", "z"])  # Header
    writer.writerows(cube.vertices)  # Write vertices

# Save faces to a CSV file
with open(faces_csv_path, mode='w', newline='') as faces_file:
    writer = csv.writer(faces_file)
    writer.writerow(["v1", "v2", "v3"])  # Header
    writer.writerows(cube.faces)  # Write faces

print(f"Vertices saved to {vertices_csv_path}")
print(f"Faces saved to {faces_csv_path}")



"""
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

vertices1 = section1.vertices
vertices2 = section2.vertices

ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='black', label='Section 1')

# Plot points from section1
ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], c='red', label='Section 1')

# Plot points from section2
ax.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], c='blue', label='Section 2')

# Formatting
ax.set_title("3D Scatter Plot of Section Points")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.legend()
plt.show()"""







"""
plt.figure(figsize=(8, 6))

# Plot the section as a line

plt.scatter(section.vertices[:, 0], section.vertices[:, 1],
            c='red', marker='o', label='Points Set 1')

plt.scatter(squarePoints[:, 0], squarePoints[:, 1],
            c='green', marker='x', s=100, label='Points Set 2')

# Formatting
plt.axis('equal')  # Equal aspect ratio
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("2D Section with Points")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()"""



print('done')