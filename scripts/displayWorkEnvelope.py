import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import glob
import os 
import sys

from ament_index_python.packages import get_package_prefix
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import trimesh
from trimesh.voxel import creation
import open3d as o3d
import alphashape
from descartes import PolygonPatch



def load_multiple_files(input_files):
    dataArray = np.empty((0, 4))
    for i in input_files:
        input_file = os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('ar4_practice'))), i)
        try:
            
            data = np.genfromtxt(input_file, delimiter=',', comments='#')
            points = np.column_stack((data[1:,0], data[1:,1],data[1:,2],data[1:,3]))
            dataArray = np.vstack((dataArray, points))
        except:
            print(f"File not found in source directory: {input_file}")
            raise
    mirror = dataArray
    mirror[:, 0] *= -1
    dataArray = np.vstack((dataArray, mirror))
    return dataArray[:,0:3], dataArray[:,3]==1
                


def visualize_convex_surface(points, labels):
    # Separate reachable and unreachable points
    reachable = points[labels]
    unreachable = points[~labels]
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(reachable[:, 0], reachable[:, 1], reachable[:, 2], 
               c='b', marker='o', label='Reachable')
    #ax.scatter(unreachable[:, 0], unreachable[:, 1], unreachable[:, 2], c='r', marker='^', label='Unreachable')
    
    # Compute and plot the convex hull
    hull = ConvexHull(reachable)
    
    # Draw the convex hull as a transparent surface
    vertices = [reachable[simplex] for simplex in hull.simplices]
    hull_surface = Poly3DCollection(vertices, alpha=0.25, edgecolor='k', linewidths=1)
    hull_surface.set_facecolor('cyan')
    ax.add_collection3d(hull_surface)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.set_title('Convex Surface Separating Reachable and Unreachable Spaces\n(Combined Data from Multiple Files)', 
                fontsize=14, pad=20)
    ax.legend(fontsize=10, markerscale=1.5)
    
    # Adjust viewing angle for better visibility
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show()

def poisson_reconstruction(points):
    """Perform Poisson surface reconstruction on a point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals (required for Poisson)
    pcd.estimate_normals()
    
    # Run Poisson reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    return mesh

def visualize_poisson_surface(points, labels):
    """Visualize the Poisson-reconstructed surface."""
    reachable = points[labels]
    unreachable = points[~labels]
    
    # Compute Poisson mesh
    mesh = poisson_reconstruction(reachable)
    
    # Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Poisson surface (semi-transparent)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        color='cyan', alpha=0.3, edgecolor='k', linewidth=0.5
    )
    
    # Scatter original points
    ax.scatter(reachable[:, 0], reachable[:, 1], reachable[:, 2], 
               c='blue', marker='o', label='Reachable', s=20)

    
    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Poisson-Reconstructed Surface', fontsize=14)
    ax.legend()
    
    # Adjust view
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()
    
def smooth_with_open3d(vertices, faces, iterations=50):
    """Smooth mesh using Open3D's Taubin smoothing"""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh.filter_smooth_taubin(number_of_iterations=iterations)

def plot_smoothed_mesh(vertices, faces):
    """Plot the smoothed mesh in 3D using matplotlib"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces,
        color='lightblue', edgecolor='none', alpha=0.8
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Smoothed Alpha Shape')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Requires matplotlib 3.3+
    
    plt.tight_layout()
    plt.show()

# Apply smoothing


def visualize_alpha_shape(points, labels, alpha=0.5):
    reachable = points[labels]
    unreachable = points[~labels]    
    
    #ax.scatter(df_3d['x'], df_3d['y'], df_3d['z'])
    

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.scatter(df_3d['x'], df_3d['y'], df_3d['z'])
    #plt.show()
    alpha_shape = alphashape.alphashape(reachable, alpha)
    """smoothed_vertices, smoothed_faces = smooth_with_open3d(
    np.array(alpha_shape.vertices),
    np.array(alpha_shape.faces))
    #alpha_shape.show()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    print(1)"""
    vertices = np.array(alpha_shape.vertices)
    faces = np.array(alpha_shape.faces)

    # Apply smoothing
    smoothed_mesh = smooth_with_open3d(vertices, faces, iterations=50)

    # Get smoothed vertices and faces
    smoothed_vertices = np.asarray(smoothed_mesh.vertices)
    smoothed_faces = np.asarray(smoothed_mesh.triangles)

    # Plot the results
    plot_smoothed_mesh(smoothed_vertices, smoothed_faces)


def ball_pivoting_reconstruction(points, radii=[0.05, 0.1, 0.2]):
    """
    Reconstruct a surface using the Ball-Pivoting Algorithm.
    - `radii`: List of ball radii to try (smaller = tighter fits, larger = bridges gaps).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals (critical for BPA)
    pcd.estimate_normals()
    
    # Run BPA
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh

def visualize_bpa_mesh(points, labels, radii=[0.05, 0.1, 0.2]):
    """Visualize the BPA-reconstructed mesh with gaps preserved."""
    reachable = points[labels]
    unreachable = points[~labels]
    
    # Reconstruct mesh
    mesh = ball_pivoting_reconstruction(reachable, radii)
    
    # Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh (semi-transparent)
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=triangles,
        color='cyan', alpha=0.3, edgecolor='k', linewidth=0.5
    )
    
    # Scatter points
    ax.scatter(reachable[:, 0], reachable[:, 1], reachable[:, 2],
               c='blue', label='Reachable', s=20)
    #ax.scatter(unreachable[:, 0], unreachable[:, 1], unreachable[:, 2],c='red', label='Unreachable', s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    
    
if __name__ == "__main__":
    import sys
    

    
    input_files = [
        'results0_2.csv',
        'results3.csv',
        'results4.csv',
        'results5.csv',
        'results6.csv',
        'results7.csv',
        'results8.csv',
        'results9.csv',
        'results10.csv',
        'results11.csv',
        'results12.csv',
        'results13.csv',
    ]
    
    print(f"Processing files: {input_files}")
    points, labels = load_multiple_files(input_files)
    print(f"Total points loaded: {len(points)} (Reachable: {sum(labels)}, Unreachable: {sum(labels==0)})")
    #visualize_convex_surface(points, labels)
    visualize_alpha_shape(points, labels,alpha=8.1)
    #visualize_bpa_mesh(points, labels, radii=[0.05, 0.1])

    #visualize_poisson_surface(points, labels)