import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import glob
import os 
from ament_index_python.packages import get_package_prefix
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from trimesh import Trimesh
from trimesh.voxel import creation
import open3d as o3d

def load_multiple_files(input_files):
    dataArray = np.empty((0, 4))
    for i in input_files:
        input_file = os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('ar4_practice'))),'src','ar4_practice', i)
        try:
            if os.path.isfile(input_file):
                data = np.genfromtxt(input_file, delimiter=',', comments='#')
                points = np.column_stack((data[1:,0], data[1:,1],data[1:,2],data[1:,3]))
                dataArray = np.vstack((dataArray, points))
        except:
            print(f"File not found in source directory: {input_file}")
            raise
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
    
def tetrahedron_circumradius(a, b, c, d):
    """Compute the circumradius of a tetrahedron defined by points a, b, c, d."""
    # Build the 4x4 matrix for Cayley-Menger determinant
    def _square_dist(p1, p2):
        return np.sum((p1 - p2) ** 2)
    
    D = np.ones((5, 5))
    D[0, 0] = 0
    D[1, 1] = _square_dist(a, a)
    D[1, 2] = _square_dist(a, b)
    D[1, 3] = _square_dist(a, c)
    D[1, 4] = _square_dist(a, d)
    D[2, 1] = D[1, 2]
    D[2, 2] = _square_dist(b, b)
    D[2, 3] = _square_dist(b, c)
    D[2, 4] = _square_dist(b, d)
    D[3, 1] = D[1, 3]
    D[3, 2] = D[2, 3]
    D[3, 3] = _square_dist(c, c)
    D[3, 4] = _square_dist(c, d)
    D[4, 1] = D[1, 4]
    D[4, 2] = D[2, 4]
    D[4, 3] = D[3, 4]
    D[4, 4] = _square_dist(d, d)
    
    # Compute determinants
    A = np.linalg.det(D[:4, :4])
    B = np.linalg.det(D[:, :])
    
    # Circumradius formula
    if np.abs(A) < 1e-12:
        return np.inf  # Degenerate tetrahedron
    R = np.sqrt(B / (2 * A))
    return R

def alpha_shape_3d(points, alpha):
    """Compute the alpha shape (concave hull) preserving gaps."""
    tetra = Delaunay(points)
    surface_triangles = set()
    
    for simplex in tetra.simplices:
        a, b, c, d = points[simplex]
        R = tetrahedron_circumradius(a, b, c, d)
        
        if R < alpha:
            # Add all 4 triangular faces of the tetrahedron
            faces = [
                tuple(sorted((simplex[0], simplex[1], simplex[2]))),
                tuple(sorted((simplex[0], simplex[1], simplex[3]))),
                tuple(sorted((simplex[0], simplex[2], simplex[3]))),
                tuple(sorted((simplex[1], simplex[2], simplex[3])))
            ]
            surface_triangles.update(faces)
    
    return np.array(list(surface_triangles))

def visualize_alpha_shape(points, labels, alpha=0.5):
    reachable = points[labels]
    unreachable = points[~labels]
    
    triangles = alpha_shape_3d(reachable, alpha)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    mesh = Poly3DCollection(reachable[triangles], alpha=0.3, edgecolor='k')
    mesh.set_facecolor('cyan')
    ax.add_collection3d(mesh)
    
    # Scatter points
    ax.scatter(reachable[:, 0], reachable[:, 1], reachable[:, 2], 
               c='b', label='Reachable')

    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()




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
    ]
    
    print(f"Processing files: {input_files}")
    points, labels = load_multiple_files(input_files)
    print(f"Total points loaded: {len(points)} (Reachable: {sum(labels)}, Unreachable: {sum(labels==0)})")
    visualize_convex_surface(points, labels)
    #visualize_alpha_shape(points, labels,alpha=4)
    #visualize_bpa_mesh(points, labels, radii=[0.05, 0.1])

    #visualize_poisson_surface(points, labels)