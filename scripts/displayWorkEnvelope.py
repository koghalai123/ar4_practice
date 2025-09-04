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
from stl import mesh

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import time



def load_multiple_files(input_files):
    dataArray = np.empty((0, 4))
    for i in input_files:
        input_file = os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('ar4_practice'))), 'src','ar4_practice',i)
        try:
            
            data = np.genfromtxt(input_file, delimiter=',', comments='#')
            points = np.column_stack((data[1:,0], data[1:,1],data[1:,2],data[1:,3]))
            dataArray = np.vstack((dataArray, points))
        except:
            print(f"File not found in source directory: {input_file}")
            raise
    mirror = dataArray.copy()
    mirror[:, 0] = -mirror[:, 0]
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

def poisson_reconstruction(points,alpha):
    """Perform Poisson surface reconstruction on a point cloud."""
    reachable = points[labels]
    unreachable = points[~labels]
    alpha_shape = alphashape.alphashape(reachable, alpha)
    vertices = np.array(alpha_shape.vertices)
    faces = np.array(alpha_shape.faces)    
    
    
    
    
    pcd = o3d.geometry.TriangleMesh()
    pcd.vertices = o3d.utility.Vector3dVector(vertices)
    pcd.triangles = o3d.utility.Vector3iVector(faces)
    
    pcd = pcd.sample_points_poisson_disk(5000)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    pcd.estimate_normals()
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    pcd.orient_normals_consistent_tangent_plane(100)
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)



    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
        # After creating your mesh
    #mesh.compute_vertex_normals()  # Important for proper shading
    mesh.compute_vertex_normals()
    #o3d.io.write_triangle_mesh("workEnvelope.stl", mesh)
    print(mesh)
    #save_mesh(mesh)
    o3d.visualization.draw_geometries([mesh])
    # Set color for all vertices (RGB values 0-1)
    '''vertex_colors = np.array([[0.1, 0.5, 0.8]] * len(mesh.vertices))  # Blueish color
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # For visualization with transparency, we need to use the Visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    # Set render options for transparency (approximation)
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True  # Helps with transparency effect
    render_option.light_on = True  # Better shading

    vis.run()
    vis.destroy_window()# 50% transparent'''

#def visualize_poisson_surface(points, labels):
def save_mesh(mesh):
    filename = 'workEnvelope.csv'
    vertices = np.asarray(mesh.vertices)  # Shape (N, 3)
    faces = np.asarray(mesh.triangles)
    with open('vertices.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])  # Header
        writer.writerows(vertices)

    # Save faces to CSV
    with open('faces.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['v1', 'v2', 'v3'])  # Header
        writer.writerows(faces)

    print("Saved vertices.csv and faces.csv")
    
    
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
        color='lightblue', edgecolor='none', alpha=0.5
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
    vertices = np.array(alpha_shape.vertices)
    faces = np.array(alpha_shape.faces)

    # Apply smoothing
    smoothed_mesh = smooth_with_open3d(vertices, faces, iterations=50)

    # Get smoothed vertices and faces
    smoothed_vertices = np.asarray(smoothed_mesh.vertices)
    smoothed_faces = np.asarray(smoothed_mesh.triangles)

    # Plot the results
    plot_smoothed_mesh(smoothed_vertices, smoothed_faces)
    
    return vertices, faces

class WorkEnvelopePublisher(Node):
    def __init__(self):
        super().__init__('work_envelope_publisher')
        self.marker_pub = self.create_publisher(Marker, '/work_envelope_mesh', 10)
        self.points_pub = self.create_publisher(MarkerArray, '/work_envelope_points', 10)
        # Remove the timer - we'll publish manually
        
        # Storage for mesh data
        self.mesh_vertices = None
        self.mesh_faces = None
        self.reachable_points = None
        self.unreachable_points = None
        
        self.get_logger().info("Work Envelope Publisher initialized")

    def set_mesh_data(self, vertices, faces):
        """Set the mesh data for publishing"""
        self.mesh_vertices = vertices
        self.mesh_faces = faces

    def publish_markers(self):
        """Publish mesh and point markers to RViz2"""
        # Publish mesh as triangle list
        if self.mesh_vertices is not None and self.mesh_faces is not None:
            self.publish_mesh_marker()
        

    def publish_mesh_marker(self):
        """Publish the work envelope mesh as a triangle list marker"""
        marker = Marker()
        marker.header.frame_id = "base_link"  # Change to your robot's base frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "work_envelope"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Set pose (identity)
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Set color (semi-transparent blue)
        marker.color.r = 0.1
        marker.color.g = 0.5
        marker.color.b = 0.8
        marker.color.a = 0.5  # 30% transparent
        
        # Add triangle vertices
        triangle_count = 0
        for face in self.mesh_faces:
            if len(face) == 3:  # Ensure it's a triangle
                # Add the 3 vertices of this triangle
                for vertex_idx in face:
                    if vertex_idx < len(self.mesh_vertices):  # Bounds check
                        point = Point()
                        point.x = float(self.mesh_vertices[vertex_idx][0])
                        point.y = float(self.mesh_vertices[vertex_idx][1])
                        point.z = float(self.mesh_vertices[vertex_idx][2])
                        marker.points.append(point)
                    else:
                        self.get_logger().warn(f"Invalid vertex index: {vertex_idx}")
                        break
                else:
                    # This executes only if the loop completed without break
                    triangle_count += 1
            else:
                self.get_logger().warn(f"Non-triangle face with {len(face)} vertices: {face}")

        
        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published mesh with {len(self.mesh_faces)} triangles")

    
def broadcast_to_rviz(points, labels, mesh_vertices=None, mesh_faces=None):
    """
    Broadcast work envelope data to RViz2
    
    Args:
        points: All points (nx3 array)
        labels: Boolean array indicating reachable points
        mesh_vertices: Mesh vertices (optional)
        mesh_faces: Mesh faces (optional)
    """
    def run_publisher():
        rclpy.init()
        publisher = WorkEnvelopePublisher()
        
        if mesh_vertices is not None and mesh_faces is not None:
            publisher.set_mesh_data(mesh_vertices, mesh_faces)
        try:
            rclpy.spin(publisher)
        except KeyboardInterrupt:
            pass
        finally:
            publisher.destroy_node()
            rclpy.shutdown()
    
    run_publisher()
    
    print("Work envelope data is being broadcast to RViz2")
    print("Topics available:")
    print("  - /work_envelope_mesh (Marker)")
    print("  - /work_envelope_points (MarkerArray)")
    print("Add these to RViz2 to visualize the work envelope")
    

def visualize_alpha_shape_with_rviz(points, labels, alpha=0.5):
    vertices, faces = visualize_alpha_shape(points, labels, alpha=0.5)

    # Initialize ROS and broadcast to RViz2
    rclpy.init()
    publisher = WorkEnvelopePublisher()
    
    publisher.set_mesh_data(vertices, faces)
    
    # Publish once
    publisher.publish_markers()
    
    print("Alpha Shape work envelope broadcasted to RViz2")
    print("Topics available:")
    print("  - /work_envelope_mesh (Marker) - Alpha Shape Surface")
    print("  - /work_envelope_points (MarkerArray) - Point Cloud")
    
    # Keep publishing for a few seconds to ensure RViz2 receives it
    for i in range(100000):
        publisher.publish_markers()
        time.sleep(1)
        rclpy.spin_once(publisher, timeout_sec=0.1)
    
    # Clean up
    publisher.destroy_node()
    rclpy.shutdown()

def display_work_envelope(pathToFaces, pathToVertices):
    
    # Load vertices from CSV
    try:
        vertices_data = np.genfromtxt(pathToVertices, delimiter=',', skip_header=1)
        vertices = vertices_data[:, :3]  # Take x, y, z columns
        print(f"Loaded {len(vertices)} vertices from {pathToVertices}")
    except Exception as e:
        print(f"Error loading vertices from {pathToVertices}: {e}")
        return
    
    # Load faces from CSV
    try:
        faces_data = np.genfromtxt(pathToFaces, delimiter=',', skip_header=1, dtype=int)
        faces = faces_data[:, :3]  # Take v1, v2, v3 columns
        print(f"Loaded {len(faces)} faces from {pathToFaces}")
    except Exception as e:
        print(f"Error loading faces from {pathToFaces}: {e}")
        return
    # Initialize ROS and broadcast to RViz2
    rclpy.init()
    publisher = WorkEnvelopePublisher()
    
    publisher.set_mesh_data(vertices, faces)
    
    # Publish once
    publisher.publish_markers()
    
    print("Alpha Shape work envelope broadcasted to RViz2")
    print("Topics available:")
    print("  - /work_envelope_mesh (Marker) - Alpha Shape Surface")
    print("  - /work_envelope_points (MarkerArray) - Point Cloud")
    
    # Keep publishing for a few seconds to ensure RViz2 receives it
    for i in range(100000):
        publisher.publish_markers()
        time.sleep(1)
        rclpy.spin_once(publisher, timeout_sec=0.1)
    
    # Clean up
    publisher.destroy_node()
    rclpy.shutdown()
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
    
    '''print(f"Processing files: {input_files}")
    points, labels = load_multiple_files(input_files)
    print(f"Total points loaded: {len(points)} (Reachable: {sum(labels)}, Unreachable: {sum(labels==0)})")
    
    visualize_alpha_shape(points, labels,alpha=8.1)'''
    
    #visualize_convex_surface(points, labels)
    #mesh, vertices, faces = visualize_alpha_shape_with_rviz(points, labels, alpha=8.1)
    display_work_envelope(pathToFaces="workEnvelopeFaces.csv", pathToVertices="workEnvelopeVertices.csv")
    #visualize_bpa_mesh(points, labels, radii=[0.05, 0.1])
    #poisson_reconstruction(points,alpha=10.1)
    #print('done')
    #visualize_poisson_surface(points, labels)
    #mesh, broadcast_thread = poisson_reconstruction_with_rviz(points, labels, alpha=10.1)


