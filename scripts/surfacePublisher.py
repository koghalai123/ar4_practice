#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import csv
import os
import numpy as np
class SurfacePublisher(Node):
    def __init__(self):
        super().__init__('surface_publisher')
        self.marker_pub = self.create_publisher(MarkerArray, 'surface_marker', 10)

        
        
        timer_period = 2.0  # seconds - no need to publish too frequently
        self.timer = self.create_timer(timer_period, self.publish_marker)
        
        self.get_logger().info("Publishing surface marker...")

    def load_vertices(self, filepath):
        """Load vertices from CSV file"""
        vertices = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Assuming format: x,y,z
                if len(row) >= 3:  # Ensure there are at least 3 values
                    try:
                        x, y, z = map(float, row[:3])
                        vertices.append(Point(x=x, y=y, z=z))
                    except ValueError:
                        self.get_logger().warn(f"Skipping invalid vertex row: {row}")
        return vertices

    def load_faces(self, filepath):
        """Load faces from CSV file"""
        faces = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Assuming each row contains vertex indices for a face
                try:
                    face = list(map(int, row))
                    faces.extend(face)  # Flatten into a single list
                except ValueError:
                    self.get_logger().warn(f"Skipping invalid face row: {row}")
        return faces

    def publish_marker(self):
        
        marker_array = MarkerArray()
        marker_array.markers.append(self.workEnvelopeCube)
        marker_array.markers.append(self.workEnvelope)
        marker_array.markers.append(self.workPlane)
        marker_array.markers.append(self.sphereList)
        self.marker_pub.publish(marker_array)
        #self.marker_pub.publish(self.workEnvelope)

    def createMarkerEnvelope(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        vertices_path = os.path.join(package_dir, 'workEnvelopeVertices.csv')
        faces_path = os.path.join(package_dir, 'workEnvelopeFaces.csv')
        
        # Load vertices and faces
        self.vertices = self.load_vertices(vertices_path)
        self.faces = self.load_faces(faces_path)
        
        marker = Marker()
        marker.header.frame_id = "base_link"  # Change to your frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "surface"
        marker.id = 0
        
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Set pose (identity means the marker will appear exactly as specified in points)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Add vertices according to faces
        for i in range(0, len(self.faces), 3):
            if i+2 >= len(self.faces):
                break  # Skip incomplete triangles
            try:
                marker.points.append(self.vertices[self.faces[i]])
                marker.points.append(self.vertices[self.faces[i+1]])
                marker.points.append(self.vertices[self.faces[i+2]])
            except IndexError:
                self.get_logger().warn(f"Invalid vertex index in face at position {i}")
        
        # Set color (RGBA, 0-1)
        marker.color.r = 0.6
        marker.color.g = 0.3
        marker.color.b = 0.3
        marker.color.a = 0.15  # Semi-transparent
        
        # Scale (irrelevant for TRIANGLE_LIST, but required)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.workEnvelope = marker
    def createMarkerCube(self):

        # Get path to CSV files (assuming they're in the same directory as this script)
        package_dir = os.path.dirname(os.path.abspath(__file__))
        vertices_path = os.path.join(package_dir, 'workEnvelopeCubeVertices.csv')
        faces_path = os.path.join(package_dir, 'workEnvelopeCubeFaces.csv')
        
        # Load vertices and faces
        self.vertices = self.load_vertices(vertices_path)
        self.faces = self.load_faces(faces_path)


        marker = Marker()
        marker.header.frame_id = "base_link"  # Change to your frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cube"
        marker.id = 0
        
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Set pose (identity means the marker will appear exactly as specified in points)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Add vertices according to faces
        for i in range(0, len(self.faces), 3):
            if i+2 >= len(self.faces):
                break  # Skip incomplete triangles
            try:
                marker.points.append(self.vertices[self.faces[i]])
                marker.points.append(self.vertices[self.faces[i+1]])
                marker.points.append(self.vertices[self.faces[i+2]])
            except IndexError:
                self.get_logger().warn(f"Invalid vertex index in face at position {i}")
        
        # Set color (RGBA, 0-1)
        marker.color.r = 0.4
        marker.color.g = 0.8
        marker.color.b = 0.4
        marker.color.a = 0.2  # Semi-transparent
        
        # Scale (irrelevant for TRIANGLE_LIST, but required)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.workEnvelopeCube = marker
    def createMarkerWorkPlane(self):

        # Get path to CSV files (assuming they're in the same directory as this script)
        package_dir = os.path.dirname(os.path.abspath(__file__))
        vertices_path = os.path.join(package_dir, 'workEnvelopeCubeVertices.csv')
        faces_path = os.path.join(package_dir, 'workEnvelopeCubeFaces.csv')
        
        # Load vertices and faces
        self.vertices = self.load_vertices(vertices_path)
        self.faces = self.load_faces(faces_path)


        marker = Marker()
        marker.header.frame_id = "base_link"  # Change to your frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "plane"
        marker.id = 0
        
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Set pose (identity means the marker will appear exactly as specified in points)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        pointArray = np.array([[point.x, point.y, point.z] for point in self.vertices])
        cubeCenter = [0.0, 0.0, 0.0]
        for i in range(len(self.vertices)):
            cubeCenter[0] += self.vertices[i].x/float(len(self.vertices))
            cubeCenter[1] += self.vertices[i].y/float(len(self.vertices))
            cubeCenter[2] += self.vertices[i].z/float(len(self.vertices))

        cubeSideLength = np.abs(np.diff(np.unique(pointArray[:,2])))
        
        cubeCenter = np.array(cubeCenter).flatten()

        planePoints = 0.8*np.array([
            [-cubeSideLength[0]/2, -cubeSideLength[0]/2, cubeSideLength[0]/2],
            [cubeSideLength[0]/2, -cubeSideLength[0]/2, cubeSideLength[0]/2],
            [-cubeSideLength[0]/2, cubeSideLength[0]/2, -cubeSideLength[0]/2],
            [cubeSideLength[0]/2, cubeSideLength[0]/2, -cubeSideLength[0]/2],
            [0,0,0],
        ]) + cubeCenter
        planeFaces  = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        planePoints_as_points = [Point(x=p[0], y=p[1], z=p[2]) for p in planePoints]
        for i in range(0, len(planeFaces)):
            if i+2 >= len(self.faces):
                break  # Skip incomplete triangles
            marker.points.append(planePoints_as_points[planeFaces[i,0]])
            marker.points.append(planePoints_as_points[planeFaces[i,1]])
            marker.points.append(planePoints_as_points[planeFaces[i,2]])

        
        
        # Set color (RGBA, 0-1)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Semi-transparent
        
        # Scale (irrelevant for TRIANGLE_LIST, but required)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.workPlane = marker


        # 3. Create SPHERE_LIST marker
        sphere_list_marker = Marker()
        sphere_list_marker.header.frame_id = "base_link"
        sphere_list_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_list_marker.ns = "work_envelope"  # Same namespace as others
        sphere_list_marker.id = 2  # Unique ID
        sphere_list_marker.type = Marker.SPHERE_LIST
        sphere_list_marker.action = Marker.ADD

        # Set appearance
        sphere_list_marker.scale.x = 0.03  # Diameter of each sphere
        sphere_list_marker.scale.y = 0.03  # Must be same as x for spheres
        sphere_list_marker.scale.z = 0.03  # Must be same as x for spheres
        sphere_list_marker.color.a = 1.0  # Fully opaque
        sphere_list_marker.color.r = 1.0  # Red
        sphere_list_marker.color.g = 0.0
        sphere_list_marker.color.b = 0.0


        for point in planePoints_as_points:
            sphere_list_marker.points.append(point)
        self.sphereList = sphere_list_marker


        np.savetxt("workPlanePoints.csv", planePoints, delimiter=",", header="x,y,z", comments='', fmt="%.6f")

        print("Array saved to output.csv")







def main(args=None):
    rclpy.init(args=args)
    surface_publisher = SurfacePublisher()
    surface_publisher.createMarkerEnvelope()
    surface_publisher.createMarkerCube()
    surface_publisher.createMarkerWorkPlane()
    rclpy.spin(surface_publisher)
    surface_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()