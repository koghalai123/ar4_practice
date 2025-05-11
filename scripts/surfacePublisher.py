#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import csv
import os

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
        marker.color.r = 0.3
        marker.color.g = 0.3
        marker.color.b = 0.6
        marker.color.a = 0.3  # Semi-transparent
        
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
        marker.color.r = 0.9
        marker.color.g = 0.9
        marker.color.b = 0.9
        marker.color.a = 0.9  # Semi-transparent
        
        # Scale (irrelevant for TRIANGLE_LIST, but required)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.workEnvelopeCube = marker

def main(args=None):
    rclpy.init(args=args)
    surface_publisher = SurfacePublisher()
    surface_publisher.createMarkerEnvelope()
    surface_publisher.createMarkerCube()
    rclpy.spin(surface_publisher)
    surface_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()