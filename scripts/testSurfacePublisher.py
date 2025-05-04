#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class SurfacePublisher(Node):
    def __init__(self):
        super().__init__('surface_publisher')
        self.marker_pub = self.create_publisher(Marker, 'surface_marker', 10)
        
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_marker)
        
        self.get_logger().info("Publishing surface marker...")

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"  # Match your robot's frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "surface"
        marker.id = 0
        
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 1.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)
        '''marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        # Define points and faces (example: a simple tetrahedron)
        points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=1.0, y=0.0, z=0.0),
            Point(x=0.0, y=1.0, z=0.0),
            Point(x=0.0, y=0.0, z=1.0)
        ]
        faces = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]  # Each triplet is a triangle

        # Add points to the marker
        for i in range(0, len(faces), 3):
            marker.points.append(points[faces[i]])
            marker.points.append(points[faces[i+1]])
            marker.points.append(points[faces[i+2]])

        # Set color (RGBA, 0-1)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5  # Semi-transparent

        # Scale (irrelevant for TRIANGLE_LIST, but required)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0'''

        

def main(args=None):
    rclpy.init(args=args)
    surface_publisher = SurfacePublisher()
    rclpy.spin(surface_publisher)
    surface_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()