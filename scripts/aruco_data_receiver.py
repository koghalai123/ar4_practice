#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np

class ArucoDataReceiver(Node):
    def __init__(self):
        super().__init__('aruco_data_receiver')
        
        # Subscribe to all ArUco pose topics
        self.raw_pose_sub = self.create_subscription(
            PoseStamped,
            '/aruco_marker/raw_pose',
            self.raw_pose_callback,
            10)
        
        self.depth_enhanced_sub = self.create_subscription(
            PoseStamped,
            '/aruco_marker/depth_enhanced_pose',
            self.depth_enhanced_callback,
            10)
        
        self.kalman_filtered_sub = self.create_subscription(
            PoseStamped,
            '/aruco_marker/kalman_filtered_pose',
            self.kalman_filtered_callback,
            10)
        
        self.get_logger().info("ArUco data receiver node initialized")
        self.get_logger().info("Listening for poses on:")
        self.get_logger().info("  /aruco_marker/raw_pose")
        self.get_logger().info("  /aruco_marker/depth_enhanced_pose")
        self.get_logger().info("  /aruco_marker/kalman_filtered_pose")

    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles in degrees"""
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        euler = rot.as_euler('xyz', degrees=True)
        return euler

    def raw_pose_callback(self, msg):
        pos = msg.pose.position
        euler = self.quaternion_to_euler(msg.pose.orientation)
        
        print(f"\n=== RAW ArUco Pose ===")
        print(f"Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) meters")
        print(f"Orientation: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
        print(f"Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def depth_enhanced_callback(self, msg):
        pos = msg.pose.position
        euler = self.quaternion_to_euler(msg.pose.orientation)
        
        print(f"\n=== DEPTH-ENHANCED ArUco Pose ===")
        print(f"Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) meters")
        print(f"Orientation: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")

    def kalman_filtered_callback(self, msg):
        pos = msg.pose.position
        euler = self.quaternion_to_euler(msg.pose.orientation)
        
        print(f"\n=== KALMAN-FILTERED ArUco Pose ===")
        print(f"Position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) meters")
        print(f"Orientation: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
        print("-" * 50)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDataReceiver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ArUco data receiver...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
