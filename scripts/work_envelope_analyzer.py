#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion  # Import for Euler/Quaternion conversion

#from work_envelope_interfaces.srv import CheckReachability
import numpy as np
import csv
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup

class WorkEnvelopeAnalyzer(Node):
    def __init__(self):
        super().__init__('work_envelope_analyzer')
        
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.analyze_work_envelope)
        
        # Create client for reachability service
        self.publisher = self.create_publisher(Pose,
                                                     "/check_reachability",10)
        
        
        
        self.get_logger().info("Work Envelope Analyzer ready")

    def analyze_work_envelope(self, grid_size=0.1, max_distance=1.0, output_file="work_envelope.csv"):
        values = [1.0,2.0,3.0]
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Replace with actual values if needed
        position = Point(x=values[0], y=values[1], z=values[2])
        quat = quaternion_from_euler(roll, pitch, yaw)
        orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        pose=Pose(position=position, orientation=orientation)
        self.publisher.publish(pose)
        self.get_logger().info('Publishing: ')

    def get_current_position(self):
        # Implement this method to get current end effector position
        # You might use MoveIt2's compute_fk or other methods
        return Point(x=0.0, y=0.0, z=0.0)  # Placeholder

def main(args=None):
    rclpy.init(args=args)
    
    analyzer = WorkEnvelopeAnalyzer()
    
    # Start analysis in a separate thread to avoid blocking
    analysis_thread = Thread(target=analyzer.analyze_work_envelope, 
                           kwargs={'grid_size': 0.1, 'max_distance': 0.5})
    analysis_thread.start()
    
    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        analyzer.get_logger().info("Shutting down...")
    
    analysis_thread.join()
if __name__ == "__main__":
    main()