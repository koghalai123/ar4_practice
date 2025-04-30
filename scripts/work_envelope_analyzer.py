#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from work_envelope_interfaces.srv import CheckReachability
import numpy as np
import csv
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup

class WorkEnvelopeAnalyzer(Node):
    def __init__(self):
        super().__init__('work_envelope_analyzer')
        
        # Create client for reachability service
        self.client = self.create_client(
            CheckReachability,
            'check_reachability',
            callback_group=ReentrantCallbackGroup()
        )
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.get_logger().info("Work Envelope Analyzer ready")

    def analyze_work_envelope(self, grid_size=0.1, max_distance=1.0, output_file="work_envelope.csv"):
        # Get current position (you might need to implement this)
        current_position = self.get_current_position()
        if current_position is None:
            self.get_logger().error("Could not get current position")
            return False

        # Calculate grid bounds
        x_range = np.arange(current_position.x - max_distance, 
                          current_position.x + max_distance, 
                          grid_size)
        y_range = np.arange(current_position.y - max_distance, 
                          current_position.y + max_distance, 
                          grid_size)
        z_range = np.arange(current_position.z - max_distance, 
                          current_position.z + max_distance, 
                          grid_size)

        total_points = len(x_range) * len(y_range) * len(z_range)
        self.get_logger().info(f"Analyzing {total_points} points in work envelope...")

        # Prepare CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'z', 'reachable'])

            points_checked = 0
            reachable_count = 0

            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        # Create request
                        request = CheckReachability.Request()
                        request.position = Point(x=float(x), y=float(y), z=float(z))
                        
                        # Call service
                        future = self.client.call_async(request)
                        rclpy.spin_until_future_complete(self, future)
                        
                        if future.result() is not None:
                            result = future.result()
                            writer.writerow([x, y, z, int(result.success)])
                            points_checked += 1
                            if result.success:
                                reachable_count += 1
                            
                            # Print progress
                            if points_checked % 100 == 0:
                                progress = (points_checked / total_points) * 100
                                self.get_logger().info(
                                    f"Progress: {progress:.1f}% - "
                                    f"Points checked: {points_checked}/{total_points} - "
                                    f"Reachable: {reachable_count} ({reachable_count/points_checked:.1%})"
                                )
                        else:
                            self.get_logger().error(f"Service call failed for point ({x}, {y}, {z})")

        self.get_logger().info(f"Analysis complete. Results saved to {output_file}")
        self.get_logger().info(f"Total reachable points: {reachable_count}/{total_points} ({reachable_count/total_points:.1%})")
        return True

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