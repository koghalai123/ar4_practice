#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion  # Import for Euler/Quaternion conversion

#from work_envelope_interfaces.srv import CheckReachability
import numpy as np
import csv
from threading import Thread
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Bool  # Add this import at the top
from pymoveit2 import MoveIt2
import os 
from ament_index_python.packages import get_package_prefix
import csv


class WorkEnvelopeAnalyzer(Node):
    def __init__(self):
        super().__init__('moveit_commander')
        self.callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=self.callback_group
        )
        
        timer_period = 0.5  # seconds
        self.counter = 0
        #self.timer = self.create_timer(timer_period, self.analyze_work_envelope,callback_group=self.callback_group)
        
        # Create client for reachability service
        self.publisher = self.create_publisher(Pose,"/check_reachability",10,callback_group=self.callback_group)
        
        #self.save_position = self.create_subscription(PoseStamped, "/get_current_pose",self.save_current_position,1,callback_group=self.callback_group)
        
        self.get_logger().info("Work Envelope Analyzer ready")

    def analyze_work_envelope(self, position, orientation):
#await asyncio.wait_for(self.moveit2.compute_ik_async(position=pose.position,quat_xyzw=pose.orientation),timeout=0.5)
        ik_result = self.moveit2.compute_ik(position=position,quat_xyzw=orientation)
        if ik_result is None:
            return False
        else:
            return True

    def generate_grid(self):
        current_pose = self.moveit2.compute_fk()
        self.current_position = current_pose
        positionObj = self.current_position.pose.position
        position = np.array([positionObj.x, positionObj.y, positionObj.z])
        # Extract the current pose from the message
        if self.grid_file is not None:
            input_file = os.path.join(os.path.dirname(os.path.dirname(get_package_prefix('ar4_practice'))),'src','ar4_practice', self.grid_file)
            if os.path.isfile(input_file):
                data = np.genfromtxt(input_file, delimiter=',', comments='#')
                points = np.column_stack((data[1:,0], data[1:,1],data[1:,2]))
                self.xx = data[1:,0]
                self.yy = data[1:,1]
                self.zz = data[1:,2]
            else:
                self.get_logger().error(f"File not found in source directory: {input_file}")
        else:
            
            # Generate a grid of points around the current pose
            grid_length = 0.75
            grid_resolution = 0.1
            num_samples = int(np.ceil(grid_length * 2 / grid_resolution))+1
            x = np.linspace(position[0]-grid_length, position[0]+grid_length, num_samples)
            y = np.linspace(position[0]-grid_length, position[0]+grid_length, num_samples)
            z = np.linspace(position[0]-grid_length, position[0]+grid_length, num_samples)
            
            # Create 3D grid using meshgrid
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            self.xx = xx.flatten()
            self.yy = yy.flatten()
            self.zz = zz.flatten()
    #def save_results(self, msg):
    def save_current_position(self,msg):
        self.current_position = msg
        if self.counter == 0:
            self.generate_grid()




def main(args=None):
    rclpy.init(args=args)
    
    analyzer = WorkEnvelopeAnalyzer()
    analyzer.grid_file = None
    analyzer.grid_file = 'boundary_samples3.csv'
    analyzer.generate_grid()
    from geometry_msgs.msg import Point, Quaternion

    position = Point(x=0.0, y=0.0, z=0.0)
    orientation = Quaternion()
    orientation = analyzer.current_position.pose.orientation

    num_samples = analyzer.xx.shape[0]
    analyzer.reachability_results = np.zeros(num_samples, dtype=bool)
    for i in range(num_samples):
        position.x,position.y,position.z = analyzer.xx[i],analyzer.yy[i],analyzer.zz[i]
        analyzer.reachability_results[i] = analyzer.analyze_work_envelope(position, orientation)

    data = np.column_stack((analyzer.xx, analyzer.yy, analyzer.zz, analyzer.reachability_results))
    
    # Save with headers
    np.savetxt(
        'results.csv',
        data,
        delimiter=",",
        header="x,y,z,is_reachable",
        fmt=["%.3f", "%.3f", "%.3f", "%s"],  # Format floats + booleans
    )
    
    analyzer.get_logger().info(f"Saved results")
    # Start analysis in a separate thread to avoid blocking
    #analysis_thread = Thread(target=analyzer.analyze_work_envelope, 
                           #kwargs={'grid_size': 0.1, 'max_distance': 0.5})
    #analysis_thread.start()
    
    
    #rclpy.spin(analyzer)

    #analyzer.get_logger().info("Shutting down...")
    print("Shutting down...")
    #analysis_thread.join()
    
if __name__ == "__main__":
    main()