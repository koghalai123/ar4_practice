#!/usr/bin/env python3


import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from pymoveit2 import MoveIt2
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import time
class JointCommander(Node):
    def __init__(self):
        super().__init__("joint_commander")
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        time.sleep(0.1)
        self.resetErrors()
        self.moveit2.max_velocity = 2
        self.moveit2.max_acceleration = 2
        
        # Circular motion parameters
        self.center = Point(x=0.5, y=0.0, z=0.5)  # Default center position
        self.radius = 0.1  # Default radius in meters
        self.revolutions = 1  # Default number of revolutions
        self.waypoints_count = 20  # Waypoints per revolution
        self.plane = 'xy'  # Default plane of rotation
        
        # Get and display current pose
        self.display_current_pose()
        
        self.get_logger().info("Joint commander initialized. Ready to accept commands.")
        self.get_logger().info("Commands:")
        self.get_logger().info("  - Joint angles: enter 6 comma-separated values")
        self.get_logger().info("  - Circular motion: enter 'circle' followed by parameters (optional)")
        self.get_logger().info("     Example: 'circle center=0.5,0,0.5 radius=0.1 revs=1 plane=xy'")

    def resetErrors(self):
        """Reset MoveIt2 state and ensure joint states are available"""
        print("Resetting MoveIt2 state...")
        
        self.moveit2.reset_new_joint_state_checker()
        self.moveit2.force_reset_executing_state()
    def display_current_pose(self):
        """Display the current end effector pose"""
        self.resetErrors()
        fk_result = self.moveit2.compute_fk()
        
        if fk_result is None:
            self.get_logger().warn("Could not compute current pose")
            return
            
        if isinstance(fk_result, list):
            current_pose = fk_result[0]  # Take first result if multiple
        else:
            current_pose = fk_result
            
        # Convert quaternion to Euler angles for more intuitive display
        quat = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quat)
        
        # Print current pose information
        self.get_logger().info("\nCurrent End Effector Pose:")
        self.get_logger().info(f"Position: x={current_pose.pose.position.x:.3f}, y={current_pose.pose.position.y:.3f}, z={current_pose.pose.position.z:.3f}")
        self.get_logger().info(f"Orientation (Quaternion): x={current_pose.pose.orientation.x:.3f}, y={current_pose.pose.orientation.y:.3f}, z={current_pose.pose.orientation.z:.3f}, w={current_pose.pose.orientation.w:.3f}")
        self.get_logger().info(f"Orientation (Euler RPY): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")

    def generate_circle_waypoints(self):
        """Generate waypoints forming a circle in the specified plane"""
        waypoints = []
        
        for i in range(self.waypoints_count * self.revolutions + 1):  # +1 to complete the circle
            angle = 2 * math.pi * i / self.waypoints_count
            
            if self.plane == 'xy':
                x = self.center.x + self.radius * math.cos(angle)
                y = self.center.y + self.radius * math.sin(angle)
                z = self.center.z
            elif self.plane == 'xz':
                x = self.center.x + self.radius * math.cos(angle)
                y = self.center.y
                z = self.center.z + self.radius * math.sin(angle)
            elif self.plane == 'yz':
                x = self.center.x
                y = self.center.y + self.radius * math.cos(angle)
                z = self.center.z + self.radius * math.sin(angle)
            else:
                raise ValueError("Invalid plane specified. Use 'xy', 'xz', or 'yz'")
            
            # Use current orientation or fixed orientation
            current_pose = self.moveit2.compute_fk()
            if current_pose is None:
                orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
            else:
                orientation = current_pose.pose.orientation
            
            waypoints.append(Pose(
                position=Point(x=x, y=y, z=z),
                orientation=orientation
            ))
        
        return waypoints

    def execute_smooth_circular_motion(self):
        """Execute a smooth continuous circular motion"""
        waypoints = self.generate_circle_waypoints()
        
        # Move to starting position first
        self.moveit2.move_to_pose(pose=waypoints[0])
        self.moveit2.wait_until_executed()
        
        # Plan the entire circular path as one continuous trajectory
        future = self.moveit2.plan_async(
            pose=waypoints[1],  # First target (rest will be waypoints)
            cartesian=True,
            max_step=0.001,  # Fine resolution for smooth motion
        )
        
        # Wait for planning to complete
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Get the complete trajectory
        trajectory = self.moveit2.get_trajectory(
            future,
            cartesian=True,
            cartesian_fraction_threshold=0.9
        )
        
        if trajectory is not None:
            # Execute the complete smooth trajectory
            self.moveit2.execute(trajectory)
            self.moveit2.wait_until_executed()
        else:
            self.get_logger().error("Failed to plan circular path")

    def parse_circle_params(self, params_str):
        """Parse circle command parameters"""
        params = params_str.split()
        for param in params:
            if '=' in param:
                key, value = param.split('=')
                if key == 'center':
                    self.center.x, self.center.y, self.center.z = map(float, value.split(','))
                elif key == 'radius':
                    self.radius = float(value)
                elif key == 'revs':
                    self.revolutions = int(value)
                elif key == 'plane':
                    self.plane = value.lower()
                    if self.plane not in ['xy', 'xz', 'yz']:
                        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

    def run(self):
        while rclpy.ok():
            try:
                input_str = input("\nEnter command: ").strip()
                
                if input_str.lower().startswith('circle'):
                    # Handle circular motion command
                    params_str = input_str[6:].strip()
                    if params_str:
                        self.parse_circle_params(params_str)
                    self.execute_smooth_circular_motion()
                    self.display_current_pose()  # Show new pose after motion
                else:
                    # Handle joint angle command
                    joint_positions = [float(x.strip()) for x in input_str.split(',')]
                    if len(joint_positions) != 6:
                        raise ValueError("Exactly 6 joint angles required")
                    self.moveit2.move_to_configuration(joint_positions)
                    self.moveit2.wait_until_executed()
                    self.display_current_pose()  # Show new pose after motion
                    
            except Exception as e:
                self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    commander = JointCommander()
    commander.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



'''
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2

class JointCommander(Node):
    def __init__(self):
        super().__init__("joint_commander")
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
        )
        
        self.moveit2.max_velocity = 0.75
        self.moveit2.max_acceleration = 0.5
        
        self.get_logger().info("Joint commander initialized. Ready to accept commands.")

    def run(self):
        while rclpy.ok():
            try:
                input_str = input("\nEnter joint angles (radians, comma separated): ")
                joint_positions = [float(x.strip()) for x in input_str.split(',')]
                self.moveit2.move_to_configuration(joint_positions)
                self.moveit2.wait_until_executed()
            except Exception as e:
                self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    commander = JointCommander()
    commander.run()
    rclpy.shutdown()

if __name__ == "__main__":
    main()'''