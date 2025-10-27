
import rclpy
from rclpy.node import Node
from moveit_action_client import MoveItActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf_transformations import quaternion_from_euler, euler_from_quaternion, euler_matrix, euler_from_matrix
import numpy as np
import time
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from loadCalibration import load_simulator
from std_msgs.msg import String, Float64


from ar4_robot_py import AR4Robot
class USBIndicatorReader(Node):
    def __init__(self):
        super().__init__('usb_indicator_reader')
        
        # Store latest values from each device
        self.device_values = {}
        self.last_update_time = {}
        
        # Subscribe to individual device topics
        self.device1_sub = self.create_subscription(
            Float64,
            '/usb_indicator/target_device_1/input',
            lambda msg: self.device_callback('device1', msg),
            10
        )
        
        self.device2_sub = self.create_subscription(
            Float64,
            'usb_indicator/target_device_2/input', 
            lambda msg: self.device_callback('device2', msg),
            10
        )
        
        self.device3_sub = self.create_subscription(
            Float64,
            'usb_indicator/target_device_3/input',
            lambda msg: self.device_callback('device3', msg), 
            10
        )
    
        self.get_logger().info('USB Indicator Reader started')

    def device_callback(self, device_name, msg):
        """Callback for individual device input"""
        self.device_values[device_name] = msg.data
        self.last_update_time[device_name] = time.time()
        self.get_logger().info(f'{device_name}: {msg.data}')

    def get_latest_value(self, device_name, timeout=5.0):
        """
        Get the latest value from a specific device
        
        Args:
            device_name: 'device1', 'device2', or 'device3'
            timeout: How long to wait for fresh data (seconds)
        
        Returns:
            float or None: Latest value or None if no recent data
        """
        current_time = time.time()
        
        if device_name in self.device_values:
            # Check if we have recent data
            if device_name in self.last_update_time:
                if (current_time - self.last_update_time[device_name]) <= timeout:
                    return self.device_values[device_name]
        
        return None
    def poll_all_devices(self):
        """Poll all devices and return current values"""
        return {
            'device1': self.get_latest_value('device1'),
            'device2': self.get_latest_value('device2'), 
            'device3': self.get_latest_value('device3')
        }


def main():
    """Example usage of the AR4Robot class with movement completion detection"""
    rclpy.init()
    simulator = load_simulator(filename='simulator_state.pkl')
    #reader = USBIndicatorReader()
    '''for _ in range(10):
        
        rclpy.spin_once(reader, timeout_sec=0.5)
        print(reader.poll_all_devices())'''
    
    """-0.072606,-0.516455,0.163363
    0.072606,-0.516455,0.163363
    -0.072606,-0.371244,0.018151
    0.072606,-0.371244,0.018151
    0.000000,-0.443850,0.090757"""

    

    '''position = np.array([0.496455, 0.072606, 0.163363])
    orientation_euler = np.array([np.pi, 0.0, 0.0])
    simulator.moveToPose(position, orientation_euler,calibrate = True)
    simulator.moveToPose(position, orientation_euler,calibrate = False)

    position = np.array([0.496455, -0.072606, 0.163363])
    orientation_euler = np.array([0.0, 0.0, 0.0])
    simulator.moveToPose(position, orientation_euler,calibrate =True)
    simulator.moveToPose(position, orientation_euler,calibrate = False)'''

    orientation_euler = np.array([0.0, -np.pi/2, np.pi/2])
    zeroPoint,notNeeded = simulator.robot.to_preferred_frame(simulator.estimatedTargetPoseHistory[-1,:3],orientation_euler)
    zeroPoint = zeroPoint+np.array([-0.01,0.0,0.0])  #move down 10 cm to account for gripper offset
    xOffset = np.array([0.506455 - 0.443850])
    yOffset = xOffset
    #Additional offset added here to account for the difference in end effector size between the camera and test block for validation
    zOffset = 2*xOffset/0.9 +0.043

    middle = np.hstack((0, 0, zOffset))
    backRight = np.hstack((-xOffset, -yOffset, zOffset))
    backLeft = np.hstack((-xOffset, yOffset, zOffset))
    frontRight = np.hstack((xOffset, -yOffset, zOffset))
    frontLeft = np.hstack((xOffset, yOffset, zOffset))

    '''simulator.moveToPose( zeroPoint+backLeft*1.2, orientation_euler,calibrate = True)
    simulator.robot.move_to_home()
    simulator.moveToPose( zeroPoint+backLeft*1.2, orientation_euler,calibrate = True)
    simulator.moveToPose( zeroPoint+backLeft, orientation_euler,calibrate = True)'''



    '''simulator.moveToPose( zeroPoint+backRight*1.2, orientation_euler,calibrate = True)
    simulator.robot.move_to_home()
    simulator.moveToPose( zeroPoint+backRight*1.2, orientation_euler,calibrate = True)
    simulator.moveToPose( zeroPoint+backRight, orientation_euler,calibrate = True)'''

    '''simulator.robot.move_to_home()
    simulator.moveToPose( zeroPoint+frontLeft, orientation_euler,calibrate = True)'''

    simulator.robot.move_to_home()
    simulator.moveToPose( zeroPoint+frontRight, orientation_euler,calibrate = True)
    #simulator.robot.move_to_home()
    #simulator.moveToPose(position, orientation_euler,calibrate = False)

    #simulator.robot.move_to_home()
    
    #rclpy.shutdown()
    print("done")

if __name__ == '__main__':
    main()