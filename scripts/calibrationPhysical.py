#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point, Pose
from geometry_msgs.msg import PoseStamped, PoseArray

from surface_publisher import SurfacePublisher
from ar4_robot_py import AR4Robot
from calibrationConvergenceSimulation import CalibrationConvergenceSimulator
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
import csv
import os
import time
import cProfile
import pstats
import io

def get_new_end_effector_position(robot):
    random_numbers = np.random.rand(3) 
    scale = 0.3
    random_pos = (random_numbers-0.5)*scale
    xOffset = 0 - scale
    yOffset = 0
    zOffset = robot.pos_offsets["z"] -scale
    x = xOffset + random_pos[0]
    y = yOffset + random_pos[1]
    z = zOffset + random_pos[2]
    relativeToHomeAtGround = np.array([x,y,z])

    globalHomePos = np.array([robot.pos_offsets["x"],robot.pos_offsets["y"],robot.pos_offsets["z"]])
    return relativeToHomeAtGround, random_pos, globalHomePos
def calculate_camera_pose(x, y, z, roll_offset=0.0, pitch_offset=0.0, yaw_offset=0.0):
    # Calculate robot end effector orientation to point toward target
    # Vector [x, y, z] is from end effector to target in world frame
    
    # Vector from end effector to target
    vector_to_target = np.array([x, y, z])
    distance = np.linalg.norm(vector_to_target)
    
    if distance == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate yaw: rotation around Z axis to point in XY direction of target
    yaw = np.arctan2(y, x)
    
    # Calculate pitch: elevation angle from XY plane
    xy_distance = np.sqrt(x**2 + y**2)
    pitch = np.arctan2(z, xy_distance)
    
    # Keep roll at 0 (no rotation around the pointing direction)
    roll = 0


    roll = np.arctan2(y, -z)#+np.pi/2
    temp = np.sqrt(z**2 + y**2)
    pitch = np.arctan2(x, temp) -np.pi/2
    yaw = 0#np.pi/2
    
    # Apply offsets
    roll += roll_offset
    pitch += pitch_offset
    yaw += yaw_offset
    
    return roll, pitch, yaw


def print_current_position(robot,frame):
    current_pose = robot.get_current_pose(frame)
    position, orientation = current_pose
    robot.get_logger().info(f"Current Pose: Position={position}, Orientation={orientation}")

def camera_vector_from_pose_and_measurement(roll, pitch, yaw, distance):
    """
    Calculate a vector to target given orientation angles and distance.
    
    Parameters:
    - roll: Roll angle in radians
    - pitch: Pitch angle in radians
    - yaw: Yaw angle in radians
    - distance: Distance to the target
    
    Returns:
    - numpy array [x, y, z]: Vector pointing to the target
    """
    # First, understand the relationship between the angles and the direction vector
    # If we have pitch = -arctan2(x, z) - π/2, then x/z = -tan(pitch + π/2)
    # Similarly for roll = -arctan2(y, r), where r = sqrt(x^2 + z^2), then y/r = -tan(roll)
    
    # Calculate the vector components based on the spherical coordinates
    x_direction = -np.sin(pitch + np.pi/2) * np.cos(roll)
    y_direction = -np.sin(roll) * np.cos(pitch + np.pi/2)
    z_direction = np.cos(pitch + np.pi/2) * np.cos(roll)
    
    # Normalize the direction vector
    direction = np.array([x_direction, y_direction, z_direction])
    direction = direction / np.linalg.norm(direction)
    
    # Scale by the distance
    vector = direction * distance
    
    return vector

def query_aruco_pose(node):
    """Query the latest ArUco pose using ROS2 API - returns position and euler angles"""
    msg_received = [None]
    def callback(msg):
        msg_received[0] = msg
    sub = node.create_subscription(PoseStamped, '/aruco_marker/raw_pose', callback, 1)
    
    # Spin for up to 1 second to get the message
    timeout = 1.0  # seconds
    start_time = time.time()
    
    while msg_received[0] is None and (time.time() - start_time) < timeout:
        rclpy.spin_once(node, timeout_sec=0.01)

    # Clean up subscription

    node.destroy_subscription(sub)
    
    if msg_received[0] is None:
        return None
    msg = msg_received[0]
    # Extract position
    
    position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    
    # Convert quaternion to euler angles
    from scipy.spatial.transform import Rotation as R
    quat = [msg.pose.orientation.x, msg.pose.orientation.y, 
            msg.pose.orientation.z, msg.pose.orientation.w]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler('xyz')  # Roll, Pitch, Yaw in radians

    return np.concatenate([position, euler_angles])



def main(args=None):
    rclpy.init(args=args)
    
    # Create a minimal node for querying topics
    query_node = Node('aruco_query_node')
    
    frame = "end_effector_link"
    robot = AR4Robot()
    
    robot.disable_logging()
    marker_publisher = SurfacePublisher()
    
    # Create simulator with camera mode for visual demonstration
    simulator = CalibrationConvergenceSimulator(n=10, numIters=20, 
                                               dQMagnitude=0.0, dLMagnitude=0.0, 
                                               dXMagnitude=0.0, camera_mode=True)
    
    if simulator.camera_mode:
        simulator.targetPosNom = np.array([0.3,0,0])
        simulator.targetOrientNom = np.array([np.pi,0,0])
        simulator.targetPosEst = simulator.targetPosNom.copy()
        simulator.targetOrientEst = simulator.targetOrientNom.copy()
        simulator.targetPosActual = simulator.targetPosNom.copy()
        simulator.targetOrientActual = simulator.targetOrientNom.copy()
    else:
        simulator.targetPosEst = np.array([0.0,0,0])
        simulator.targetOrientEst = np.array([0,0,0])
        simulator.targetPosActual = simulator.targetPosNom + simulator.dX[:3]
        simulator.targetOrientActual = simulator.targetOrientNom
    
    # Process each iteration separately
    for j in range(simulator.numIters):
        print(f"\n--- Starting Iteration {j} ---")
        simulator.set_current_iteration(j)

        # Generate individual measurements
        for i in range(simulator.n):
            counter = 0
            successfulMeasurement = False
            while successfulMeasurement is False:
                # Generate random end-effector position pointing toward target
                relativeToHomeAtGround, relativeToHomePos, globalHomePos = get_new_end_effector_position(robot)

                globalEndEffectorPos = relativeToHomePos + globalHomePos
                
                # Calculate camera orientation to point at target
                # Vector should point FROM camera TO target
                vectorToTarget = simulator.targetPosActual - globalEndEffectorPos
                roll, pitch, yaw = calculate_camera_pose(vectorToTarget[0], 
                                                         vectorToTarget[1], 
                                                         vectorToTarget[2],
                                                         )
                # Generate measurement using the simulator's proper interface
                #pos_desired, orient_desired = robot.from_preferred_frame(position=globalEndEffectorPos, euler_angles=np.array([roll,pitch,yaw]),old_reference_frame="base_link", new_reference_frame="base_link")
                pose_desired = np.concatenate((globalEndEffectorPos, np.array([roll,pitch,yaw])))
                robot.warn_enabled = False
                pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_pose(
                    robot=robot, pose=pose_desired, calibrate=True, frame="base_link"
                )
                robot.warn_enabled = True
                if pose_actual is None:
                    continue
                
                joint_positions_est = joint_positions_commanded.copy()
                joint_positions_est = joint_positions_est - np.sum(simulator.dQMat, axis=0)
                joint_positions_est[4] -= np.pi/2
                motionSucceeded = robot.move_to_joint_positions(joint_positions_est)
                if motionSucceeded:
                    arucoSensedPose = query_aruco_pose(query_node)
                    if arucoSensedPose is None:
                        continue
                    simulator.camera_to_target_actual = arucoSensedPose
                    #Rerun the command with an actual camera measurement
                    pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_pose(
                    robot=robot, pose=pose_desired, calibrate=True, frame="base_link", 
                    camera_to_target_meas=arucoSensedPose
                    )   
                    targetPosWeirdFrameEst, targetOrientWeirdFrameEst = robot.from_preferred_frame(
                        simulator.targetPosEst, simulator.targetOrientEst, 
                        old_reference_frame="base_link", new_reference_frame="global")
                    endEffectorPosWeirdFrame, endEffectorOrientWeirdFrame = robot.from_preferred_frame(
                        globalEndEffectorPos, np.array([roll,pitch,yaw]), 
                        old_reference_frame="base_link", new_reference_frame="global")
                
                    marker_publisher.publishPlane(np.array([0.146]), targetPosWeirdFrameEst, id=1,
                                                  color = np.array([0.2, 0.8, 0.2])
                                                  , euler=  targetOrientWeirdFrameEst)
                    
                    marker_publisher.publishPlane(np.array([0.146]), simulator.targetPoseMeasured[simulator.current_sample][:3], id=2,
                                                  color = np.array([1.0, 1.0, 1.0])
                                                  , euler=  simulator.targetPoseMeasured[simulator.current_sample][3:])
                    '''marker_publisher.publishPlane(np.array([0.146]), simulator.cameraFocus[:3], id=3,
                                                  color = np.array([1.0, 0.0, 0.0])
                                                  , euler=  simulator.cameraFocus[3:])'''
                    marker_publisher.publish_arrow_between_points(
                    start=np.array([pose_commanded[0], pose_commanded[1], pose_commanded[2]]),
                    end=np.array([targetPosWeirdFrameEst[0], targetPosWeirdFrameEst[1], targetPosWeirdFrameEst[2]]),
                    thickness=0.01,
                    id=1,
                    color=np.array([0.0, 1.0, 0.0])
                    )
                    successfulMeasurement = True
                    break
                counter += 1

            if simulator.camera_mode:
                # Use the actual camera-to-target measurement
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample], 
                    camera_to_target=simulator.camera_to_target_actual)
            else:
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample])
                
            error = simulator.targetPoseExpected[simulator.current_sample] - simulator.targetPoseMeasured[simulator.current_sample]
            print(f"Measurement {i}: Generated successfully, Error: {error}")
            simulator.current_sample += 1  
            

        results = simulator.process_iteration_results(
                simulator.targetPoseExpected,
                simulator.targetPoseMeasured,
                simulator.numJacobianTrans,
                simulator.numJacobianRot)
    
    # Save results to CSV
    #simulator.save_to_csv(filename='visual_calibration_data.csv')
    
    print('Visual calibration simulation completed!')
    query_node.destroy_node()
    rclpy.shutdown()

def profile_main():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your existing main() function call
    main()
    
    pr.disable()
    
    # Create a stats object and sort by cumulative time
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats(30)  # Show top 30 time-consuming functions
    #print(s.getvalue())

if __name__ == "__main__":
    profile_main()