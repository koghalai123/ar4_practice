#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point, Pose
from geometry_msgs.msg import PoseStamped, PoseArray
from scipy.spatial.transform import Rotation as R
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
import subprocess
from loadCalibration import save_simulator

import copy




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
    
    quat = [msg.pose.orientation.x, msg.pose.orientation.y, 
            msg.pose.orientation.z, msg.pose.orientation.w]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler('xyz')  # Roll, Pitch, Yaw in radians

    return np.concatenate([position, euler_angles])

def kill_ros2_nodes():
    """Kill specific ROS2 nodes related to MoveIt"""
    nodes_to_kill = [
        'move_group',
        'planning_scene_monitor', 
        'trajectory_execution_manager'
    ]
    
    for node_name in nodes_to_kill:
        try:
            result = subprocess.run([
                'ros2', 'node', 'list'
            ], capture_output=True, text=True, timeout=5)
            
            if node_name in result.stdout:
                print(f"Killing ROS2 node: {node_name}")
                subprocess.run([
                    'ros2', 'lifecycle', 'set', f'/{node_name}', 'shutdown'
                ], timeout=3)
        except subprocess.TimeoutExpired:
            print(f"Timeout killing node {node_name}")
        except Exception as e:
            print(f"Error killing node {node_name}: {e}")
def launchMoveIt():
    moveit_cmd = [
        "ros2", "launch", "annin_ar4_moveit_config", "moveit.launch.py",
        "use_sim_time:=true", "include_gripper:=False"
    ]

    # Launch MoveIt2 in a new GNOME terminal window
    moveItProcess = subprocess.Popen([
        "gnome-terminal", "--", "bash", "-c",
        " ".join(moveit_cmd) + "; exec bash"
    ])
    time.sleep(4)
    return moveItProcess
        
def restartMoveIt(moveItProcess):
    moveit_cmd = [
        "ros2", "launch", "annin_ar4_moveit_config", "moveit.launch.py",
        "use_sim_time:=true", "include_gripper:=False"
    ]
    print("Restarting MoveIt2...")

    subprocess.run(["pkill", "-f", "moveit.launch.py"], timeout=5)
    subprocess.run(["pkill", "-f", "move_group"], timeout=5)
    subprocess.run(["pkill", "-f", "planning_scene_monitor"], timeout=5)
    subprocess.run(["pkill", "-f", "trajectory_execution_manager"], timeout=5)

    moveItProcess.terminate()
    try:
        moveItProcess.wait(timeout=10)
    except subprocess.TimeoutExpired:
        moveItProcess.kill()
    moveItProcess = subprocess.Popen([
        "gnome-terminal", "--", "bash", "-c",
        " ".join(moveit_cmd) + "; exec bash"
    ])
    time.sleep(8)  
    return moveItProcess

def main(args=None):

    moveItProcess = launchMoveIt()

    rclpy.init(args=args)
    
    # Create a minimal node for querying topics
    query_node = Node('aruco_query_node')
    
    frame = "end_effector_link"
    robot = AR4Robot()
    
    robot.disable_logging()
    marker_publisher = SurfacePublisher()
    
    # Create simulator with camera mode for visual demonstration
    simulator = CalibrationConvergenceSimulator(n=10, numIters=6, 
                                               dQMagnitude=0.0, dLMagnitude=0.0, 
                                               dXMagnitude=0.0, camera_mode=True)
    simulator.robot = robot
    simulator.dLMat[0,5] = 1.25
    lastMotionSuccess = time.time()
    if simulator.camera_mode:
        simulator.targetPosNom, simulator.targetOrientNom = simulator.robot.from_preferred_frame(
            np.array([0.45,0.0,0]),np.array([np.pi,-np.pi/2,0]))
        simulator.targetPosActual = simulator.targetPosNom + simulator.dX[:3]
        simulator.targetOrientActual = simulator.targetOrientNom + simulator.dX[3:]
        simulator.targetPosEst = simulator.targetPosNom
        simulator.targetOrientEst = simulator.targetOrientNom
    else:
        simulator.targetPosEst = np.array([0.0,0,0])
        simulator.targetOrientEst = np.array([0,0,0])
        simulator.targetPosActual = simulator.targetPosNom + simulator.dX[:3]
        simulator.targetOrientActual = simulator.targetOrientNom
        simulator.targetPosEst = simulator.targetPosNom
        simulator.targetOrientEst = simulator.targetOrientNom
        
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
                relativeToHomeAtGround, relativeToHomePos, globalHomePos = simulator.get_new_end_effector_position()

                globalEndEffectorPos = relativeToHomePos + globalHomePos

                targetPosEstNiceFrame, temp = simulator.robot.to_preferred_frame(simulator.targetPosEst,simulator.targetOrientEst)
                vectorToTarget = targetPosEstNiceFrame - globalEndEffectorPos
                roll, pitch, yaw = simulator.calculate_camera_pose(vectorToTarget[0], 
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
                joint_positions_est[5]= joint_positions_est[5] +np.pi/2
                motion1Succeeded = robot.move_to_joint_positions(joint_positions_est-0.1)
                if motion1Succeeded:
                    motionSucceeded = robot.move_to_joint_positions(joint_positions_est)
                else:
                    if time.time() - lastMotionSuccess > 8:
                        moveItProcess = restartMoveIt(moveItProcess)
                        simulator.robot.move_to_home()
                        lastMotionSuccess = time.time()  # Reset timer after restart
                    continue
                

                if motionSucceeded:
                    for k in range(1):
                        time.sleep(0.2)
                        lastMotionSuccess = time.time()
                        arucoSensedPose = query_aruco_pose(query_node)
                        if arucoSensedPose is None:
                            continue
                        
                        #arucoSensedPose[1] = -arucoSensedPose[1]  
                        #arucoSensedPose[0] = -arucoSensedPose[0]  
                        simulator.camera_to_target_actual = arucoSensedPose
                        #Rerun the command with an actual camera measurement
                        pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_pose(
                        robot=robot, pose=pose_desired, calibrate=True, frame="base_link", 
                        camera_to_target_meas=arucoSensedPose
                        )   
                        marker_publisher.publishPlane(np.array([0.146]), simulator.targetPosEst, id=1,
                                                    color=np.array([0.2, 0.8, 0.2])
                                                    , euler=simulator.targetOrientEst)
                        
                        marker_publisher.publishPlane(np.array([0.146]), simulator.targetPoseMeasured[simulator.current_sample][:3], id=2,
                                                    color = np.array([1.0, 1.0, 1.0])
                                                    , euler=  simulator.targetPoseMeasured[simulator.current_sample][3:])
                        '''marker_publisher.publishPlane(np.array([0.146]), simulator.cameraFocus[:3], id=3,
                                                    color = np.array([1.0, 0.0, 0.0])
                                                    , euler=  simulator.cameraFocus[3:])'''
                        marker_publisher.publish_arrow_between_points(
                        start=np.array([pose_commanded[0], pose_commanded[1], pose_commanded[2]]),
                        end=np.array([simulator.targetPosEst[0], simulator.targetPosEst[1], simulator.targetPosEst[2]]),
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
            #print(f"Measured Target Pose: {simulator.targetPoseMeasured[simulator.current_sample]}")
            simulator.current_sample += 1  
            

        results = simulator.process_iteration_results(
                simulator.targetPoseMeasured,
                simulator.targetPoseExpected,
                simulator.numJacobianTrans,
                simulator.numJacobianRot)
        # Save results to CSV
        simulator.save_to_csv(filename='physical_calibration_data.csv')
        simulator.robot = None

        simulator_copy = copy.deepcopy(simulator)
        simulator.robot = robot
        save_simulator(simulator_copy, filename=f'simulator_state_iter_{j}.pkl')
    

    print('Physical calibration simulation completed!')
    query_node.destroy_node()
    rclpy.shutdown()
    save_simulator(simulator)

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