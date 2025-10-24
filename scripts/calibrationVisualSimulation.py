#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from scipy.spatial.transform import Rotation as R

from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point, Pose
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
import pickle

def main(args=None):
    rclpy.init()
    frame = "end_effector_link"

    robot = AR4Robot()
    robot.disable_logging()
    marker_publisher = SurfacePublisher()
    # Create simulator with camera mode for visual demonstration
    simulator = CalibrationConvergenceSimulator(n=7, numIters=1, 
                                               dQMagnitude=0.01, dLMagnitude=0.01, 
                                               dXMagnitude=0.01, camera_mode=True, noiseMagnitude=0.00, robot = robot)
    if simulator.camera_mode:
        simulator.targetPosNom, simulator.targetOrientNom = simulator.robot.from_preferred_frame(
            np.array([0.3,0,0]),np.array([np.pi,-np.pi/2,0]))
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
            motionSucceeded = False
            while motionSucceeded is False:
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
                
                motionSucceeded = robot.move_to_joint_positions(joint_positions_actual)
                
                
                if motionSucceeded:
                    marker_publisher.publishPlane(np.array([0.146]), simulator.targetPoseMeasured[simulator.current_sample][:3], id=2,
                                                    color = np.array([1.0, 1.0, 1.0])
                                                    , euler=  simulator.targetPoseMeasured[simulator.current_sample][3:])

                    marker_publisher.publishPlane(np.array([0.146]), simulator.targetPosEst, id=1,
                                                  color=np.array([0.2, 0.8, 0.2])
                                                  , euler=simulator.targetOrientEst)
                    marker_publisher.publishPlane(np.array([0.146]), simulator.targetPosActual, id=0
                                                  , euler=simulator.targetOrientActual)
                    
                    marker_publisher.publish_arrow_between_points(
                    start=np.array([pose_commanded[0], pose_commanded[1], pose_commanded[2]]),
                    end=np.array([simulator.targetPosEst[0], simulator.targetPosEst[1], simulator.targetPosEst[2]]),
                    thickness=0.01,
                    id=1,
                    color=np.array([0.0, 1.0, 0.0])
                    )
                    break
                counter += 1

            
            
            # Compute Jacobians properly based on mode
            if simulator.camera_mode:
                # Use the actual camera-to-target measurement
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample], 
                    camera_to_target=simulator.camera_to_target_meas_test)
            else:
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample])
                
            simulator.current_sample += 1  
            print(f"Measurement {i}: Generated successfully")
        
        results = simulator.process_iteration_results(
                simulator.targetPoseMeasured,
                simulator.targetPoseExpected,
                simulator.numJacobianTrans,
                simulator.numJacobianRot)
    
    # Save results to CSV
    #simulator.save_to_csv(filename='visual_calibration_data.csv')
    simulator_data = {
        'targetPosEst': simulator.targetPosEst,
        'targetOrientEst': simulator.targetOrientEst,
        'targetPosActual': simulator.targetPosActual,
        'targetOrientActual': simulator.targetOrientActual,
        'targetPosNom': simulator.targetPosNom,
        'targetOrientNom': simulator.targetOrientNom,
        'targetPoseMeasured': getattr(simulator, 'targetPoseMeasured', None),
        'targetPoseExpected': getattr(simulator, 'targetPoseExpected', None),
        'numJacobianTrans': getattr(simulator, 'numJacobianTrans', None),
        'numJacobianRot': getattr(simulator, 'numJacobianRot', None),
        'joint_positions_commanded': getattr(simulator, 'joint_positions_commanded', None),
        'joint_positions_actual': getattr(simulator, 'joint_positions_actual', None),
        'camera_to_target_meas_test': getattr(simulator, 'camera_to_target_meas_test', None),
        'n': simulator.n,
        'numIters': simulator.numIters,
        'camera_mode': simulator.camera_mode,
        'dQMagnitude': simulator.dQMagnitude,
        'dLMagnitude': simulator.dLMagnitude,
        'dXMagnitude': simulator.dXMagnitude,
        'noiseMagnitude': simulator.noiseMagnitude,
        'dX': getattr(simulator, 'dX', None),
        'current_sample': getattr(simulator, 'current_sample', 0),
        'current_iteration': getattr(simulator, 'current_iteration', 0)
    }

    with open('simulator_state.pkl', 'wb') as f:
        pickle.dump(simulator_data, f)

    print('Visual calibration simulation completed!')
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
    profile_main()#!/usr/bin/env python3
