#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import argparse
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point, Pose
from geometry_msgs.msg import PoseStamped, PoseArray
from control_msgs.msg import JointTrajectoryControllerState
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




def _pose_stamped_to_array(msg):
    """Convert a PoseStamped into [x, y, z, roll, pitch, yaw] (euler xyz, radians)."""
    if msg is None:
        return None
    position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    quat = [msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w]
    euler_angles = R.from_quat(quat).as_euler('xyz')  # Roll, Pitch, Yaw in radians
    return np.concatenate([position, euler_angles])


class ArucoPoseSubscriber:
    """Persistent subscriptions to the aruco pose topics.

    Holds one long-lived subscription per topic and caches the latest message,
    so callers read the most recent pose without creating/destroying a
    subscription on every query (the old query_aruco_pose pattern).

    Topics:
      - /aruco_marker/raw_pose : the estimated aruco marker pose
      - /aruco_marker_pose     : the live camera measurement of the marker
    """

    ESTIMATE_TOPIC = '/aruco_marker/raw_pose'
    MEASUREMENT_TOPIC = '/aruco_marker_pose'

    def __init__(self, node):
        self.node = node
        self._estimate_msg = None
        self._measurement_msg = None
        self._estimate_sub = node.create_subscription(
            PoseStamped, self.ESTIMATE_TOPIC, self._on_estimate, 1)
        self._measurement_sub = node.create_subscription(
            PoseStamped, self.MEASUREMENT_TOPIC, self._on_measurement, 1)

    def _on_estimate(self, msg):
        self._estimate_msg = msg

    def _on_measurement(self, msg):
        self._measurement_msg = msg

    def _wait_for(self, attr, timeout, require_fresh):
        """Spin until a (optionally fresh) message arrives on `attr`, or timeout.

        Returns the pose as [x, y, z, roll, pitch, yaw], or None on timeout.
        """
        if require_fresh:
            setattr(self, attr, None)
        start_time = time.time()
        while getattr(self, attr) is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return _pose_stamped_to_array(getattr(self, attr))

    def get_estimate(self, timeout=1.0, require_fresh=True):
        """Latest /aruco_marker/raw_pose as [x,y,z,roll,pitch,yaw], or None."""
        return self._wait_for('_estimate_msg', timeout, require_fresh)

    def get_measurement(self, timeout=1.0, require_fresh=True):
        """Latest /aruco_marker_pose as [x,y,z,roll,pitch,yaw], or None."""
        return self._wait_for('_measurement_msg', timeout, require_fresh)


class ControllerStateMonitor:
    """Caches joint_trajectory_controller state so we can report exactly which
    joint failed to reach its target when a motion fails / times out (-6)."""

    TOPIC = '/joint_trajectory_controller/controller_state'

    def __init__(self, node):
        self.node = node
        self._msg = None
        self._sub = node.create_subscription(
            JointTrajectoryControllerState, self.TOPIC, self._on_state, 10)

    def _on_state(self, msg):
        self._msg = msg

    def log_tracking_error(self, label="", target=None, timeout=0.5):
        """Print a per-joint breakdown of where the robot actually ended up vs.
        where the trajectory wanted it (and vs. our commanded target)."""
        # Spin briefly for a fresh controller_state sample.
        self._msg = None
        start = time.time()
        while self._msg is None and (time.time() - start) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        if self._msg is None:
            print(f"[DIAG {label}] no controller_state on {self.TOPIC}")
            return
        names = list(self._msg.joint_names)
        err = list(self._msg.error.positions)
        actual = list(self._msg.feedback.positions)
        desired = list(self._msg.reference.positions)
        if not names:
            names = [f"j{i+1}" for i in range(len(err))]
        worst = max(range(len(err)), key=lambda i: abs(err[i]))
        print(f"[DIAG {label}] worst joint: {names[worst]} "
              f"tracking_err={np.degrees(err[worst]):.1f} deg")
        for i, n in enumerate(names):
            tgt = (f"  cmd_target={np.degrees(target[i]):7.1f}"
                   if target is not None and i < len(target) else "")
            print(f"    {n}: actual={np.degrees(actual[i]):7.1f}  "
                  f"desired={np.degrees(desired[i]):7.1f}  "
                  f"err={np.degrees(err[i]):6.1f} deg{tgt}")


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
        "use_sim_time:=false", "include_gripper:=False"
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
        "use_sim_time:=false", "include_gripper:=False"
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

    # Persistent subscriptions to the aruco pose topics (estimate + camera measurement)
    aruco_poses = ArucoPoseSubscriber(query_node)

    # Diagnostic: caches controller state to report which joint fails on a -6.
    ctrl_monitor = ControllerStateMonitor(query_node)

    frame = "end_effector_link"
    robot = AR4Robot()
    
    robot.disable_logging()
    marker_publisher = SurfacePublisher()
    
    # Create simulator with camera mode for visual demonstration
    simulator = CalibrationConvergenceSimulator(n=8, numIters=40, 
                                               dQMagnitude=0.0, dLMagnitude=0.0, 
                                               dXMagnitude=0.0, camera_mode=True)
    simulator.robot = robot
    simulator.dLMat[0,5] = 1.25
    lastMotionSuccess = time.time()
    if simulator.camera_mode:
        simulator.targetPosNom, simulator.targetOrientNom = simulator.robot.from_preferred_frame(
            np.array([0.47,-0.03,0]),np.array([np.pi,-np.pi/2,0]))
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
            # Publish the target estimate plane every iteration, independent of
            # motion/measurement success, so RViz always has a marker to display.
            marker_publisher.publishPlane(np.array([0.146]), simulator.targetPosEst, id=1,
                                          color=np.array([0.2, 0.8, 0.2]),
                                          euler=simulator.targetOrientEst)
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
                approach_target = joint_positions_est - 0.1
                # !!! DO NOT raise these toward 1.0 without testing on hardware !!!
                # These scale the planned move speed/accel. At/near full scaling,
                # joint 6 (which makes the largest move due to the +pi/2 offset
                # above) overshoots its target by ~6-12 deg, exceeds the controller
                # goal tolerance, and the move is ABORTED by move_group with
                # "error code: -6" (TIMED_OUT). That stalls each measurement ~12 s
                # and, in the worst case, looks like the robot "freezing". 0.6 was
                # the empirical sweet spot (0.3 = very safe but slow; 1.0 = frequent
                # -6). If you change these and start seeing -6 errors, this is why.
                MOVE_VEL_SCALE = 0.6
                MOVE_ACC_SCALE = 0.6
                motion1Succeeded = robot.move_to_joint_positions(
                    approach_target, MOVE_VEL_SCALE, MOVE_ACC_SCALE)
                if motion1Succeeded:
                    motionSucceeded = robot.move_to_joint_positions(
                        joint_positions_est, MOVE_VEL_SCALE, MOVE_ACC_SCALE)
                    if not motionSucceeded:
                        print(f"[MOVE-FAIL] TARGET move failed | "
                              f"cmd J(deg)={np.round(np.degrees(joint_positions_est), 1)}")
                        ctrl_monitor.log_tracking_error(
                            label="target", target=joint_positions_est)
                else:
                    print(f"[MOVE-FAIL] APPROACH move failed | "
                          f"cmd J(deg)={np.round(np.degrees(approach_target), 1)}")
                    ctrl_monitor.log_tracking_error(
                        label="approach", target=approach_target)
                    # RViz/MoveIt restart disabled for investigation -- just
                    # re-sample a new pose instead of restarting MoveIt.
                    continue
                

                if motionSucceeded:
                    # The live aruco detection node publishes the camera measurement
                    # on /aruco_marker/raw_pose (== get_estimate()). Retry the read a
                    # few times at this pose before giving up.
                    arucoSensedPose = None
                    for _ in range(5):
                        time.sleep(0.2)
                        arucoSensedPose = aruco_poses.get_estimate()
                        if arucoSensedPose is not None:
                            break
                    if arucoSensedPose is None:
                        # Marker not seen here -- re-sample a new pose instead of
                        # recording a nominal (cameraless) measurement as if it were real.
                        continue
                    lastMotionSuccess = time.time()

                    #arucoSensedPose[1] = -arucoSensedPose[1]
                    #arucoSensedPose[0] = -arucoSensedPose[0]
                    # Rerun the command with the ACTUAL camera measurement
                    pose_actual, pose_commanded, joint_positions_actual, joint_positions_commanded = simulator.generate_measurement_pose(
                    robot=robot, pose=pose_desired, calibrate=True, frame="base_link",
                    camera_to_target_meas=arucoSensedPose
                    )
                    marker_publisher.publishPlane(np.array([0.146]), simulator.targetPosEst, id=1,
                                                color=np.array([0.2, 0.8, 0.2])
                                                , euler=simulator.targetOrientEst)
                    marker_publisher.publish_arrow_between_points(
                    start=np.array([pose_commanded[0], pose_commanded[1], pose_commanded[2]]),
                    end=np.array([simulator.targetPosEst[0], simulator.targetPosEst[1], simulator.targetPosEst[2]]),
                    thickness=0.01,
                    id=1,
                    color=np.array([0.0, 1.0, 0.0])
                    )
                    successfulMeasurement = True
                else:
                    counter += 1

            # Publish the measured target plane (white) for this measurement so RViz
            # visualizes the incoming measurement, independent of the aruco/motion guards.
            marker_publisher.publishPlane(
                np.array([0.146]),
                simulator.targetPoseMeasured[simulator.current_sample][:3], id=2,
                color=np.array([1.0, 1.0, 1.0]),
                euler=simulator.targetPoseMeasured[simulator.current_sample][3:])

            if simulator.camera_mode:
                # Use the actual camera-to-target measurement
                numJacobianTrans, numJacobianRot = simulator.compute_jacobians(
                    simulator.joint_positions_commanded[simulator.current_sample], 
                    camera_to_target=simulator.camera_to_target_meas)
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
        simulator.save_to_csv(filename='videoGeneration.csv')
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