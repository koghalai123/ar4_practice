#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter



''' may need this command to install the correct version of numpy and opencv
pip uninstall opencv-python opencv-contrib-python numpy
pip install numpy==1.26.4 opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66
'''
class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator')
        
        # ROS2 Subscribers
        # on the d415i use this
        '''self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)'''
        #on the d405 use this:
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_rect_raw',  # for D435i
            # '/camera/camera/color/image_rect_raw' #for d405
            self.image_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10)
        self.bridge = CvBridge()
        
        # Publishers for marker poses
        self.raw_pose_pub = self.create_publisher(
            PoseStamped,
            '/aruco_marker/raw_pose',
            10)
        
        '''self.depth_enhanced_pose_pub = self.create_publisher(
            PoseStamped,
            '/aruco_marker/depth_enhanced_pose',
            10)
        
        self.kalman_pose_pub = self.create_publisher(
            PoseStamped,
            '/aruco_marker/kalman_filtered_pose',
            10)'''
        
        # Publisher for all detected markers
        self.all_markers_pub = self.create_publisher(
            PoseArray,
            '/aruco_markers/all_poses',
            10)
        
        # Depth data storage
        self.depth_image = None
        self.depth_scale = 0.001  # Default scale (assuming depth is in mm)

        # ARUCO Setup
        self.MARKER_SIZE_CM = 14.6/2#* 0.1021
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Kalman Filter
        self.kf = KalmanFilter(dim_x=6, dim_z=6)
        self.kf.F = np.eye(6)
        self.kf.H = np.eye(6)
        self.kf.P *= 1.
        self.kf.R = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.kf.Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])

        # Camera Calibration
        self.camera_matrix = np.array([
            [920.0,   0, 640.0],
            [  0,   920.0, 360.0],
            [  0,     0,     1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)

        # Visualization parameters
        self.text_color = (0, 255, 0)  # Green
        self.method_colors = [
            (255, 0, 0),    # Blue - Raw ArUco
            (0, 255, 255),  # Yellow - Depth-enhanced
            (0, 255, 0)     # Green - Kalman-filtered
        ]

    def depth_callback(self, msg):
        """Store the latest depth image"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth CV Bridge error: {e}")

    def get_depth_at_marker(self, corners):
        """Get median depth value at marker corners"""
        if self.depth_image is None:
            return None

        points = corners.reshape(-1, 2)
        depths = []

        for (u, v) in points:
            u, v = int(round(u)), int(round(v))
            if 0 <= u < self.depth_image.shape[1] and 0 <= v < self.depth_image.shape[0]:
                # Extract a small region around the corner
                region = self.depth_image[max(0, v-2):v+3, max(0, u-2):u+3]
                valid_depths = region[region > 0] * self.depth_scale  # Ignore invalid (zero) depths
                if valid_depths.size > 0:
                    depths.append(np.median(valid_depths))  # Use the median of the region

        # Return the median of all valid depths
        return np.median(depths) if len(depths) > 2 else None  # Require at least 3 valid depths

    def format_pose(self, t, r, prefix=""):
        """Format pose for display"""
        return (f"{prefix}X: {float(t[0]):.2f}cm Y: {float(t[1]):.2f}cm Z: {float(t[2]):.2f}cm\n"
                f"{prefix}Roll: {float(r[0]):.2f}° Pitch: {float(r[1]):.2f}° Yaw: {float(r[2]):.2f}°")

    def publish_pose(self, pose, publisher, frame_id="camera_color_optical_frame"):
        """Publish a pose as PoseStamped message"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id
        
        # Position (convert cm to meters)
        pose_msg.pose.position.x = float(pose[0]) / 100.0
        pose_msg.pose.position.y = float(pose[1]) / 100.0
        pose_msg.pose.position.z = float(pose[2]) / 100.0
        
        # Orientation (convert euler angles to quaternion)
        rot = R.from_euler('xyz', pose[3:6])
        quat = rot.as_quat()  # [x, y, z, w]
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
        publisher.publish(pose_msg)
        
    def publish_all_markers(self, poses_with_ids, frame_id="camera_color_optical_frame"):
        """Publish all detected markers as PoseArray"""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = frame_id
        
        for pose, marker_id in poses_with_ids:
            pose_stamped = PoseStamped()
            
            # Position (convert cm to meters)
            pose_stamped.pose.position.x = float(pose[0]) / 100.0
            pose_stamped.pose.position.y = float(pose[1]) / 100.0
            pose_stamped.pose.position.z = float(pose[2]) / 100.0
            
            # Orientation (convert euler angles to quaternion)
            rot = R.from_euler('xyz', pose[3:6])
            quat = rot.as_quat()  # [x, y, z, w]
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])
            
            pose_array.poses.append(pose_stamped.pose)
        
        self.all_markers_pub.publish(pose_array)

    def image_callback(self, msg):
        if self.depth_image is None:
            self.get_logger().warn("Waiting for depth data...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Color CV Bridge error: {e}")
            return

        # Create a black background for text
        text_bg = np.zeros((200, cv_image.shape[1], 3), dtype=np.uint8)
        
        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict,
            parameters=self.aruco_params
        )
        
        if ids is not None:
            # Subpixel refinement
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            for corner in corners:
                cv2.cornerSubPix(
                    gray, corner, 
                    winSize=(5, 5),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                )

            # 1. Raw ArUco pose estimation
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.MARKER_SIZE_CM, 
                self.camera_matrix, self.dist_coeffs
            )
            rot = R.from_rotvec(rvec[0].flatten())
            euler = rot.as_euler('xyz')
            raw_pose = np.concatenate([tvec[0].flatten(), euler])
            
            # 2. Depth-enhanced pose
            depth_z = self.get_depth_at_marker(corners[0])
            depth_enhanced_pose = raw_pose.copy()
            if depth_z is not None:
                depth_enhanced_pose[2] = depth_z * 100  # Convert meters to cm

            # 3. Kalman-filtered pose
            self.kf.predict()
            self.kf.update(raw_pose)
            kalman_pose = self.kf.x.flatten()

            # Publish poses
            self.publish_pose(raw_pose, self.raw_pose_pub)
            #self.publish_pose(depth_enhanced_pose, self.depth_enhanced_pose_pub)
            #self.publish_pose(kalman_pose, self.kalman_pose_pub)
            
            # Publish all markers (for multiple marker scenarios)
            poses_with_ids = [(raw_pose, ids[0][0])]  # Use Kalman-filtered pose as primary
            self.publish_all_markers(poses_with_ids)
            
            '''# Log the published data
            self.get_logger().info(f"Published marker {ids[0][0]} pose: "
                                 f"Pos({kalman_pose[0]:.2f}, {kalman_pose[1]:.2f}, {kalman_pose[2]:.2f})cm "
                                 f"Rot({np.degrees(kalman_pose[3]):.1f}, {np.degrees(kalman_pose[4]):.1f}, {np.degrees(kalman_pose[5]):.1f})°")
'''
            # Visualization
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.drawFrameAxes(
                cv_image, self.camera_matrix, self.dist_coeffs,
                rvec[0], tvec[0], self.MARKER_SIZE_CM / 2
            )

            # Display marker ID and orientation info on the image
            marker_center = corners[0][0].mean(axis=0).astype(int)
            cv2.putText(cv_image, f"ID: {ids[0][0]}", 
                       (marker_center[0] - 30, marker_center[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display orientation as Euler angles
            euler_deg = np.degrees(kalman_pose[3:6])
            cv2.putText(cv_image, f"Roll: {euler_deg[0]:.1f}°", 
                       (marker_center[0] - 50, marker_center[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(cv_image, f"Pitch: {euler_deg[1]:.1f}°", 
                       (marker_center[0] - 50, marker_center[1] + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(cv_image, f"Yaw: {euler_deg[2]:.1f}°", 
                       (marker_center[0] - 50, marker_center[1] + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Display all three measurement methods
            y_offset = 30
            methods = [
                ("Raw ArUco", raw_pose),
                ("Depth-Enhanced", depth_enhanced_pose),
                ("Kalman-Filtered", kalman_pose)
            ]
            
            for i, (name, pose) in enumerate(methods):
                t = pose[:3]
                r = np.degrees(pose[3:])
                
                # Method label
                cv2.putText(
                    text_bg, f"{name}:",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.method_colors[i], 1
                )
                y_offset += 20
                
                # Position
                cv2.putText(
                    text_bg, f"  Pos: {float(t[0]):.2f}, {float(t[1]):.2f}, {float(t[2]):.2f} cm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.method_colors[i], 1
                )
                y_offset += 20
                
                # Rotation
                cv2.putText(
                    text_bg, f"  Rot: {float(r[0]):.2f}, {float(r[1]):.2f}, {float(r[2]):.2f} deg",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.method_colors[i], 1
                )
                y_offset += 30

        # Combine images
        display_img = np.vstack([cv_image, text_bg])
        cv2.imshow("ARUCO Pose Estimation (Raw | Depth | Kalman)", display_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseEstimator()
    rclpy.spin(node)
    #node.destroy_node()
    #rclpy.shutdown()
    print("made aruco pose estimator")

if __name__ == '__main__':
    main()