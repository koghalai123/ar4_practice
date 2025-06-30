#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter

class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator')
        
        # ROS2 Subscribers
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10)
        self.bridge = CvBridge()
        
        # Depth data storage
        self.depth_image = None
        self.depth_scale = 0.001  # Default scale (assuming depth is in mm)

        # ARUCO Setup
        self.MARKER_SIZE_CM = 14.6
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
                depth = self.depth_image[v, u] * self.depth_scale
                if depth > 0:  # Valid measurement
                    depths.append(depth)
        
        return np.median(depths) if depths else None

    def format_pose(self, t, r, prefix=""):
        """Format pose for display"""
        return (f"{prefix}X: {float(t[0]):.1f}cm Y: {float(t[1]):.1f}cm Z: {float(t[2]):.1f}cm\n"
                f"{prefix}Roll: {float(r[0]):.1f}° Pitch: {float(r[1]):.1f}° Yaw: {float(r[2]):.1f}°")

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
            self.kf.update(depth_enhanced_pose)
            kalman_pose = self.kf.x

            # Visualization
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.drawFrameAxes(
                cv_image, self.camera_matrix, self.dist_coeffs,
                rvec[0], tvec[0], self.MARKER_SIZE_CM / 2
            )

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
                    text_bg, f"  Pos: {float(t[0]):.1f}, {float(t[1]):.1f}, {float(t[2]):.1f} cm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.method_colors[i], 1
                )
                y_offset += 20
                
                # Rotation
                cv2.putText(
                    text_bg, f"  Rot: {float(r[0]):.1f}, {float(r[1]):.1f}, {float(r[2]):.1f} deg",
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
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()