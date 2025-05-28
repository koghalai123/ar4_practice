#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Aruco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()
        
        # Camera parameters - REPLACE WITH YOUR CAMERA'S CALIBRATION
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.marker_size = 0.05  # 5cm = 50mm
        self.arucoRows = 7
        self.arucoCols = 5
        self.marginSize= 0.050 #5 cm = 50 mm

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.GridBoard(
        size=(self.arucoRows, self.arucoCols),  # Number of markers in (rows, cols)
        markerLength=self.marker_size,  # Length of one marker side (meters)
        markerSeparation=self.marginSize,  # Distance between markers (meters)
        dictionary=dictionary
        )
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # for D435i
            # '/camera/camera/color/image_rect_raw' #for d405
            self.image_callback,
            10)
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/aruco_marker_pose',
            10)
        
        # Create OpenCV window
        cv2.namedWindow('Aruco Marker Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Aruco Marker Detection', 800, 600)
        
        self.get_logger().info("Aruco Marker Detector node initialized - Press 'q' to quit")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            display_image = cv_image.copy()

            # Detect markers
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                # Estimate pose for each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                
                # Draw detected markers and axes
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
                
                for i in range(len(ids)):
                    # Draw axis at the center
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                                    rvecs[i], tvecs[i], self.marker_size * 0.5)
                    
                    # ====== NEW: Draw axis at the center of the marker ======
                    # Compute the center pose (shift by half marker size along x and y)
                    boardRotation = np.mean(rvecs,axis=0)
                    markerRow = np.floor(np.divide(ids,self.arucoCols))
                    markerCol = np.mod(ids,self.arucoCols)

                    rowVec = (self.arucoRows-1)/2 - markerRow[i][0]
                    colVec = (self.arucoCols-1)/2 - markerCol[i][0]

                    displacementVec = np.array([[(self.marker_size+self.marginSize)*colVec], [-(self.marker_size+self.marginSize)*rowVec], [0]])
                    R_marker, _ = cv2.Rodrigues(rvecs[i])
                    t_center = tvecs[i].reshape(3, 1) + R_marker @ displacementVec
                    t_center = t_center.flatten()       
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                            rvecs[i], t_center, self.marker_size * 0.5)
                    
                    '''#draw axes at corners
                    R_marker, _ = cv2.Rodrigues(rvecs[i])
                    t_center = tvecs[i].reshape(3, 1) + R_marker @ np.array([[self.marker_size/2], [self.marker_size/2], [0]])
                    t_center = t_center.flatten()       
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                                    rvecs[i], t_center, self.marker_size * 0.5)'''
                    
                    # Calculate distance
                    distance = np.linalg.norm(tvecs[i])
                    
                    # Get marker center for text placement
                    center = corners[i][0].mean(axis=0)
                    
                    # Display marker info
                    cv2.putText(display_image, f"ID: {ids[i][0]}", 
                                (int(center[0]) - 30, int(center[1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Dist: {distance:.2f}m", 
                                (int(center[0]) - 30, int(center[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    self.publish_pose(rvecs[i], tvecs[i], ids[i][0])
                
            # Display the image
            cv2.imshow('Aruco Marker Detection', display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.shutdown()
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def publish_pose(self, rvec, tvec, marker_id):
        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Convert to quaternion
        rot = R.from_matrix(rotation_matrix)
        quat = rot.as_quat()
        
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "camera_color_optical_frame"
        pose_msg.pose.position.x = tvec[0][0]
        pose_msg.pose.position.y = tvec[0][1]
        pose_msg.pose.position.z = tvec[0][2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        
        
        
        self.pose_pub.publish(pose_msg)
        self.get_logger().info(f"Detected Marker {marker_id} at {tvec[0]}")
        
        

    def shutdown(self):
        cv2.destroyAllWindows()
        self.get_logger().info("Shutting down Aruco Detector")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    
    try:
        rclpy.spin(aruco_detector)
    except KeyboardInterrupt:
        pass
    finally:
        aruco_detector.shutdown()

if __name__ == '__main__':
    main()