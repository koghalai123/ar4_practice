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
                R_marker, J = cv2.Rodrigues(rvecs[i])
                t_center = tvecs[i].reshape(3, 1) + R_marker @ displacementVec
                t_center = t_center.flatten()       
                #cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                 #       rvecs[i], t_center, self.marker_size * 0.5)
                
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
        #poi = optimize_poi(rvecs, tvecs, ids, marker_positions)
            x_opt, theta_opt = self.optimize_poses(rvecs, tvecs, ids)
        
            x_opt_2d, _ = cv2.projectPoints(x_opt.reshape(3, 1), np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs)
            cv2.circle(display_image, tuple(x_opt_2d[0][0].astype(int)), 5, (0, 0, 255), -1)

        
        #print("Optimized common point:", x_opt)
        #print("Angle corrections:", theta_opt)

        # Optional: Update rvecs with corrections
        #for i in range(len(ids)):
         #   rvecs[i] += theta_opt[i].reshape(3, 1)
        # Display the image
        cv2.imshow('Aruco Marker Detection', display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.shutdown()
            



    def optimize_poses(self, rvecs, tvecs, ids):
        """Solve for common point x and angle corrections theta_i."""
        N = len(ids)
        dim_x = 3
        dim_theta = 3

        # === Collect A_i and t_center_i ===
        A_list = []
        t_centers = []
        for i in range(N):
            # Compute displacementVec (as in your existing code)
            markerRow = np.floor(ids[i] / self.arucoCols)
            markerCol = np.mod(ids[i], self.arucoCols)
            rowVec = (self.arucoRows - 1)/2 - markerRow[0]
            colVec = (self.arucoCols - 1)/2 - markerCol[0]
            displacementVec = np.array([
                [(self.marker_size + self.marginSize) * colVec],
                [-(self.marker_size + self.marginSize) * rowVec],
                [0]
            ])

            # Get Jacobian J and compute A_i
            R_marker, J = cv2.Rodrigues(rvecs[i])
            J_reshaped = J.reshape(3, 3, 3)  # Shape: (3, 3, 3)
            A_i = np.tensordot(J_reshaped, displacementVec, axes=([2], [0]))  # Shape: (3, 3, 1)
            A_list.append(A_i.squeeze())  # Remove singleton dim to get (3, 3)

            # Compute t_center (as in your existing code)
            t_center = tvecs[i].reshape(3, 1) + R_marker @ displacementVec
            t_centers.append(t_center)

        # === Build Constraint Matrix C ===
        C_top = np.hstack([
            np.eye(dim_x),
            -A_list[0],  # Now shape (3, 3)
            np.zeros((dim_x, dim_theta * (N - 1)))
        ])
        C_middle = [
            np.hstack([
                np.eye(dim_x),
                np.zeros((dim_x, dim_theta * i)),
                -A_list[i],  # Shape (3, 3)
                np.zeros((dim_x, dim_theta * (N - i - 1)))
            ])
            for i in range(1, N)
        ]
        C = np.vstack([C_top] + C_middle)  # Shape: (3N, 3 + 3N)

        # === Stack t_centers ===
        T = np.vstack(t_centers)  # Shape: (3N, 1)

        # === Solve Least Squares (with regularization) ===
        lambda_val = 0.1  # Adjust based on trust in angles
        W = np.eye(dim_theta * N)  # Shape: (3N, 3N)

        # Pad W to match C's columns
        W_padded = np.zeros((W.shape[0], C.shape[1]))  # Shape: (3N, 3 + 3N)
        W_padded[:, dim_x:] = W  # Only penalize theta terms (last 3N columns)

        C_aug = np.vstack([C, np.sqrt(lambda_val) * W_padded])  # Shape: (6N, 3 + 3N)
        T_aug = np.vstack([T, np.zeros((dim_theta * N, 1))])  # Shape: (6N, 1)

        X = np.linalg.lstsq(C_aug, T_aug, rcond=None)[0]

        # === Extract Results ===
        x = X[:dim_x].flatten()  # Common point (3,)
        thetas = X[dim_x:].reshape(N, dim_theta)  # Angle corrections (N, 3)

        return x, thetas


    def image_callback2(self, msg):
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
                
                # ====== Define marker positions in the sheet frame ======
                marker_positions = {}
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    marker_row = np.floor(marker_id / self.arucoCols)
                    marker_col = marker_id % self.arucoCols
                    # Convert grid position to 3D coordinates (e.g., 10cm spacing)
                    x = (marker_col - (self.arucoCols - 1) / 2) * (self.marker_size + self.marginSize)
                    y = ((self.arucoRows - 1) / 2 - marker_row) * (self.marker_size + self.marginSize)
                    marker_positions[marker_id] = np.array([x, y, 0])  # Z=0 for flat sheet
                
                # ====== Collect t_center observations ======
                t_centers = []
                R_markers = []
                for i in range(len(ids)):
                    # Compute displacement vector
                    marker_row = np.floor(ids[i] / self.arucoCols)
                    marker_col = ids[i] % self.arucoCols
                    row_vec = (self.arucoRows - 1) / 2 - marker_row
                    col_vec = (self.arucoCols - 1) / 2 - marker_col
                    displacement_vec = np.array([
                        [(self.marker_size + self.marginSize) * col_vec],
                        [-(self.marker_size + self.marginSize) * row_vec],
                        [0]
                    ])
                    
                    # Compute t_center (POI in camera frame)
                    R_marker, _ = cv2.Rodrigues(rvecs[i])
                    t_center = tvecs[i].reshape(3, 1) + R_marker @ displacement_vec
                    t_centers.append(t_center.flatten())
                    R_markers.append(R_marker)
                    
                    # Draw axes (existing code)
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                                    rvecs[i], tvecs[i], self.marker_size * 0.5)
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs,
                                    rvecs[i], t_center.flatten(), self.marker_size * 0.5)
                    
                    # Publish poses (existing code)
                    self.publish_pose(rvecs[i], tvecs[i], ids[i][0])

            # ====== Optimize POI ======
            def optimize_poi(R_markers, t_centers, marker_positions):
                """
                Least-squares optimization for the Point of Interest (POI).
                Args:
                    R_markers: List of rotation matrices for each marker.
                    t_centers: List of t_center observations (camera frame).
                    marker_positions: Dict {id: [x, y, z]} of marker positions in the sheet frame.
                Returns:
                    Optimized POI in the sheet frame.
                """
                def residuals(p):
                    res = []
                    for i, id_ in enumerate(ids):
                        R_i = R_markers[i]
                        t_i = t_centers[i].reshape(3, 1)
                        p_marker = marker_positions[id_[0]].reshape(3, 1)
                        res.append(t_i - (R_i @ p.reshape(3, 1) + p_marker))
                    return np.vstack(res).flatten()

                from scipy.optimize import least_squares
                result = least_squares(
                    fun=residuals,
                    x0=np.zeros(3),
                    method='lm'
                )
                return result.x

            poi = optimize_poi(R_markers, t_centers, marker_positions)
            self.get_logger().info(f"Optimized POI: {poi}")

            # ====== Visualize POI (project into image) ======
            # Use the first marker's pose to project POI
            R_first = R_markers[0]
            t_first = t_centers[0].reshape(3, 1)
            poi_camera = R_first @ poi.reshape(3, 1) + t_first
            poi_pixel, _ = cv2.projectPoints(
                poi_camera, np.zeros(3), np.zeros(3),
                self.camera_matrix, self.dist_coeffs
            )
            cv2.circle(display_image, tuple(poi_pixel[0][0].astype(int)), 10, (0, 0, 255), -1)

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
        #self.get_logger().info(f"Detected Marker {marker_id} at {tvec[0]}")
        

    



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