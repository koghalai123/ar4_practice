from rclpy.node import Node

from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Point
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
import rclpy



class SurfacePublisher(Node):
    def __init__(self):
        super().__init__('surface_publisher')
        self.marker_pub = self.create_publisher(MarkerArray, 'surface_marker', 10)

    '''def publishMarkers(self):
        
        marker_array = MarkerArray()
        marker_array.markers.append(self.workPlane)
        marker_array.markers.append(self.sphereList)
        self.marker_pub.publish(marker_array)'''
    def publishPlane(self, sideLength = 0.5, pos = np.array([0, 0, 0]), euler = np.array([0, 0, 0]), id=0, 
                     color = np.array([0.0, 0.0, 0.0])):

        marker = Marker()
        marker.header.frame_id = "base_link"  # Change to your frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "plane"
        marker.id = id
        
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        # Set pose (identity means the marker will appear exactly as specified in points)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        

        cubeSideLength = sideLength
        cubeCenter = pos

        # Create plane points in local coordinates (before rotation)
        planePoints_local = np.array([
            [-cubeSideLength[0]/2, -cubeSideLength[0]/2, 0],
            [cubeSideLength[0]/2, -cubeSideLength[0]/2, 0],
            [-cubeSideLength[0]/2, cubeSideLength[0]/2, 0],
            [cubeSideLength[0]/2, cubeSideLength[0]/2, 0],
            [0,0,0],
        ])
        
        # Apply rotation using Euler angles
        if np.any(euler != 0):
            # Create rotation matrix from Euler angles (roll, pitch, yaw)
            rotation = R.from_euler('xyz', euler, degrees=False)
            # Apply rotation to each point
            planePoints_rotated = rotation.apply(planePoints_local)
        else:
            planePoints_rotated = planePoints_local
        
        # Translate to final position
        planePoints = planePoints_rotated + cubeCenter
        planeFaces  = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ])
        planePoints_as_points = [Point(x=p[0], y=p[1], z=p[2]) for p in planePoints]
        for i in range(0, len(planeFaces)):
            marker.points.append(planePoints_as_points[planeFaces[i,0]])
            marker.points.append(planePoints_as_points[planeFaces[i,1]])
            marker.points.append(planePoints_as_points[planeFaces[i,2]])

        
        
        # Set color (RGBA, 0-1)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Semi-transparent
        
        # Scale (irrelevant for TRIANGLE_LIST, but required)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        self.workPlane = marker


        # 3. Create SPHERE_LIST marker
        sphere_list_marker = Marker()
        sphere_list_marker.header.frame_id = "base_link"
        sphere_list_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_list_marker.ns = "work_envelope"  # Same namespace as others
        sphere_list_marker.id = id  # Unique ID
        sphere_list_marker.type = Marker.SPHERE_LIST
        sphere_list_marker.action = Marker.ADD

        # Set appearance
        sphere_list_marker.scale.x = 0.03  # Diameter of each sphere
        sphere_list_marker.scale.y = 0.03  # Must be same as x for spheres
        sphere_list_marker.scale.z = 0.03  # Must be same as x for spheres
        sphere_list_marker.color.a = 1.0  # Fully opaque
        sphere_list_marker.color.r = 1.0  # Red
        sphere_list_marker.color.g = 0.0
        sphere_list_marker.color.b = 0.0


        for point in planePoints_as_points:
            sphere_list_marker.points.append(point)
        self.sphereList = sphere_list_marker

        marker_array = MarkerArray()
        marker_array.markers.append(self.workPlane)
        marker_array.markers.append(self.sphereList)
        self.marker_pub.publish(marker_array)

    def publish_arrow(self, position=np.array([0, 0, 0]), orientation=np.array([0, 0, 0]), 
                  length=0.2, thickness=0.02, id=0, color=np.array([0.0, 1.0, 0.0])):
        """
        Publish an arrow marker to RViz with specified position and orientation
        
        Args:
            position: np.array([x, y, z]) - Position of the arrow's base
            orientation: np.array([roll, pitch, yaw]) - Orientation in euler angles (radians)
            length: float - Length of the arrow
            id: int - Unique marker ID
            color: np.array([r, g, b]) - RGB color (0-1 range)
        """

        '''marker_publisher.publish_arrow(
            position=np.array([endEffectorPosWeirdFrame[0], endEffectorPosWeirdFrame[1], endEffectorPosWeirdFrame[2]]),  # Position of arrow's base
            #orientation=np.array([endEffectorOrientWeirdFrame[0], endEffectorOrientWeirdFrame[1], endEffectorOrientWeirdFrame[2]]),     # Default orientation (points along x-axis)
            orientation = np.array([-(pitch+np.pi/2),-roll,+yaw+np.pi/2]),  # Adjust orientation to point towards target
            length=np.linalg.norm(vectorToTarget),  
            thickness=0.01,                 # 30cm long arrow
            id=10,                              # Unique ID for this arrow
            color=np.array([1.0, 0.0, 0.0])     # Red color
        )'''
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "arrows"
        marker.id = id
        
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        
        # Set orientation from euler angles
        quat = quaternion_from_euler(orientation[0], orientation[1], orientation[2],axes='szxy')
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        # Set the scale - controls the size of the arrow
        marker.scale.x = length      # Shaft diameter
        marker.scale.y = thickness*2.0  # Head diameter
        marker.scale.z = thickness*3.0  # Head length
        
        
        
        # Set color (RGBA, 0-1)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque
        
        # Create a marker array if it doesn't exist
        if not hasattr(self, 'arrow_markers'):
            self.arrow_markers = MarkerArray()
        
        # Add or update the marker in the array
        # First check if marker with this id already exists in the array
        found = False
        for i, m in enumerate(self.arrow_markers.markers):
            if m.id == id and m.ns == "arrows":
                self.arrow_markers.markers[i] = marker
                found = True
                break
        
        if not found:
            self.arrow_markers.markers.append(marker)
        
        # Publish the marker array
        self.marker_pub.publish(self.arrow_markers)
        
        return marker
    
    def publish_arrow_between_points(self, start=np.array([0, 0, 0]), end=np.array([1, 0, 0]), 
                               thickness=0.02, id=0, color=np.array([0.0, 1.0, 0.0])):
        """
        Publish an arrow marker to RViz between two points
    
        Args:
            start: np.array([x, y, z]) - Starting point of the arrow
            end: np.array([x, y, z]) - Ending point of the arrow
            thickness: float - Thickness of the arrow shaft
            id: int - Unique marker ID
            color: np.array([r, g, b]) - RGB color (0-1 range)
        """
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "arrows_between_points"
        marker.id = id
    
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
    
        # For arrows between points, we use the points field instead of pose
        marker.points.append(Point(x=start[0], y=start[1], z=start[2]))
        marker.points.append(Point(x=end[0], y=end[1], z=end[2]))
    
        # Set the scale - controls the size of the arrow
        marker.scale.x = thickness      # Shaft diameter
        marker.scale.y = thickness*2.0  # Head diameter
        marker.scale.z = thickness*3.0  # Head length
    
        # Set color (RGBA, 0-1)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque
    
        # Create a marker array if it doesn't exist
        if not hasattr(self, 'arrows_between_points_markers'):
            self.arrows_between_points_markers = MarkerArray()
    
        # Add or update the marker in the array
        found = False
        for i, m in enumerate(self.arrows_between_points_markers.markers):
            if m.id == id and m.ns == "arrows_between_points":
                self.arrows_between_points_markers.markers[i] = marker
                found = True
                break
    
        if not found:
            self.arrows_between_points_markers.markers.append(marker)
    
        # Publish the marker array
        self.marker_pub.publish(self.arrows_between_points_markers)
    
        return marker