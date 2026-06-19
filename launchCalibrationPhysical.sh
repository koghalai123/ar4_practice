ros2 run realsense2_camera realsense2_camera_node
python /home/koghalai/ar4_ws/src/ar4_practice/scripts/arucoTesting.py
# Use the stable by-id path so the driver always finds THIS Teensy regardless of
# which /dev/ttyACMx number it re-enumerates as after a USB disconnect/reconnect.
ros2 launch annin_ar4_driver driver.launch.py serial_port:=/dev/serial/by-id/usb-Teensyduino_USB_Serial_16869110-if00 calibrate:=True include_gripper:=False
sleep 2
#ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=False