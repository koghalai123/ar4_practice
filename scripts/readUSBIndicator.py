#!/usr/bin/env python3

import struct
import os
import fcntl
import sys
import select
import evdev
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Point




#device 3 is x
#device 2 is -z
#device 1 is -y



# Linux input ioctl constants
EVIOCGRAB = 0x40044590



# Numpad key codes mapping
NUMPAD_KEYS = {
    79: "1", 80: "2", 81: "3",
    75: "4", 76: "5", 77: "6", 
    71: "7", 72: "8", 73: "9",
    82: "0", 83: ".",  # Period
    96: "Enter", 78: "+", 74: "-", 55: "*", 98: "/"
}

class NumberBuffer:
    def __init__(self):
        self.buffer = []
        self.decimal_entered = False

        self.last_number = None

    def add_key(self, key_char):
        """Add a key to the buffer"""
        if key_char == '.':
            if not self.decimal_entered:
                self.buffer.append('.')
                self.decimal_entered = True
                return True
            return False
        else:
            self.buffer.append(key_char)
            return True
    
    def get_number(self):
        """Convert buffer to a decimal number WITHOUT clearing buffer"""
        if not self.buffer:
            return None
        
        number_str = ''.join(self.buffer)
        try:
            number = float(number_str)
            return number
        except ValueError:
            print(f"Warning: Could not convert '{number_str}' to number")
            return None
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.decimal_entered = False
    
    def is_empty(self):
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def get_current_input(self):
        """Get current input as string for display"""
        return ''.join(self.buffer) if self.buffer else ""

class KeyboardDevice:
    def __init__(self, device_path, device_id="", ros_node=None):
        self.device_path = device_path
        self.device_id = device_id
        self.device_file = None
        self.number_buffer = NumberBuffer()
        self.is_grabbed = False
        self.ros_node = ros_node
    
    def open_device(self):
        """Open and grab the keyboard device"""
        try:
            self.device_file = open(self.device_path, 'rb')
            self.grab_device()
            if self.ros_node:
                self.ros_node.get_logger().info(f"Device {self.device_id} ({self.device_path}): Ready")
            return True
        except (PermissionError, FileNotFoundError) as e:
            if self.ros_node:
                self.ros_node.get_logger().error(f"Device {self.device_id} ({self.device_path}): {e}")
            return False
    
    def grab_device(self):
        """Grab the input device to intercept events"""
        try:
            fcntl.ioctl(self.device_file, EVIOCGRAB, 1)
            self.is_grabbed = True
            return True
        except OSError as e:
            if self.ros_node:
                self.ros_node.get_logger().warn(f"Device {self.device_id}: Failed to grab device - {e}")
            return False
    
    def release_device(self):
        """Release the input device"""
        if self.device_file and self.is_grabbed:
            try:
                fcntl.ioctl(self.device_file, EVIOCGRAB, 0)
                self.is_grabbed = False
            except OSError:
                pass
    
    def close_device(self):
        """Close the device file"""
        self.release_device()
        if self.device_file:
            self.device_file.close()
            self.device_file = None
    
    def read_event(self):
        """Read a single event from the device. Returns True if event was processed."""
        if not self.device_file:
            return False
        
        try:
            event_data = self.device_file.read(24)
            if not event_data:
                return False
            
            # Unpack the event data (format: llHHI)
            (tv_sec, tv_usec, event_type, event_code, value) = struct.unpack('llHHI', event_data)
            
            # Process key press events only
            if event_type == 0x01 and value == 1:  # Key press
                if event_code in NUMPAD_KEYS:
                    key_char = NUMPAD_KEYS[event_code]
                    
                    if key_char == "Enter":
                        # Get the current input BEFORE clearing
                        current_input = self.number_buffer.get_current_input()
                        number = self.number_buffer.get_number()
                        self.last_number = number
                        # Publish the completed number BEFORE clearing
                        if self.ros_node and number is not None:
                            self.ros_node.publish_input(self.device_id, number, current_input)
                            self.ros_node.get_logger().info(f"Device {self.device_id}: Input = {current_input} -> {number}")
                        
                        # Clear buffer AFTER publishing
                        self.number_buffer.clear()
                    else:
                        # Add digit or decimal to buffer
                        if self.number_buffer.add_key(key_char):
                            current_input = self.number_buffer.get_current_input()
                    
                    return True
            
            return False
            
        except (OSError, struct.error):
            return False
    
    def get_file_descriptor(self):
        """Get the file descriptor for select()"""
        if self.device_file:
            return self.device_file.fileno()
        return None

class USBIndicatorNode(Node):
    def __init__(self):
        super().__init__('usb_indicator_node')
        
        # Publishers for each device
        self.device_publishers = {}
        self.partial_input_publishers = {}
        
        # General publishers
        self.all_inputs_pub = self.create_publisher(String, 'usb_indicator/all_inputs', 10)
        self.devices_status_pub = self.create_publisher(String, 'usb_indicator/devices_status', 10)
        
        # Device manager
        self.devices = {}
        
        # Timer for non-blocking device monitoring
        self.timer = self.create_timer(0.01, self.monitor_devices)  # 10ms timer
        
        self.get_logger().info('USB Indicator Node started')
    
    def add_device(self, device_path, device_id=None):
        """Add a keyboard device to monitor"""
        if device_id is None:
            device_id = f"device_{len(self.devices) + 1}"
        
        device = KeyboardDevice(device_path, device_id, self)
        if device.open_device():
            self.devices[device_id] = device
            
            # Create publishers for this device
            self.device_publishers[device_id] = self.create_publisher(
                Float64, f'usb_indicator/{device_id}/input', 10
            )
            
            self.get_logger().info(f'Added device: {device_id} at {device_path}')
            self.publish_device_status()
            return True
        return False
    
    def remove_device(self, device_id):
        """Remove a keyboard device"""
        if device_id in self.devices:
            self.devices[device_id].close_device()
            del self.devices[device_id]
            
            # Clean up publishers
            if device_id in self.device_publishers:
                del self.device_publishers[device_id]
            if device_id in self.partial_input_publishers:
                del self.partial_input_publishers[device_id]
            
            self.get_logger().info(f'Removed device: {device_id}')
            self.publish_device_status()
    
    def publish_input(self, device_id, number, input_string):
        """Publish completed input from a device"""
        # Publish to device-specific topic
        if device_id in self.device_publishers:
            msg = Float64()
            msg.data = number
            self.device_publishers[device_id].publish(msg)
        
        # Publish to general topic
        general_msg = String()
        general_msg.data = f"{device_id}: {input_string} -> {number}"
        self.all_inputs_pub.publish(general_msg)
    
    def publish_device_status(self):
        """Publish current device status"""
        status_msg = String()
        device_list = list(self.devices.keys())
        status_msg.data = f"Active devices: {device_list}"
        self.devices_status_pub.publish(status_msg)
    
    def monitor_devices(self):
        """Monitor all devices for input (called by timer)"""
        if not self.devices:
            return
        
        # Use select to monitor multiple file descriptors
        fds = [device.get_file_descriptor() for device in self.devices.values()]
        fds = [fd for fd in fds if fd is not None]
        
        if not fds:
            return
        
        # Non-blocking select
        try:
            ready, _, _ = select.select(fds, [], [], 0.0)  # Non-blocking
            
            # Process devices that have data ready
            for device in self.devices.values():
                fd = device.get_file_descriptor()
                if fd in ready:
                    device.read_event()
        except (OSError, ValueError):
            # Handle device disconnection
            pass
    
    def cleanup(self):
        """Clean up all devices"""
        for device_id, device in list(self.devices.items()):
            self.get_logger().info(f'Closing {device_id}...')
            device.close_device()
        self.devices.clear()

def connect_to_target_device(target_name="XAOJU LABORATORIES C8051F3xx Development"):
    """
    Automatically finds and connects to input devices by name.

    Args:
        target_name (str): The name of the device to connect to.

    Returns:
        list: List of evdev.InputDevice objects that match the target name.
    """
    # Get list of all input device paths
    device_paths = evdev.list_devices()
    found_devices = []
    
    # Create InputDevice objects and filter by the target name
    for path in device_paths:
        try:
            device = evdev.InputDevice(path)
            if device.name == target_name:
                print(f"Found target device at {device.path}")
                found_devices.append(device)
        except (OSError, IOError):
            # Handle devices that become unavailable during listing
            continue
    
    if not found_devices:
        print(f"Target device '{target_name}' was not found.")
    else:
        print(f"Found {len(found_devices)} matching device(s)")
    
    return found_devices

def main(args=None):
    rclpy.init(args=args)
    
    node = USBIndicatorNode()
    
    try:
        # Connect to target devices
        target_devices = connect_to_target_device()
        
        # Connect to all found target devices
        if target_devices:
            for i, device in enumerate(target_devices):
                device_id = f"target_device_{i+1}" if len(target_devices) > 1 else "target_device"
                node.add_device(device.path, device_id)
        
        # If no target devices found, try common event devices as fallback
        if not target_devices:
            # Try common event devices
            for i in range(4):
                device_path = f"/dev/input/event{i}"
                if os.path.exists(device_path):
                    if node.add_device(device_path, f"event{i}"):
                        node.get_logger().info(f"Added {device_path}")
        
        # If still no devices, use the default
        if not node.devices:
            if node.add_device("/dev/input/event3", "default"):
                node.get_logger().info("Added default device event3")
            else:
                node.get_logger().error("No usable keyboard devices found")
                return
        
       #device status
        node.publish_device_status()
        
        # Start the ROS2 node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down...')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()