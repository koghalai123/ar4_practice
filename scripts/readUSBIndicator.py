#!/usr/bin/env python3

import hid
import time
import sys
from datetime import datetime

def find_holtek_device():
    """Find the specific Holtek HID device"""
    target_vid = 0x04d9  # Holtek Semiconductor, Inc.
    target_pid = 0x1702  # Keyboard LKS02
    
    # List all HID devices
    devices = hid.enumerate()
    
    for device in devices:
        if device['vendor_id'] == target_vid and device['product_id'] == target_pid:
            return device
    
    return None

def main():
    """Simple USB HID reader for Holtek device"""
    
    # Find the specific Holtek device
    device_info = find_holtek_device()
    
    if device_info is None:
        print("Holtek LKS02 device not found!")
        print("Looking for: VID:PID = 04d9:1702")
        print("\nAvailable HID devices:")
        devices = hid.enumerate()
        for device in devices:
            print(f"  VID:PID = {device['vendor_id']:04x}:{device['product_id']:04x} - {device['manufacturer_string']} {device['product_string']}")
        sys.exit(1)
    
    # Connect to the Holtek device
    try:
        h = hid.device()
        h.open(device_info['vendor_id'], device_info['product_id'])
        
        print(f"Connected to Holtek LKS02:")
        print(f"  Manufacturer: {device_info['manufacturer_string']}")
        print(f"  Product: {device_info['product_string']}")
        print(f"  VID:PID = {device_info['vendor_id']:04x}:{device_info['product_id']:04x}")
        print("Reading HID data... (Press Ctrl+C to stop)")
        print("-" * 50)
        
        # Set non-blocking mode
        h.set_nonblocking(1)
        
        while True:
            # Read data from HID device
            data = h.read(64)  # Read up to 64 bytes
            
            if data:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                # Convert bytes to hex string for debugging
                hex_data = ' '.join([f'{b:02x}' for b in data])
                print(f"[{timestamp}] Raw: {hex_data}")
                
                # Try to interpret as text (you may need to adjust this based on your device)
                try:
                    text_data = ''.join([chr(b) for b in data if 32 <= b <= 126])
                    if text_data.strip():
                        print(f"[{timestamp}] Text: {text_data.strip()}")
                except:
                    pass
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        print("You may need to run with sudo or add udev rules for HID device access")
    finally:
        if 'h' in locals():
            h.close()
            print("Disconnected from Holtek device")

if __name__ == "__main__":
    main()