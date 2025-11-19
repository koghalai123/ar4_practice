#!/usr/bin/env python3

import serial
import serial.tools.list_ports
import time
import sys
import csv
from datetime import datetime

def find_prolific_device():
    """Find the specific Prolific PL2303 device"""
    # Look for the specific Prolific device by vendor/product ID
    target_vid = 0x067b  # Prolific Technology, Inc.
    target_pid = 0x2303  # PL2303 Serial Port
    
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Check if this port matches our target device
        if (hasattr(port, 'vid') and hasattr(port, 'pid') and 
            port.vid == target_vid and port.pid == target_pid):
            return port.device
        
        # Also check description for Prolific or PL2303
        if port.description and ('Prolific' in port.description or 'PL2303' in port.description):
            return port.device
    
    return None

def main():
    """Simple USB serial reader for Prolific PL2303 device"""
    
    # Create CSV filename with timestamp
    csv_filename = f"dial_indicator_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Find the specific Prolific device
    port = find_prolific_device()
    
    if port is None:
        print("Prolific PL2303 device not found!")
        print("Looking for: Bus 007 Device 003: ID 067b:2303 Prolific Technology, Inc. PL2303")
        print("\nAvailable ports:")
        for p in serial.tools.list_ports.comports():
            vid_pid = f"VID:PID = {p.vid:04x}:{p.pid:04x}" if p.vid and p.pid else "No VID/PID"
            print(f"  {p.device} - {p.description} ({vid_pid})")
        sys.exit(1)
    
    # Connect to the Prolific device
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        print(f"Connected to Prolific PL2303 on {port} at 9600 baud")
        print(f"Saving data to: {csv_filename}")
        print("Reading numeric data... (Press Ctrl+C to stop)")
        print("-" * 50)
        
        # Open CSV file for writing
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Reading'])  # Write header
            
            buffer = b""  # Buffer to accumulate data
            
            while True:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    buffer += data
                    
                    # Process complete readings (ending with \r)
                    while b'\r' in buffer:
                        line, buffer = buffer.split(b'\r', 1)
                        
                        # Skip the first null byte if present
                        if line.startswith(b'\x00'):
                            line = line[1:]
                        
                        if line:
                            try:
                                # Decode and convert to float
                                reading_str = line.decode('ascii', errors='ignore').strip()
                                if reading_str:
                                    reading_value = float(reading_str)
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                    display_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                    
                                    # Write to CSV
                                    csv_writer.writerow([timestamp, reading_value])
                                    csvfile.flush()  # Ensure data is written immediately
                                    
                                    # Display to console
                                    print(f"[{display_time}] {reading_value:+8.3f}")
                            except ValueError:
                                # Skip invalid readings
                                pass
                
                time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        print(f"Data saved to: {csv_filename}")
    except PermissionError:
        print(f"Permission denied accessing {port}")
        print("Try: sudo usermod -a -G dialout $USER")
        print("Then log out and back in.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
            print(f"Disconnected from {port}")

if __name__ == "__main__":
    main()