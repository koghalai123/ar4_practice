#!/usr/bin/env python3

import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ar4_robot_py import AR4Robot

def main():
    """Test logging control functionality"""
    try:
        print("=== Testing AR4Robot Logging Control ===")
        
        # Create robot interface
        robot = AR4Robot()
        
        print("\n1. Testing with logging ENABLED (default):")
        robot.print_current_state("Test with logging enabled")
        
        print("\n2. Disabling logging...")
        robot.disable_logging()
        
        print("\n3. Testing with logging DISABLED:")
        robot.print_current_state("Test with logging disabled")
        
        print("\n4. Re-enabling logging...")
        robot.enable_logging()
        
        print("\n5. Testing with logging RE-ENABLED:")
        robot.print_current_state("Test with logging re-enabled")
        
        print("\n=== Logging Control Test Completed ===")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        if 'robot' in locals():
            robot.shutdown()

if __name__ == '__main__':
    main()
