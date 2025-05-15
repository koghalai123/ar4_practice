#!/usr/bin/env python3
import sympy as sp
from urdf_parser_py.urdf import URDF
from xacro import process_file
import os
import numpy as np
from functools import partial
import csv

roundSym = partial(round, ndigits=4)

def save_matrix_to_csv(matrix, filename, joint_name=None):
    """
    Saves a SymPy matrix to CSV.
    - matrix: SymPy matrix to save
    - filename: Output CSV file
    - joint_name: Optional joint name for labeling
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Add joint name as header (if provided)
        if joint_name:
            writer.writerow([f"Transformation Matrix for Joint: {joint_name}"])
        
        # Write matrix row by row
        for row in matrix.tolist():
            writer.writerow([str(sp.simplify(element)) for element in row])



# ==================================================================
# STEP 1: Load and parse the URDF/XACRO file
# ==================================================================
# Path to your AR4 robot's XACRO file
xacro_path = os.path.expanduser(
    "~/ar4_ws/src/ar4_ros_driver/annin_ar4_description/urdf/ar.urdf.xacro"
)

# Process XACRO into URDF (resolve macros and parameters)
urdf_string = process_file(
    xacro_path,
    mappings={
        "ar_model": "mk3",          # Replace with your model (mk1/mk2/mk3)
        "include_gripper": "true"   # Include gripper if needed
    }
).toxml()

# Parse URDF
robot = URDF.from_xml_string(urdf_string)

# ==================================================================
# STEP 2: Define symbolic joint variables
# ==================================================================
joint_vars = []
for joint in robot.joints:
    if joint.type in ["revolute", "prismatic"]:
        joint_vars.append(sp.symbols(f"q_{joint.name}"))

# ==================================================================
# STEP 3: Build symbolic transformation matrices
# ==================================================================
T_total = sp.eye(4)  # Start with identity matrix
counter = 1
for joint in robot.joints:
    if joint.type == "fixed":
        continue  # Skip fixed joints (no movement)

    # Get joint origin (translation + rotation)
    xyz = joint.origin.xyz if joint.origin else [0, 0, 0]
    rpy = joint.origin.rpy if joint.origin else [0, 0, 0]

    # Build translation matrix
    T_trans = sp.Matrix([
        [1, 0, 0, xyz[0]],
        [0, 1, 0, xyz[1]],
        [0, 0, 1, xyz[2]],
        [0, 0, 0, 1]
    ])

    # Build rotation matrix from RPY angles
    cr, sr = sp.cos(rpy[0]), sp.sin(rpy[0])
    cp, sp_ = sp.cos(rpy[1]), sp.sin(rpy[1])
    cy, sy = sp.cos(rpy[2]), sp.sin(rpy[2])

    R_rpy = sp.Matrix([
        [cy*cp, cy*sp_*sr - sy*cr, cy*sp_*cr + sy*sr],
        [sy*cp, sy*sp_*sr + cy*cr, sy*sp_*cr - cy*sr],
        [-sp_, cp*sr, cp*cr]
    ])

    # Combine translation and rotation

    T_joint = T_trans * sp.Matrix(sp.BlockMatrix([[R_rpy, sp.zeros(3, 1)], [sp.zeros(1, 3), sp.Matrix([[1]])]])).applyfunc(roundSym)

    # Add joint-specific transform (revolute/prismatic)
    if joint.type == "revolute":
        theta = joint_vars.pop(0)
        axis = joint.axis if joint.axis else [0, 0, 1]  # Default axis if not specified
        
        # Rodrigues' rotation formula for arbitrary axis
        K = sp.Matrix([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_joint = sp.eye(3) + sp.sin(theta) * K + (1 - sp.cos(theta)) * K**2
        T_joint = T_joint * sp.Matrix(sp.BlockMatrix([[R_joint, sp.zeros(3, 1)], [sp.zeros(1, 3), sp.Matrix([[1]])]]))

    elif joint.type == "prismatic":
        d = joint_vars.pop(0)
        axis = joint.axis if joint.axis else [0, 0, 1]
        T_joint = T_joint * sp.Matrix([
            [1, 0, 0, d * axis[0]],
            [0, 1, 0, d * axis[1]],
            [0, 0, 1, d * axis[2]],
            [0, 0, 0, 1]
        ])
    # Accumulate transforms
    save_matrix_to_csv(T_joint, "Joint"+str(counter)+".csv", joint_name="Joint"+str(counter))
    counter = counter + 1
    T_total = T_total * T_joint
save_matrix_to_csv(T_total, "FullForwardKinematics.csv", joint_name="Full Forward Kinematics")
# Simplify the final transform
#T_total = sp.simplify(T_total)

# ==================================================================
# STEP 4: Extract position and orientation
# ==================================================================
# Position (x, y, z)
x = T_total[0, 3]
y = T_total[1, 3]
z = T_total[2, 3]

# Orientation (RPY angles)
R = T_total[:3, :3]
roll = sp.atan2(R[2, 1], R[2, 2])
pitch = sp.atan2(-R[2, 0], sp.sqrt(R[2, 1]**2 + R[2, 2]**2))
yaw = sp.atan2(R[1, 0], R[0, 0])

# ==================================================================
# STEP 5: Print results
# ==================================================================
print("=" * 50)
print("Symbolic Forward Kinematics Matrix:")
sp.pprint(T_total)

print("\nEnd-Effector Position:")
print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")

print("\nEnd-Effector Orientation (RPY):")
print(f"roll = {roll}")
print(f"pitch = {pitch}")
print(f"yaw = {yaw}")
print("=" * 50)

print('done')
