import sympy as sp

# Define symbolic variables for Euler angles
roll, pitch, yaw = sp.symbols('roll pitch yaw')

# Define rotation matrices for each axis
Rx = sp.Matrix([
    [1, 0, 0],
    [0, sp.cos(roll), -sp.sin(roll)],
    [0, sp.sin(roll), sp.cos(roll)]
])

Ry = sp.Matrix([
    [sp.cos(pitch), 0, sp.sin(pitch)],
    [0, 1, 0],
    [-sp.sin(pitch), 0, sp.cos(pitch)]
])

Rz = sp.Matrix([
    [sp.cos(yaw), -sp.sin(yaw), 0],
    [sp.sin(yaw), sp.cos(yaw), 0],
    [0, 0, 1]
])

# Combine the rotations in XYZ order
rotation_matrix_xyz = Rz * Ry * Rx

# Display the symbolic rotation matrix
sp.pprint(rotation_matrix_xyz)