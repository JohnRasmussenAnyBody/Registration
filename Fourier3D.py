# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 11:07:01 2024

@author: jr
"""

import numpy as np
import trimesh as tri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert complex Fourier coefficients to real cosine and sine coefficients.

# Parameters:
# F (numpy.ndarray): Complex Fourier coefficients.

# Returns:
# A (numpy.ndarray): Cosine coefficients (real part).
# B (numpy.ndarray): Sine coefficients (imaginary part).
def complex_to_real_coefficients(F):

    # Compute the amplitude and phase
    amplitude = np.abs(F)
    phase = np.angle(F)

    # Compute the cosine and sine coefficients
    A = amplitude * np.cos(phase)
    B = amplitude * np.sin(phase)

    return A, B


# Convert real cosine and sine coefficients to complex Fourier coefficients.

# Parameters:
# A (numpy.ndarray): Cosine coefficients (real part).
# B (numpy.ndarray): Sine coefficients (imaginary part).

# Returns:
# F (numpy.ndarray): Complex Fourier coefficients.

def real_to_complex_coefficients(A, B):

    # Compute the amplitude and phase
    amplitude = np.sqrt(A**2 + B**2)
    phase = np.arctan2(B, A)

    # Compute the complex Fourier coefficients
    F = amplitude * (np.cos(phase) + 1j * np.sin(phase))

    return F

"""
Compute the function value at the spatial point (x, y, z) using the real 
coefficients A and B.

Parameters:
A (numpy.ndarray): 3D array of cosine coefficients.
B (numpy.ndarray): 3D array of sine coefficients.
x, y, z (float): Spatial coordinates where the function value is to be computed.
Lx, Ly, Lz (float): Dimensions of the spatial domain.

Returns:
float: Function value at the specified spatial point.
"""
def interpolate_d(A, B, x, y, z, kx, ky, kz):

    # Get the shape of the coefficient arrays
    nx, ny, nz = A.shape
        
    # Initialize the result
    d = 0.0
    
    # Sum the contributions of all Fourier modes
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                phase = 2 * np.pi * (kx[i] * x + ky[j] * y + kz[k] * z)
                d += A[i, j, k] * np.cos(phase) + B[i, j, k] * np.sin(phase)
    
    return d

# Load the mesh
bone = tri.load_mesh('bone.obj')

# Align the mesh with the global coordinate system
A = bone.principal_inertia_transform
bone.apply_transform(A)
# bone.visual.vertex_colors = [  2, 66,  236, 255]
bone.visual.vertex_colors = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), 255]

# Find the bounding box
bbox = np.array(bone.bounding_box.bounds)

# Translate bone to prepare for antisymmetric Fourier transform
trans = bbox[1]
bone.apply_translation(-trans)
bbox[0] -= trans
bbox[1] -= trans

# Double the box
brange = bbox[1]-bbox[0]
bbox[1] += brange

# Side lengths
Lx = bbox[1,0] - bbox[0,0]
Ly = bbox[1,1] - bbox[0,1]
Lz = bbox[1,2] - bbox[0,2]

# Create a 3D array (for example, a simple 3D Gaussian blob)
gridsize = 8  # Should be a power of 2
gridx = np.linspace(bbox[0,0], bbox[1,0], gridsize)
gridy = np.linspace(bbox[0,1], bbox[1,1], gridsize)
gridz = np.linspace(bbox[0,2], bbox[1,2], gridsize)
gridx, gridy, gridz = np.meshgrid(gridx, gridy, gridz, indexing='ij')

# # Vector of 3D displacements in each grid point
# dx = np.zeros((gridsize,gridsize,gridsize))
# dy = np.zeros((gridsize,gridsize,gridsize))
# dz = np.zeros((gridsize,gridsize,gridsize))
# # dy[1,0,0] = 1.0
# # dy[2,0,0] = -1.0

# # Initialize the 3D Fourier transform with zeros
# Fx = np.fft.fftn(dx)
# Fy = np.fft.fftn(dy)
# Fz = np.fft.fftn(dz)

# # Convert to real form. We are only going to use the sine coefs
# Ax, Bx = complex_to_real_coefficients(Fx)
# Ay, By = complex_to_real_coefficients(Fy)
# Az, Bz = complex_to_real_coefficients(Fz)

Ax = np.zeros((gridsize, gridsize, gridsize))
Ay = np.zeros((gridsize, gridsize, gridsize))
Az = np.zeros((gridsize, gridsize, gridsize))
Bx = np.zeros((gridsize, gridsize, gridsize))
By = np.zeros((gridsize, gridsize, gridsize))
Bz = np.zeros((gridsize, gridsize, gridsize))

# Change a sine coefficient
# Bx[0,3,3] = 10.0
# Bz[1,1,0] = -10.0
# Bz[2,1,1] = 10.0
# Fx = real_to_complex_coefficients(Ax, Bx)
# Fy = real_to_complex_coefficients(Ay, By)
# Fz = real_to_complex_coefficients(Az, Bz)

# Compute the inverse 3D Fourier transform
# dx = np.real(np.fft.ifftn(Fx))
# dy = np.real(np.fft.ifftn(Fy))
# dz = np.real(np.fft.ifftn(Fz))

# Create random shapes
for i in range(gridsize):
    for j in range(gridsize):
        for k in range(gridsize):
            n = ((i+1)*(j+1)*(k+1))
            Bx[i,j,k] = np.random.rand()*10/n
            By[i,j,k] = np.random.rand()*10/n
            Bz[i,j,k] = np.random.rand()*10/n

# Create the frequency grids
nx, ny, nz = gridsize, gridsize, gridsize
kx = np.fft.fftfreq(nx, d=Lx/nx)
ky = np.fft.fftfreq(ny, d=Ly/ny)
kz = np.fft.fftfreq(nz, d=Lz/nz)

for i in range(bone.vertices.shape[0]):
    v = bone.vertices[i]
    dx = interpolate_d(Ax, Bx, v[0], v[1], v[2], kx, ky, kz)
    dy = interpolate_d(Ay, By, v[0], v[1], v[2], kx, ky, kz)
    dz = interpolate_d(Az, Bz, v[0], v[1], v[2], kx, ky, kz)
    bone.vertices[i] += np.array([dx,dy,dz])

bone.export('bone_def.obj')

# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.set_title('Mesh')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# # Scatter plot the points
# ax1.scatter(gridx, gridy, gridz, c='r', s=1)
# ax1.set_box_aspect(bbox.max(axis=0)-bbox.min(axis=0))

# # Add arrows
# # Draw the vectors at each grid point
# # d = np.random.rand(gridsize, gridsize, gridsize, 3)
# for i in range(gridsize):
#     for j in range(gridsize):
#         for k in range(gridsize):
#             # print(gridx[i, j, k], gridy[i, j, k], gridz[i, j, k])
#             ax1.quiver(
#                 gridx[i, j, k], gridy[i, j, k], gridz[i, j, k],  # Starting points
#                 dx[i, j, k], dy[i, j, k], dz[i, j, k],    # Vector components
#                 length=10, normalize=True, color='b'
#             )
# plt.show()
