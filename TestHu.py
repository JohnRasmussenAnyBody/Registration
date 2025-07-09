# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:39:09 2024

@author: jr
"""

import trimesh as tri
import numpy as np
from scipy.spatial.transform import Rotation as R
import meshtools as mt
import matplotlib.pyplot as plt


def apply_random_rotation(mesh):
    """
    Apply a random rotation to the mesh.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.

    Returns:
    trimesh.Trimesh: Rotated mesh.
    """
    # Generate a random rotation using scipy
    random_rotation = R.random().as_matrix()

    # Create a 4x4 homogeneous transformation matrix
    rotation_matrix_4x4 = np.eye(4)  # Start with identity matrix
    rotation_matrix_4x4[:3, :3] = random_rotation  # Replace the top-left 3x3 part with the rotation matrix

    # Apply the transformation to the mesh
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix_4x4)
    
    return rotated_mesh

def apply_random_scaling(mesh):
    """
    Apply a random uniform scaling to the mesh.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.

    Returns:
    trimesh.Trimesh: Scaled mesh.
    """
    # Generate a random scaling factor
    scale_factor = np.random.uniform(0.5, 2.0)  # Random scaling factor between 0.5 and 2.0
    scaling_matrix = np.eye(4) * scale_factor
    scaling_matrix[3, 3] = 1.0  # Keep homogeneous coordinate as 1

    scaled_mesh = mesh.copy()
    scaled_mesh.apply_transform(scaling_matrix)
    return scaled_mesh

def test_hu_moments_invariance(mesh, iterations=5):
    """
    Test the invariance of Hu-like moments under random rotations and scalings.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.
    iterations (int): Number of random transformations to apply.

    Returns:
    list: List of Hu-like moments recorded after each transformation.
    """
    # original_hu_moments = mt.compute_hu_moments_3d(mesh)
    # print(f"Original Hu-like moments: {original_hu_moments}")

    # List to store Hu moments after transformations
    # hu_moments_list = [original_hu_moments]
    hu_moments_list = []

    # Perform a series of random rotations and scalings
    Hs = np.zeros((iterations,6))
    for i in range(iterations):
        # Apply a random rotation
        rotated_mesh = apply_random_rotation(mesh)
        # rotated_mesh = mesh.copy()
        
        # Apply a random scaling
        transformed_mesh = apply_random_scaling(rotated_mesh)
        # transformed_mesh = rotated_mesh.copy()
        # scaling_matrix = np.eye(4) * (i+1)
        # scaling_matrix[3, 3] = 1.0  # Keep homogeneous coordinate as 1
        # transformed_mesh.apply_transform(scaling_matrix)

        # Compute the Hu moments for the transformed mesh
        # Hs[i,0] = mt.nth_moment_about_center_of_mass_normalized(transformed_mesh, 0)
        # Hs[i,1] = mt.nth_moment_about_center_of_mass_normalized(transformed_mesh, 1)
        # Hs[i,2] = mt.nth_moment_about_center_of_mass_normalized(transformed_mesh, 2)
        # Hs[i,3] = mt.nth_moment_about_center_of_mass_normalized(transformed_mesh, 3)
        # Hs[i,4] = mt.nth_moment_about_center_of_mass_normalized(transformed_mesh, 4)
        # Hs[i,5] = mt.nth_moment_about_center_of_mass_normalized(transformed_mesh, 5)
        transformed_hu_moments = mt.compute_hu_moments_3d(transformed_mesh)
        hu_moments_list.append(transformed_hu_moments)

        print(f"Hu-like moments after transformation {i+1}: {transformed_hu_moments}")

    return hu_moments_list, Hs

# Example usage:
# Load a sample mesh (for example, an icosphere)
mesh = tri.load_mesh('m2.obj')

# Test the Hu moments invariance under random transformations
hu_moments_results, Hs = test_hu_moments_invariance(mesh, iterations=15)
hu = np.array(hu_moments_results)

# absc = np.linspace(0,Hs.shape[0]-1,Hs.shape[0])
# for i in range(Hs.shape[1]):
#     plt.plot(absc,Hs[:,i])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt.title('Moment '+str(i))
#     plt.show()
