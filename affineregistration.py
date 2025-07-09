# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:38:14 2024

Implementation of an affine registration function, i.e. fully fledged
with the rigid body operations of translation and rotation,
and the deformations non-uniform scaling and skewing.

@author: jr
"""

# import sys
import numpy as np
import trimesh as tri
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import meshtools as mt


# Main
Bb = tri.load_mesh('Femurs/Bbprox.obj')
# Bt = tri.load_mesh('Femurs/Btprox.obj')

# vertices = tri.registration.nricp_amberg(Bb, Bt)
# print('Amberg done')

# faces = Bb.faces

# # Ensure that the res array has the correct shape (it should have the same number of vertices as Bn)
# # Bn.vertices.shape[0] gives the number of vertices in Bn
# assert vertices.shape[0] == Bb.vertices.shape[0], "The number of vertices in res must match the original mesh"

# # Create a new target mesh with the same vertex structure as Bb
# Bbt = tri.Trimesh(vertices=vertices, faces=faces)

# diff = squared_sum_of_distances(Bb, Bbt)
# print('Difference = ',diff)

Bbt = tri.load_mesh('Femurs/Bbt.obj')

# Example usage with Bb and Bbt as your meshes
result = mt.optimize_affine_transformation(Bb, Bbt)

# Extract the optimized parameters
optimized_params = result.x
scaling_optimized = optimized_params[0:3]
rotation_vector_optimized = optimized_params[3:6]
shearing_vector_optimized = optimized_params[6:9]
translation_optimized = optimized_params[9:12]

A = mt.recompose_affine_matrix(scaling_optimized, rotation_vector_optimized, shearing_vector_optimized, translation_optimized)

print("Optimized Scaling:", scaling_optimized)
print("Optimized Rotation Vector:", rotation_vector_optimized)
print("Optimized Shearing Vector:", shearing_vector_optimized)
print("Optimized Translation:", translation_optimized)

Bb.apply_transform(A)
Bb.visual.vertex_colors = mt.randomcolor()
Bb.export('Femurs/Bbmapped.obj')
