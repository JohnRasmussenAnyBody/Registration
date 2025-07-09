# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:06:44 2024

@author: jr
"""

import numpy as np
import pandas as pd
import trimesh as tri
import meshtools as mt

femurs = pd.read_excel('Femurs/femurs.xlsx',index_col=0)

Bb = tri.load('Femurs/tlem2.obj')
Bbprox, Bbdist = mt.splitmesh(Bb, 0.33, True)
Bt = tri.load('Femurs/01L_Femur.obj')

row = femurs.loc['01L_Femur']

params = row.iloc[5:17]
scaling = np.array(params[0:3])
rotation_vector = np.array(params[3:6])
shearing_vector = np.array(params[6:9])
translation = np.array(params[9:12])
Aprox = mt.recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)

# Form distal matrix
params = row.iloc[17:29]
scaling = np.array(params[0:3])
rotation_vector = np.array(params[3:6])
shearing_vector = np.array(params[6:9])
translation = np.array(params[9:12])
Adist = mt.recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)

# for i in range(11):
#     B = Bb.copy()
#     A = mt.interpolate_affine_matrices(Adist, Aprox, i/10.0)
#     B.apply_transform(A)
#     mt.savemesh(B,'Bb'+str(i)+'.obj')

vertices = Bb.vertices
transformed_vertices = np.zeros_like(vertices)
minz = Bb.bounds[0,2]
maxz = Bb.bounds[1,2]
lz = maxz-minz

# Loop over each vertex
for i, vertex in enumerate(vertices):
    # Convert the vertex to homogeneous coordinates
    vertex_homogeneous = np.append(vertex, 1)
    alpha = mt.heaviside((vertex[2]-minz)/lz)
    A = mt.interpolate_affine_matrices(Adist, Aprox, alpha)
    
    # Transform the vertex using the modified matrix
    transformed_vertex_homogeneous = A @ vertex_homogeneous
    
    # Convert back to 3D coordinates
    transformed_vertices[i] = transformed_vertex_homogeneous[:3]

# Create a new mesh with the transformed vertices and the same faces
Bbtrans = tri.Trimesh(vertices=transformed_vertices, faces=Bb.faces)
mt.savemesh(Bbtrans, 'Bbmapped.obj')
