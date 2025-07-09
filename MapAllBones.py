# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:49:41 2024

This program reads precomputed affine mapping parameters and applies them
to check the algorithm.

@author: jr
"""
import numpy as np
import pandas as pd
import trimesh as tri
import meshtools as mt

directory = 'C:/Users/jr/Documents/GitHub/Registration/Femurs/'
obj = directory+'obj/'
rigs = directory+'rigs/'
maps = directory+'maps/'
base = 'tlem2'
baseside = 'R'

# Load base bone
Bb = tri.load_mesh(obj+base+'.obj')
minz = Bb.bounds[0,2]
maxz = Bb.bounds[1,2]
lz = maxz-minz

# distances to display bones in a grid
dy = lz/4
dz = lz*1.1

bcolor = np.array([179, 192,  43],dtype=int)
tcolor = np.array([165, 132, 122],dtype=int)

# Read existing list of femurs
femurs = pd.read_excel(directory+'femurs.xlsx',index_col=0)

# Determine grid dimensions, ny and nz
n_bone = len(femurs.index)
ratio = 16/9
nz = np.sqrt((n_bone*dy)/(ratio*dz))
nz = max(1,int(nz))
ny = n_bone//nz

gridy = 0
gridz = 0
for n, bone in enumerate(femurs.index):
    
    print(bone)
    Bt = tri.load(obj+bone+'.obj')
    rig = tri.load(rigs+bone+'_rig.obj')
    
    if bone==base:
        continue

    row = femurs.loc[bone]

    # Form proximal matrix
    scaling = np.array([row['ProxSx'],row['ProxSy'],row['ProxSz']])
    rotation_vector = np.array([row['ProxRx'],row['ProxRy'],row['ProxRz']])
    shearing_vector = np.array([row['ProxSH1'],row['ProxSH2'],row['ProxSH3']])
    translation = np.array([row['ProxTx'],row['ProxTy'],row['ProxTz']])
    Aprox = mt.recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)
    
    # Form distal matrix
    scaling = np.array([row['DistSx'],row['DistSy'],row['DistSz']])
    rotation_vector = np.array([row['DistRx'],row['DistRy'],row['DistRz']])
    shearing_vector = np.array([row['DistSH1'],row['DistSH2'],row['DistSH3']])
    translation = np.array([row['DistTx'],row['DistTy'],row['DistTz']])
    Adist = mt.recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)
    
    # Perform an interpolated mapping between the two ends
    vertices = Bb.vertices
    transformed_vertices = np.zeros_like(vertices)

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
    
    # Translate meshes before saving
    tl = [0.0, gridy*dy, gridz*dz]
    Bbtrans.apply_translation(tl)
    Bt.apply_translation(tl)
    rig.apply_translation(tl)
        
    mt.savemesh(Bbtrans, maps+base+'_'+bone+'.obj', color=bcolor)
    mt.savemesh(Bt, maps+bone+'_t.obj', color=tcolor)
    rig.export(maps+bone+'_rig.obj')
    
    # Update grid position
    gridy += 1
    if gridy > ny:
        gridy = 0
        gridz -= 1

    