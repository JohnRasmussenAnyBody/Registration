# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:38:14 2024

This program  does two things:
    1. Aligns a base bone CoM and principal inertia axes with the global xyz
    2. Rigidly registers the remaining stl files in a directory to the base

@author: jr
"""

# import sys
import os
import numpy as np
import pandas as pd
import trimesh as tri
import meshtools as mt
from BoneEndsClassify import compute_end_params
import matplotlib.pyplot as plt
import pickle


# This function uses previously saved cluster analysis results to determine
# which end of the femur is proximal versus distal and flips the bone if 
# necessary
def pre_align_femur(mesh,direction):
    
    # This aligns the bone long axis along the global z axis,
    # but the proximal/distal orientation will be random
    mt.align_principal_axes(mesh)

    params = compute_end_params(mesh)

    projection_values = params @ direction
    
    # Flip mesh, if it is upside-down
    if projection_values.iloc[0] > projection_values.iloc[1]:
        flip = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]])
        mesh.apply_transform(flip)      
    
    return

# Perform registration of one long bone on another by independent
# registration of the two ends. This brings robustness to the process
# because the ends are easier to distinguish than the entire bone.
# Base bone, Bb, must have been pre-aligned with the z axis and the
# proximal end on the positive axis side.
def rigid_end_registration(Bb,Bt,direction,mirror=False):
        
    # Mirror if necessary
    if mirror:
        print('Mirrorring')
        mirror_x = np.array([
            [-1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ])
        
        # Apply the mirroring transformation to the mesh
        Bt.apply_transform(mirror_x)

    pre_align_femur(Bt,direction)
    
    # Split the two meshes into halves along the long axis,
    # which should be the z axis after the initial alignment
    Bbprox, Bbdist = mt.splitmesh(Bb)
    Btprox, Btdist = mt.splitmesh(Bt)
        
    # Now, register the proximal and distal ends to Bb and decompose the
    # tranformation matrices to rigid and soft parts
    # Aprox = tri.registration.icp(Btprox.vertices, Bbprox)
    Aprox = mt.registration_mesh_other(Btprox, Bbprox, scale=True, icp_first=100)
    AproxRig, AproxSoft = mt.decompose_affine_transformation(Aprox[0])

    Adist = mt.registration_icp(Btdist.vertices, Bbdist, scale=True, initial=Aprox[0])
    # Adist = mt.registration_mesh_other(Btdist, Bbdist, scale=True, icp_first=10, initial=Aprox[0])
    AdistRig, AdistSoft = mt.decompose_affine_transformation(Adist[0])
    
    # Btprox.apply_transform(Aprox[0])
    # Btdist.apply_transform(Adist[0])
    # mt.savemesh(Btprox,'Btprox.obj')
    # mt.savemesh(Btdist,'Btdist.obj')
    # mt.savemesh(Bbprox,'Bbprox.obj')
    # mt.savemesh(Bbdist,'Bbdist.obj')    
    
    # Average the rigid part of the two transformation matrices and move the target
    # bone ends into position
    Apre = mt.interpolate_affine_matrices(AproxRig, AdistRig, 0.5)
    # Btprox.apply_transform(Apre)
    # Btdist.apply_transform(Apre)
    Bt.apply_transform(Apre)
    
    # Now, the two bones should be rigidly aligned, and we can apply
    # the non-rigid (soft) part of the transformation to the ends
    Btprox.apply_transform(Aprox[0])
    Btdist.apply_transform(Adist[0])
    
    AproxInv = mt.inverse_affine_transformation(Aprox[0])
    AdistInv = mt.inverse_affine_transformation(Adist[0])
    diff1 = Aprox[-1]
    diff2 = Adist[-1]
    
    return(AproxInv, AdistInv, diff1, diff2, Bt)

"""
This function scans the folder stl for stl files containing femurs which
have anot already been processed into obj fies in the obj directory.
"""
def filelist(stl,obj):

    stlfiles = set()
    objfiles = set()
    for filename in os.listdir(stl):
        if filename.endswith('.stl'):
            stlfiles.add(filename[:-4])
    for filename in os.listdir(obj):
        if filename.endswith('.obj'):
            objfiles.add(filename[:-4])

    return list(stlfiles-objfiles)

# Check whether the aligned object file of base is not available
# or older than the stl file
def processbase(stl,obj):
    
    # The case where obj does not exist
    if not os.path.exists(obj):
        return True
    
    # The case where the stl is newer than the obj
    stl_mtime = os.path.getmtime(stl)
    obj_mtime = os.path.getmtime(obj)
    if stl_mtime > obj_mtime:
        return True
    
    return False

# Define and align, if necessary, the base bone, Bb
directory = 'C:/Users/jr/Documents/GitHub/Registration/Femurs'
obj = directory+'/obj'
stl = directory+'/stl'
base = 'tlem2'
baseside = 'R'
stl = directory+'/stl'
baseobj = obj+'/'+base+'.obj'

# Read classification vector created by BoneEndsClassify.py
with open('femur_classification_vector.pcl', 'rb') as f:
    direction = pickle.load(f)

if processbase(stl,baseobj):
    Bb = tri.load_mesh(stl)
    pre_align_femur(Bb,direction)
    mt.savemesh(Bb, baseobj)
else:
    Bb = tri.load_mesh(baseobj)

# Read existing list of femurs
femurs = pd.read_excel('Femurs/femurs.xlsx',index_col=0)

stls = filelist(stl,obj)

Bb1, Bb2 = mt.splitmesh(Bb)

for bone in stls:
    print('processing '+bone)
        
    if bone == base:
        continue
    
    assert bone in femurs.index, "Please add this bone to femurs.xlsx"
    
    Bt = tri.load_mesh(directory+'/stl/'+bone+'.stl')
    if not Bt.is_watertight:
        print('Arrrgh, this bone has holes. I hate holes in my bones! Skipping.')
        continue
    
    mirror = False
    if femurs.loc[bone,'Side'] != baseside:
        mirror = True
    
    A1inv, A2inv, diff1, diff2, Btrigid = rigid_end_registration(Bb, Bt, direction, mirror)

    # Assign color and export
    mt.savemesh(Btrigid, obj+'/'+bone+'.obj')
