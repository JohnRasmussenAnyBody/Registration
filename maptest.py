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

    # Use cluster analysis to determine the need for a flip
    params = compute_end_params(mesh)
    params.drop(columns=['E0'], errors='ignore', inplace=True) #E0 is confounding   

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

def align_bt_to_bb_by_inertia_and_com(Bb, Bt):
    """
    Return an affine transform that moves Bt to align its center of mass and 
    principal inertia axes with those of Bb, without altering Bb.
    
    Parameters:
    - Bb (trimesh.Trimesh): reference mesh
    - Bt (trimesh.Trimesh): target mesh to be aligned
    
    Returns:
    - affine_transform (np.ndarray): 4x4 matrix transforming Bt to Bb's frame
    """

    assert Bb.is_watertight and Bt.is_watertight, "Both meshes must be watertight."

    # Step 1: Compute center of mass
    com_Bb = Bb.center_mass
    com_Bt = Bt.center_mass

    # Step 2: Get principal axes (rotation matrices from principal inertia transform)
    R_Bb = Bb.principal_inertia_transform[:3, :3]
    R_Bt = Bt.principal_inertia_transform[:3, :3]

    # Ensure proper orientation (orthonormal, right-handed)
    if np.linalg.det(R_Bb) < 0:
        R_Bb[:, -1] *= -1
    if np.linalg.det(R_Bt) < 0:
        R_Bt[:, -1] *= -1

    # Step 3: Rotation to align Bt’s principal axes to those of Bb
    R_align = R_Bb @ R_Bt.T

    # Step 4: Translation to align Bt’s COM with Bb’s COM
    t_align = com_Bb - R_align @ com_Bt

    # Step 5: Construct the full affine transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_align
    T[:3, 3] = t_align

    return T

"""
This is just a helper function for align_bt_to_bb_best_flip_by_volume.
It condenses all the steps needed to tranfer and align Bt to Bb into
a single transform, T_final. The steps are:
1. Translates Bt to origin (by its COM),
2. Rotates Bt into its principal frame,
3. Rotates into Bb’s frame (optionally with axis flips),
4. Translates to Bb’s COM.

Parameters:
- Bb: reference mesh (trimesh.Trimesh)
- Bt: target mesh to align (trimesh.Trimesh)
- flip: optional flip matrix (3x3) with det=+1

Returns:
- 4x4 numpy affine transformation matrix
"""
def condensed_alignment_transform(Bb, Bt, flip=np.diag([1, 1, 1])):

    assert Bb.is_watertight and Bt.is_watertight

    com_Bb = Bb.center_mass
    com_Bt = Bt.center_mass

    R_Bb = Bb.principal_inertia_transform[:3, :3]
    R_Bt = Bt.principal_inertia_transform[:3, :3]

    # Fix handedness
    if np.linalg.det(R_Bb) < 0:
        R_Bb[:, -1] *= -1
    if np.linalg.det(R_Bt) < 0:
        R_Bt[:, -1] *= -1

    # Apply optional flip to Bt’s frame
    R_Bt_flipped = R_Bt @ flip

    # Compose the full transformation matrix: T = T4 @ T3 @ T2 @ T1
    T = np.eye(4)

    # Step 1: Translate Bt to origin
    T1 = np.eye(4)
    T1[:3, 3] = -com_Bt

    # Step 2: Rotate to Bt inertia frame
    T2 = np.eye(4)
    T2[:3, :3] = R_Bt_flipped

    # Step 3: Rotate to Bb frame
    T3 = np.eye(4)
    T3[:3, :3] = R_Bb.T

    # Step 4: Translate to Bb's COM
    T4 = np.eye(4)
    T4[:3, 3] = com_Bb

    # Final transform: T = T4 @ T3 @ T2 @ T1
    T = T4 @ T3 @ T2 @ T1

    return T

"""
Try all valid right-handed flip alignments of Bt to Bb.
Return the one minimizing volume of symmetric difference.

Parameters:
- Bb: base mesh (trimesh)
- Bt: target mesh to align (trimesh)

Returns:
- best_T: best 4x4 transformation matrix
"""
def align_bt_to_bb_best_flip_by_volume(Bb, Bt):

    assert Bb.is_watertight and Bt.is_watertight, "Meshes must be watertight."

    R_Bb = Bb.principal_inertia_transform[:3, :3]
    R_Bt = Bt.principal_inertia_transform[:3, :3]

    if np.linalg.det(R_Bb) < 0:
        R_Bb[:, -1] *= -1
    if np.linalg.det(R_Bt) < 0:
        R_Bt[:, -1] *= -1

    flip_options = [
        np.diag([+1, +1, +1]),
        np.diag([+1, -1, -1]),
        np.diag([-1, +1, -1]),
        np.diag([-1, -1, +1])
    ]

    best_T = None
    min_vol_error = np.inf
    
    # Temporary
    # Bt_test = Bt.copy()
    # T = np.eye(4)
    # T[:3, 3] = -com_Bt
    # Bt_test.apply_transform(T)
    # T = np.eye(4)
    # T[:3, :3] = R_Bt
    # Bt_test.apply_transform(T)
    
    # T = np.eye(4)
    # T[:3, :3] = R_Bb.T
    # Bt_test.apply_transform(T)

    # T = np.eye(4)
    # T[:3, 3] = -com_Bb
    # Bt_test.apply_transform(T)
    # mt.savemesh(Bt_test,'Bt_test1.obj')
    
    for flip in flip_options:
        T = condensed_alignment_transform(Bb, Bt, flip=flip)
    
        Bt_test = Bt.copy()
        Bt_test.apply_transform(T)
        mt.savemesh(Bt_test,'Bt_test1.obj')
    
        vol_error = mt.volume_difference(Bb, Bt_test)
        print(f"Flip {flip.diagonal()} -> Volume difference: {vol_error:.3f}")
    
        if vol_error < min_vol_error:
            min_vol_error = vol_error
            best_T = T

    return best_T

def rigid_end_registration2(Bb,Bt,direction,mirror=False):
        
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
    Btprox, Btdist = mt.splitmesh(Bt, alpha=0.33)
    Bbprox, Bbdist = mt.splitmesh(Bb, alpha=0.33)
    
    # Now, register the proximal and distal ends to Bb and decompose the
    # tranformation matrices to rigid and soft parts
    # Aprox = tri.registration.icp(Btprox.vertices, Bbprox)
    AproxRig = align_bt_to_bb_best_flip_by_volume(Bbprox, Btprox)
    AdistRig = align_bt_to_bb_best_flip_by_volume(Bbdist, Btdist)
    
    # Btprox.apply_transform(AproxRig)
    # Btdist.apply_transform(AdistRig)
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
    
    return(Bt)

# Alternative version using affine optimization and difference volume optimization
def rigid_end_registration3(Bb,Bt,direction,mirror=False):
        
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
    Bbprox, Bbdist = mt.splitmesh(Bb, alpha=0.33)
    Btprox, Btdist = mt.splitmesh(Bt, alpha=0.33)
        
    # Now, register the proximal and distal ends to Bb and decompose the
    # tranformation matrices to rigid and soft parts
    result = mt.optimize_affine_transformation2(Btprox, Bbprox)
    Aprox = mt.recompose_affine_matrix(result.x[0:3], result.x[3:6], result.x[6:9], result.x[9:12])
    AproxRig, AproxSoft = mt.decompose_affine_transformation(Aprox)

    result = mt.optimize_affine_transformation2(Btdist, Bbdist)
    Adist = mt.recompose_affine_matrix(result.x[0:3], result.x[3:6], result.x[6:9], result.x[9:12])
    AdistRig, AdistSoft = mt.decompose_affine_transformation(Adist)
    
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



# Main

Bb = tri.load_mesh('base.obj')
Bt = tri.load_mesh('target.obj')

flip = np.diag([-1, -1, +1])

params = np.zeros(12)
params[0:3] = [1,1,1] #scaling_factors
params[3:6] = [0, 0, 0] #rotation_vector
params[6:9] = [0,0,0]
params[9:12] = [0,0,0]

momentorder = 3
axis = 0
Btmoments = mt.raw_moments_about_origin(Bt,momentorder,Bt.center_mass)

num = 50
table = np.zeros((num,momentorder))
angles = np.linspace(0, 2*np.pi, num=num)
for i,angle in enumerate(angles):
    params[3+axis] = angle
    scaling = params[0:3]
    rotation_vector = params[3:6]
    shearing_vector = params[6:9]
    translation = params[9:12]
#    Bb_trans = mt.apply_affine_transformation(Bb, scaling, rotation_vector, shearing_vector, translation)
    objf = mt.objective_function3(params, Bb, Bt, momentorder, Btmoments)
    table[i,0] = objf

#for i in range(momentorder):
plt.plot(table[:,0])
plt.grid()
plt.title(f'Moment order {momentorder}, axis {axis}')
plt.show()
    
# objf = mt.objective_function3(params, Bb, Bt, momentorder, Btmoments)
# print(objf)

