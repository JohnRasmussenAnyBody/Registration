# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:38:14 2024

This program maps a base bone to a collection of other bones and stores the
mapping parameters in a table. The bones must have been pre-registered before
this process.

@author: jr
"""

# import sys
import os
import numpy as np
import pandas as pd
import trimesh as tri
from scipy.spatial.transform import Rotation
import meshtools as mt

# Perform registration of one long bone on another by independent
# registration of the two ends. This will allow a subsequent interpolated
# mapping between the bones that allows for bends and twists.
# The base mesh must be aligned with the z axis with the proximal
# end at the top.
def end_registration(Bb,Bt):
        
    # Split the two meshes into halves along the long axis,
    # which should be the z axis after the initial alignment
    Bbprox, Bbdist = mt.splitmesh(Bb, 0.50, True)
    Btprox, Btdist = mt.splitmesh(Bt, 0.50, True)
        
    # Now, register the proximal end to Bt on a node-by-node
    # basis to create a target with the same node structure as the base
    transprox = mt.translation_matrix(Btprox.center_mass-Bbprox.center_mass)
    Bbprox.apply_transform(transprox)
    A = tri.registration.icp(Bbprox.vertices, Btprox, scale=True)
    Bbprox.apply_transform(A[0])
    vertices = tri.registration.nricp_amberg(Bbprox, Btprox)
    faces = Bbprox.faces
    print('Amberg proximal done')
    
    assert vertices.shape[0] == Bbprox.vertices.shape[0], "The number of vertices in res must match the original mesh"
    
    # Create a new target mesh with the same vertex structure as Bb
    Bbt = tri.Trimesh(vertices=vertices, faces=faces, process=False)
    result = mt.optimize_affine_transformation(Bbprox, Bbt)
    Aopt = mt.recompose_affine_matrix(result.x[0:3], result.x[3:6], result.x[6:9], result.x[9:12])
    ProxResults = Aopt @ A[0] @ transprox # The initial movement from icp must be included

    # Same for the distal end
    transdist = mt.translation_matrix(Btdist.center_mass-Bbdist.center_mass)
    Bbdist.apply_transform(transdist)
    A = tri.registration.icp(Bbdist.vertices, Btdist, scale=True)
    Bbdist.apply_transform(A[0])
    vertices = tri.registration.nricp_amberg(Bbdist, Btdist)
    faces = Bbdist.faces
    print('Amberg distal done')
    
    assert vertices.shape[0] == Bbdist.vertices.shape[0], "The number of vertices in res must match the original mesh"
    
    # Create a new target mesh with the same vertex structure as Bb
    Bbt = tri.Trimesh(vertices=vertices, faces=faces, process=False)
    result = mt.optimize_affine_transformation(Bbdist, Bbt)
    Aopt = mt.recompose_affine_matrix(result.x[0:3], result.x[3:6], result.x[6:9], result.x[9:12])
    DistResults = Aopt @ A[0] @ transdist
    
    return(ProxResults, DistResults)

# Inventory
directory = 'C:/Users/jr/Documents/GitHub/Registration/Femurs'
obj = directory+'/obj'
base = 'tlem2'
baseside = 'R'

# Load base bone
Bb = tri.load_mesh(obj+'/'+base+'.obj')
        
# Path to reference Excel file
excel_path = os.path.join(directory, 'femurs.xlsx')
excel_mtime = os.path.getmtime(excel_path)  # Modification time of femurs.xlsx

# Inventory of .obj files that are newer than femurs.xlsx
objfiles = set()
for filename in os.listdir(obj):
    if filename.endswith('.obj'):
        file_path = os.path.join(obj, filename)
        if os.path.getmtime(file_path) > excel_mtime:
            objfiles.add(filename[:-4])  # strip '.obj' extension
                
objfiles = objfiles - set(base)
        
# Read existing list of femurs
femurs = pd.read_excel('Femurs/femurs.xlsx',index_col=0)

for bone in objfiles:
    print('processing '+bone)
    
    Bt = tri.load_mesh(obj+'/'+bone+'.obj')
        
    A1, A2 = end_registration(Bb, Bt)
    scale1, rot1, shear1, trans1 = mt.decompose_affine_matrix(A1)
    scale2, rot2, shear2, trans2 = mt.decompose_affine_matrix(A2)
    rot1, rot2 = mt.coordinate_rotation_vectors(rot1, rot2)
    res1 = np.concatenate([scale1, rot1, shear1, trans1])
    res2 = np.concatenate([scale2, rot2, shear2, trans2])
    
    femurs.loc[bone,'ProxSx'] = scale1[0]
    femurs.loc[bone,'ProxSy'] = scale1[1]
    femurs.loc[bone,'ProxSz'] = scale1[2]

    femurs.loc[bone,'ProxRx'] = rot1[0]
    femurs.loc[bone,'ProxRy'] = rot1[1]
    femurs.loc[bone,'ProxRz'] = rot1[2]

    femurs.loc[bone,'ProxSH1'] = shear1[0]
    femurs.loc[bone,'ProxSH2'] = shear1[1]
    femurs.loc[bone,'ProxSH3'] = shear1[2]

    femurs.loc[bone,'ProxTx'] = trans1[0]
    femurs.loc[bone,'ProxTy'] = trans1[1]
    femurs.loc[bone,'ProxTz'] = trans1[2]

    femurs.loc[bone,'DistSx'] = scale2[0]
    femurs.loc[bone,'DistSy'] = scale2[1]
    femurs.loc[bone,'DistSz'] = scale2[2]

    femurs.loc[bone,'DistRx'] = rot2[0]
    femurs.loc[bone,'DistRy'] = rot2[1]
    femurs.loc[bone,'DistRz'] = rot2[2]

    femurs.loc[bone,'DistSH1'] = shear2[0]
    femurs.loc[bone,'DistSH2'] = shear2[1]
    femurs.loc[bone,'DistSH3'] = shear2[2]

    femurs.loc[bone,'DistTx'] = trans2[0]
    femurs.loc[bone,'DistTy'] = trans2[1]
    femurs.loc[bone,'DistTz'] = trans2[2]

    # Mapped properties, hip joint center, knee joint axis end, antetorsion angle
    hjcbase = np.array([femurs.loc[base,'HJCx'],femurs.loc[base,'HJCy'],femurs.loc[base,'HJCz']])
    hjc = mt.point_affine_transformation(A1, hjcbase)
    femurs.loc[bone,'HJCx'] = hjc[0]
    femurs.loc[bone,'HJCy'] = hjc[1]
    femurs.loc[bone,'HJCz'] = hjc[2]

    kjcmbase = np.array([femurs.loc[base,'KJCmx'],femurs.loc[base,'KJCmy'],femurs.loc[base,'KJCmz']])
    kjcm = mt.point_affine_transformation(A2, kjcmbase)
    femurs.loc[bone,'KJCmx'] = kjcm[0]
    femurs.loc[bone,'KJCmy'] = kjcm[1]
    femurs.loc[bone,'KJCmz'] = kjcm[2]

    kjclbase = np.array([femurs.loc[base,'KJClx'],femurs.loc[base,'KJCly'],femurs.loc[base,'KJClz']])
    kjcl = mt.point_affine_transformation(A2, kjclbase)
    femurs.loc[bone,'KJClx'] = kjcl[0]
    femurs.loc[bone,'KJCly'] = kjcl[1]
    femurs.loc[bone,'KJClz'] = kjcl[2]
    
    kjc = (kjcm+kjcl)/2

    # Map neck point. We use the proximal mapping for this
    ncbase = np.array([femurs.loc[base,'NCx'],femurs.loc[base,'NCy'],femurs.loc[base,'NCz']])
    nc = mt.point_affine_transformation(A1, ncbase)
    femurs.loc[bone,'NCx'] = nc[0]
    femurs.loc[bone,'NCy'] = nc[1]
    femurs.loc[bone,'NCz'] = nc[2]

    # Compute angles
    ushaft = (nc-kjc)/np.linalg.norm(nc-kjc)
    uneck = (nc-hjc)/np.linalg.norm(nc-hjc)
    ufunc = (kjc-hjc)/np.linalg.norm(kjc-hjc)
    uknee = (kjcl-kjcm)/np.linalg.norm(kjcl-kjcm)
    
    rneck, rmsd = Rotation.align_vectors([ushaft], [uneck])
    aneck = rneck.magnitude()
    rfm, rmsd = Rotation.align_vectors([ufunc], [uknee])
    afm = rfm.magnitude()
    
    # Neck angle
    femurs.loc[bone,'NA'] = aneck*180/np.pi
    
    # Valgus (for femur isolated)
    femurs.loc[bone,'Valgus'] = afm*180/np.pi - 90.0
    
    # Functional length
    femurs.loc[bone,'FL'] = np.linalg.norm(hjc-kjc)
    
    # Knee epicondyle width
    femurs.loc[bone,'KEW'] = np.linalg.norm(kjcm-kjcl)

    # Antetorsion angle is the difference between z rotations
    femurs.loc[bone,'Antetor'] = (rot2[2]-rot1[2])*180/np.pi
    
    # Knee joint radius. The axis is rougly aligned with the y axis, so we scale with the mean of x and z scaling
    femurs.loc[bone,'KJr'] = femurs.loc[base,'KJr']*(scale2[0]+scale2[2])/2
    
    # Hip joint radius
    femurs.loc[bone,'HJr'] = femurs.loc[base,'HJr']*scale1.mean()
    
    # Create a rig
    funcaxis = mt.create_cyl(kjc,hjc,extend=1.2)
    kneeaxis = mt.create_cyl(kjcm,kjcl,extend=1.2)
    shaftaxis = mt.create_cyl(kjc,nc,extend=1.2)
    neckaxis = mt.create_cyl(hjc,nc,extend=2.4)
    rig = funcaxis+kneeaxis+shaftaxis+neckaxis
    rig.export(directory+'/rigs/'+bone+'_rig.obj')    
    
femurs.to_excel('Femurs/femurs.xlsx')
