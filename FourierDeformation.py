# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:40:57 2023

@author: jr
"""

import sys
import os
import numpy as np
import trimesh as tri
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def decompose_affine(affine_matrix):
    # Extract the 3x3 rotation, scaling, and shearing matrix
    rotation_scaling_shearing = affine_matrix[:3, :3]
    # Decompose the rotation_scaling_shearing matrix using SVD
    U, s, V = np.linalg.svd(rotation_scaling_shearing)
    # Create a diagonal scaling matrix
    scaling = np.diag(s)
    # Create separate rotation matrices for left and right
    # singular vectors (which are already orthonormal)
    rotation_left = U
    rotation_right = V.T
    # Calculate the shearing matrix as the product of scaling,
    # left rotation, and right rotation matrices
    shearing = np.dot(np.dot(rotation_left.T, scaling), rotation_right.T)
    # Convert the rotation matrix to Euler angles
    r = R.from_matrix(rotation_right)
    euler_angles = r.as_euler('zyx', degrees=True)
    # Return the scaling, shearing, rotation matrices, and Euler angles
    return scaling, shearing, rotation_right, euler_angles

def prepare(fammr,fu):
    Mammr = tri.load_mesh(fammr)

    # Temporarily align the AMMR mesh with the global coordinate system
    A = Mammr.principal_inertia_transform
    Mammr.apply_transform(A)
    Mammr.visual.vertex_colors = [  2, 252,  66, 255]
        
    Mu = tri.load_mesh(fu)
    
    # Align Mu to Mammr rigidly
    A = tri.registration.mesh_other(Mu,Mammr,scale=False)
    Mr = Mu.apply_transform(A[0])
    Mr.visual.vertex_colors = [  2,  81, 252, 255]

    # Compute the period of the Fourier transformation. Double the
    # bounding box to allow for non-periodic mappings
    T = (Mammr.bounding_box.bounds[1]-Mammr.bounding_box.bounds[0])*2

    return(Mammr, Mr, T)

def fourier(a,b,T,x):
    d = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        prod = 1.0
        for k in range(a.shape[2]):
            fou = 0.0
            for j in range(a.shape[1]):
                omega = 2*np.pi/T[j]
                fou = a[i,j,k]*np.cos((k+1)*omega*x[j]) + b[i,j,k]*np.sin((k+1)*omega*x[j])
                prod *= fou
        d[i] += prod
    return d

def fouriermapmesh(mesh,a,b,T):
    for i,p0 in enumerate(mesh.vertices):
        mesh.vertices[i] += fourier(a,b,T,p0)
    return

# Calculate rotation matrix of a given angle about an axis given as a vector
def rotmat(angle, axis):
    u = np.array(axis) / np.linalg.norm(axis)
    u_x, u_y, u_z = u[0], u[1], u[2]
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    T = np.array([[cos_theta + u_x**2 * (1-cos_theta), u_x*u_y*(1-cos_theta) - u_z*sin_theta, u_x*u_z*(1-cos_theta) + u_y*sin_theta],
                  [u_y*u_x*(1-cos_theta) + u_z*sin_theta, cos_theta + u_y**2 * (1-cos_theta), u_y*u_z*(1-cos_theta) - u_x*sin_theta],
                  [u_z*u_x*(1-cos_theta) - u_y*sin_theta, u_z*u_y*(1-cos_theta) + u_x*sin_theta, cos_theta + u_z**2 * (1-cos_theta)]])
    return T

# Twist a mesh along an axis by angle theta per length + offset b.
def twist(mesh,axis,theta,b):
    for i,p0 in enumerate(mesh.vertices):
        angle = theta * np.inner(p0,axis) + b
        T = rotmat(angle, axis)
        mesh.vertices[i] = np.dot(T,p0)
    return

# Twist a mesh along an axis by angle theta per length + offset b.
def twistmove(mesh,axis,theta,b,dx,dy,dz):
    for i,p0 in enumerate(mesh.vertices):
        angle = theta * np.inner(p0,axis) + b
        T = rotmat(angle, axis)
        mesh.vertices[i] = np.dot(T,p0)+[dx,dy,dz]
    return


def makeab(dv):
    
    # Check dimensions of design variables
    length = dv.shape[0]
    
    if length < 18:
        print('Error: A minimum of 18 design variables required')
        return(0)
        
    if (length % 2) != 0:
        print('Error: length of design variable array must be even')
        return(0)
    
    length = int(length/2)
    
    if (length % 9) != 0:
        print('Error: length of design variable array must be divisible by 9')
        return(0)
    
    nf = int(length/9)
    a = np.zeros((3,3,nf))
    b = np.zeros((3,3,nf))
    
    count = 0
    for i in range(3):
        for j in range(3):
            for k in range(nf):
                a[i,j,k] = dv[count]
                b[i,j,k] = dv[length+count]
                count += 1    

    return a,b,nf

# Compute the relative difference between two meshes, M1 and M2 
def meshmetric(M1,M2):
    diff = [(M1.volume-M2.volume)/M2.volume,
            (M1.area-M2.area)/M2.area,
            (M1.center_mass[0]-M2.center_mass[0])/M2.center_mass[0],
            (M1.center_mass[1]-M2.center_mass[1])/M2.center_mass[1],
            (M1.center_mass[2]-M2.center_mass[2])/M2.center_mass[2],
            (M1.mass_properties['inertia'][0,0]-M2.mass_properties['inertia'][0,0])/M2.mass_properties['inertia'][0,0],
            (M1.mass_properties['inertia'][1,1]-M2.mass_properties['inertia'][1,1])/M2.mass_properties['inertia'][1,1],
            (M1.mass_properties['inertia'][2,2]-M2.mass_properties['inertia'][2,2])/M2.mass_properties['inertia'][2,2], 
            (M1.mass_properties['inertia'][0,1]-M2.mass_properties['inertia'][0,1])/M2.mass_properties['inertia'][0,1],
            (M1.mass_properties['inertia'][0,2]-M2.mass_properties['inertia'][0,2])/M2.mass_properties['inertia'][0,2],
            (M1.mass_properties['inertia'][1,2]-M2.mass_properties['inertia'][1,2])/M2.mass_properties['inertia'][1,2],
            ]
    res = np.linalg.norm(diff)
    return res



# Objective function
def objective(dv,Mammr,Mr,T,method):

    # Make a copy to avoid deformations accumulating    
    Mtmp = Mammr.copy(True)
    
    if method == 'twist':
        twist(Mtmp,[0,0,1],dv[0],dv[1])
    if method == 'twistmove':
        twistmove(Mtmp,[0,0,1],dv[0],dv[1],dv[2],dv[3],dv[4])
        
    if method=='fourier':
        a, b, nf = makeab(dv)
        fouriermapmesh(Mtmp,a,b,T)
        
    res = meshmetric(Mtmp,Mr)
    print(res)
    return res

mammrfname = 'STL_Files_Pig/5_Left_Pre.stl'
mrfname = 'STL_Files_Pig/5_Left_Post.stl'
Mammr, Mr, T = prepare(mammrfname,mrfname)
# Mammr, Mr, T = prepare('STL_Files_Pig/5_Left_Post.stl','STL_Files_Pig/5_Right_Post.stl')
# Mammr, Mr, T = prepare('PelvisMale.stl','PelvisFemale.stl')
# Mammr, Mr, T = prepare('SourceFemur.stl','TargetFemur.stl')
Mr.export(mrfname[:-3]+'obj')

# Pretransform Mammr by an affine mapping to Mr
A = tri.registration.mesh_other(Mammr,Mr,scale=True)
Mammr = Mammr.apply_transform(A[0])
Mammr.export(mammrfname[:-3]+'obj')

# method = 'twist'
# method = 'twistmove'
# method = 'fourier'
method = ''

if method=='twist':
    dv = np.zeros(2)
    res = objective(dv,Mammr,Mr,T,method)
    res = minimize(objective,dv,args=(Mammr,Mr,T,method),method='SLSQP', options={'eps': 1.0E-04, 'maxiter':100, 'disp':True, 'ftol':0.0001} )
    twist(Mammr,[0,0,1],res.x[0],res.x[1])

if method=='twistmove':
    dv = np.zeros(5)
    res = objective(dv,Mammr,Mr,T,method)
    res = minimize(objective,dv,args=(Mammr,Mr,T,method),method='SLSQP', options={'eps': 1.0E-04, 'maxiter':100, 'disp':True, 'ftol':0.0001} )
    twistmove(Mammr,[0,0,1],res.x[0],res.x[1],res.x[2],res.x[3],res.x[4])

if method=='fourier':
    dv = np.zeros(18)
    res = objective(dv,Mammr,Mr,T,method)
    for i in range(dv.shape[0]):
        dv[i] = (i+1)*0.01
    res = minimize(objective,dv,args=(Mammr,Mr,T,method),method='SLSQP', options={'eps': 1.0E-04, 'maxiter':100, 'disp':True, 'ftol':0.0001} )
    a, b, nf = makeab(res.x)
    fouriermapmesh(Mammr,a,b,T)
    
# (Mr+Mammr).show()

# Save everything
Mammr.visual.vertex_colors = [  200, 66,  48, 255]
Mammr.export(mammrfname[:-3]+'morph.obj')
original_stdout = sys.stdout
# with open('res.txt','wt') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(res)
#     sys.stdout = original_stdout
# radii = np.linalg.norm(mesh.vertices - mesh.center_mass, axis=1)

# # Assign colors to vertices based on curvature
# mesh.visual.vertex_colors = tri.visual.interpolate(radii, color_map='viridis')
# mesh.show()

