
"""
Created on Tue Mar 28 14:42:16 2023

@author: shima
"""

import numpy as np
#import os
import trimesh as tri
# import pymeshlab as ml
# import open3d as o3d
from scipy.linalg import polar
from scipy.spatial.transform import Rotation

#os.chdir('/Users/shima/Documents/Guided_Growth_Project/Registration')

def randomcolor():
    color = (np.random.rand(3)*255).astype(int)
    np.append(color,255)  # Use opague color
    return color

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

# Twist a mesh along an axis by angle.
# Concentrate the twist to the middle 30% of the length
def twistmiddle(mesh,axis,theta):
    
    # Robust length-finding method
    vmin = vmax = np.inner(mesh.vertices[0],axis)
    for i,p0 in enumerate(mesh.vertices):    
        pos = np.inner(p0,axis)
        vmin = min(pos, vmin)
        vmax = max(pos, vmax)
    length = vmax-vmin
    x1 = vmin + length*0.3
    x2 = vmax - length*0.3

    # Twist nodes in the mesh
    for i,p0 in enumerate(mesh.vertices):
        pos = np.inner(p0,axis)
        angle = 0.0
        
        if pos >= x1 and pos <= x2:
            angle = theta*(pos-x1)/(x2-x1)
        else:
            if pos > x2:
                angle = theta
                
        T = rotmat(angle, axis)
        mesh.vertices[i] = np.dot(T,p0)       
    return

# Prepare mesh by aligning its principal axes with the coordinate system
def prepare(mesh):
    mesh_g = tri.load_mesh(mesh)

    # Temporarily align the mesh with the global coordinate system
    A = mesh_g.principal_inertia_transform
    mesh_g.apply_transform(A)
    mesh_g.visual.vertex_colors = [  2, 252,  66, 255]
    return (mesh_g)

def decompose_affine1(affine_matrix):
    
    H = affine_matrix[0]
    
    # Translation matrix: H = TL
    T = np.eye(4)
    T[:3,3] = H[:3,3]
    L = H.copy()
    L[:3,3] = 0
    
    # Check
    check = np.allclose(H, T @ L)
    
    # Rotation L = RK, 𝐻 = 𝑇𝑅𝐾
    R, K = polar(L)
    
    # Check that det(R) > 0 and correct if not
    if np.linalg.det(R) < 0:
        R[:3,:3] = -R[:3,:3]
        K[:3,:3] = -K[:3,:3]
    
    r = Rotation.from_matrix(R[:3,:3])
    # euler_angles = r.as_euler('XYZ', degrees=True)
    euler_angles = r.as_rotvec(degrees=True)
    return K, euler_angles

def decompose_affine(affine_matrix):

    # Extract the 3x3 rotation, scaling, and shearing matrix
    rotation_scaling_shearing = affine_matrix[0][:3, :3]
    
    # Scaling vector
    s = rotation_scaling_shearing.sum(axis = 0)
    
    # Rotation matrix
    rotmat = np.array(rotation_scaling_shearing)
    for i in range(3):
        rotmat[:,i] = rotmat[:,i]/s[i]
        
    r = Rotation.from_matrix(rotmat)      
    euler_angles = r.as_euler('XYZ', degrees=True)
    return s, euler_angles
    
    # # Decompose the rotation_scaling_shearing matrix using SVD
    # U, s, V = np.linalg.svd(rotation_scaling_shearing)
    # # Create a diagonal scaling matrix
    # scaling = np.diag(s)
    
    # # Create separate rotation matrices for left and right
    # # singular vectors (which are already orthonormal)
    # rotation_left = U
    # rotation_right = V.T
    
    # # Calculate the shearing matrix as the product of scaling,
    # # left rotation, and right rotation matrices
    # shearing = np.dot(np.dot(rotation_left.T, scaling), rotation_right.T)
   
    # # Convert the rotation matrix to Euler angles
    # r = R.from_matrix(rotation_right)
    # euler_angles = r.as_euler('XYZ', degrees=True)
    
    # Return the scaling, shearing, rotation matrices, and Euler angles
    # return scaling, shearing, rotation_right, euler_angles
    
def findtwist(prename, postname):
    # pre = prepare(prename)
    post = prepare(postname)
    
    pre = tri.load_mesh(prename)
    # post = tri.load_mesh(postname)
    
    pre.export('p1.obj')
    post.export('p2.obj')

    # Register Pre on Post
    A = tri.registration.mesh_other(pre,post,scale= True, icp_first = 100)
    ps = pre.apply_transform(A[0])
    
    ps.visual.vertex_colors = randomcolor()
    ps.export('ps.obj')

    # Find length 
    vertices = ps.vertices
    vmin = np.min(vertices, axis=0)
    vmax = np.max(vertices, axis=0)
    length = vmax[2]-vmin[2]
    
    # Set fractiuon of length to define an end
    frac = 0.25

    # cut ps proximally
    ps_p = tri.intersections.slice_mesh_plane(ps, [0.0, 0.0, 1.0], [0.0, 0.0, vmax[2] - length*frac])

    # cut ps Distally
    ps_d = tri.intersections.slice_mesh_plane(ps, [0.0, 0.0, -1.0], [0.0, 0.0, vmin[2] + length*frac])

    # Register the ends on the post bone
    A_p = tri.registration.icp(ps_p.vertices,post, scale = True, max_iterations=100, threshold=1.0e-8)
    A_d = tri.registration.icp(ps_d.vertices,post, scale = True, max_iterations=100, threshold=1.0e-8)
    
    ps_p.apply_transform(A_p[0])
    ps_p.visual.vertex_colors = randomcolor()
    ps_p.export('ps_p.obj')
    ps_d.apply_transform(A_d[0])
    ps_d.visual.vertex_colors = randomcolor()
    ps_d.export('ps_d.obj')

    # scaling, shearing, rotation_right, euler_p = decompose_affine(A_p)
    # scaling, shearing, rotation_right, euler_d = decompose_affine(A_d)
    
    s_p, euler_p = decompose_affine1(A_p)
    s_d, euler_d = decompose_affine1(A_d)
    
    twistangle = euler_p - euler_d
    return twistangle
    
       
# Validation of the end rotation detection:
# 1. Read in and prepare a bone
# 2. Twist the bone the specified angle
# 3. Register twisted and untwisted ends to each other
# 4. Compute angle and compare with original twist
def validate(fname, angle):
    bone = prepare(fname)
    bone.visual.vertex_colors = randomcolor()
    bone.export('bone.obj')
    
    # Create a copy of the bone and twist it    
    twistbone = bone.copy()
    twistbone.visual.vertex_colors = randomcolor()
    twistmiddle(twistbone,[0,0,1],angle*np.pi/180)
    twistbone.export('twistbone.obj')
    
    detectedangle = findtwist('bone.obj','twistbone.obj')
    print('Detected angle = ',detectedangle,' True angle = ',angle)
    return

if __name__ == "__main__":
    # validate('pre.obj',0)
    
    twistangle = findtwist('STL_Files_Pig/4_right_pre.obj','STL_Files_Pig/4_right_post.obj')
    print(twistangle)

# # load and prepare STL files

# fname = 'STL_Files_Pig/5_Left_Post.stl'
# targname = 'STL_Files_Pig/5_Left_Pre.stl'
# post = prepare(fname)
# post.visual.vertex_colors = [  2, 252,  66, 255]
# pre = prepare(targname)
# pre.visual.vertex_colors = [  2,  81, 252, 255]

# # pre.export('pre.obj')
# # post.export('post.obj')

# #(post+pre).show()

# # Register Pre on Post
# A = tri.registration.mesh_other(pre,post,scale= True, icp_first = 20)
# ps = pre.apply_transform(A[0])
# ps.visual.vertex_colors = [  2,  81, 252, 255]

# ps.export('ps.obj')

# #(ps+ post).show()


# # Slice PS bone using tri.intersections.slice_mesh_plane function 
# vertices = ps.vertices
# vmin = np.min(vertices, axis=0)
# vmax = np.max(vertices, axis=0)
# length = vmax[2]-vmin[2]
# frac = 0.40

# # cut ps proximally
# ps_p = tri.intersections.slice_mesh_plane(ps, [0.0, 0.0, 1.0], [0.0, 0.0, vmax[2] - length*frac])
# ps_p.visual.vertex_colors = [235, 85, 79, 255]
# ps_p.export('ps_p.obj')

# # cut ps Distally
# ps_d = tri.intersections.slice_mesh_plane(ps, [0.0, 0.0, -1.0], [0.0, 0.0, vmin[2] + length*frac])
# ps_d.visual.vertex_colors = [  235,  81, 252, 255]
# ps_d.export('ps_d.obj')

# # Register the ends on the post bone
# A_p = tri.registration.icp(ps_p.vertices,post, scale = True)
# A_d = tri.registration.icp(ps_d.vertices,post, scale = True)

# # Just for visualization, create the registered end meshes and save them
# ps_p_T = ps_p.apply_transform(A_p[0])
# ps_p_T.visual.vertex_colors = [128, 85, 79, 255]
# ps_p_T.export('ps_p_T.obj')
# ps_d_T = ps_d.apply_transform(A_d[0])
# ps_d_T.visual.vertex_colors = [  128,  81, 252, 255]
# ps_d_T.export('ps_d_T.obj')

# # scaling, shearing, rotation_right, euler_p = decompose_affine(A_p)
# # scaling, shearing, rotation_right, euler_d = decompose_affine(A_d)

# s_p, euler_p = decompose_affine1(A_p)
# s_d, euler_d = decompose_affine1(A_d)

# euler = euler_p - euler_d
# print("Rotations of "+fname+" [x, y', z'']: ",euler)

# # show the sliced meshes
# # ps.visual.vertex_colors = [113, 227, 76, 255]

# # (ps+ps_p+ps_d).show()

# # # Slice Post bone using tri.intersections.slice_mesh_plane function 
# # post_p = tri.intersections.slice_mesh_plane(post, plane_normal_pos, 
# #                                             plane_origin= vmax/3,
# #                                             cached_dots = positive_dots_cache )

# # # cut ps proximally

# # post_d = tri.intersections.slice_mesh_plane(post, plane_normal= plane_normal_neg,
# #                                             plane_origin=vmin/3,
# #                                             cached_dots= negative_dots_cache)


# # # show the sliced meshes

# # # post_p.visual.vertex_colors = [143,255,163,255]
# # # post_d.visual.vertex_colors = [87,0,195,255]
# # # post.visual.vertex_colors = [123,111 ,31,255]
# # # (post+post_p+post_d).show()



# # # Register Ps_p on Post_p (proximal part of the bones)

# # A_p = tri.registration.mesh_other(ps_p,post_p, scale = True, icp_first = 40)

# # ps_p_T = ps_p.apply_transform(A_p[0])

# # # show the registered proximal part of the ps and post bones 
# # # ps_p_T.visual.vertex_colors = [239,255,151,255]
# # # (ps_p_T+ post_p).show()


# # # Register Ps_d on Post_d (distall part of the bones)

# # A_d = tri.registration.mesh_other(ps_d,post_d, scale = True, icp_first = 40)

# # ps_d_T = ps_d.apply_transform(A_d[0])

# # # show the registered distal part of the ps and post bones 
# # # ps_d_T.visual.vertex_colors = [143, 255, 163, 255]
# # # (ps_d_T+ post_d).show()

# # #Register ps_p on post_p

# # scaling, shearing, rotation_right, euler_angles = decompose_affine(A_p)
# # scaling, shearing, rotation_right, euler_angles = decompose_affine(A_d)







