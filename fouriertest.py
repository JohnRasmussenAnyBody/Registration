# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:16:04 2023

@author: jr
"""

import numpy as np
import trimesh as tri
import plotly.graph_objects as go
import plotly.io as pio

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

def prepare(fammr,fu):
    Mammr = tri.load_mesh(fammr)
    Mammr.visual.vertex_colors = [  2, 252,  66, 255]
    Mu = tri.load_mesh(fu)
    Mu.visual.vertex_colors = [  2,  81, 252, 255]
    
    # (Mu + Mammr).show()

    # Temporarily align the AMMR mesh with the global coordinate system
    A = Mammr.principal_inertia_transform
    Mammr.apply_transform(A)

    
    # Align Mu to Mammr by affine mapping
    A = tri.registration.mesh_other(Mu,Mammr,scale=True)    
    Mr = Mu.apply_transform(A[0])
    Mr.visual.vertex_colors = [  2,  81, 252, 255]
    A = tri.registration.icp(Mr.vertices,Mammr,scale=True)    
    Mr = Mr.apply_transform(A[0])
    (Mr + Mammr).show()

    # Compute the period of the Fourier transformation. Double the
    # bounding box to allow for non-periodic mappings
    T = (Mammr.bounding_box.bounds[1]-Mammr.bounding_box.bounds[0])*2

    return(Mammr, Mr, T)


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

# BNend a mesh around an axis by angle theta per length + offset b.
def bend(mesh,axis1,axis2,theta,b):
    for i,p0 in enumerate(mesh.vertices):
        angle = theta * np.inner(p0,axis2) + b
        T = rotmat(angle, axis1)
        mesh.vertices[i] = np.dot(T,p0)
    return

Mammr, Mr, T = prepare('SourceFemur.stl','TargetFemur.stl')

twist(Mammr,[0,0,1],0*(np.pi)/T[2],0)
# bend(Mammr,[1,0,0],[0,0,1],(np.pi/2)/T[2],0)

Mammr.show()


# T = [2.0, 2.0]
# # a = np.zeros((1,2,2))
# # b = np.zeros((1,2,2))
# a = (np.random.rand(1,2,2)-0.5)*0.1
# b = (np.random.rand(1,2,3)-0.5)*0.1
# # a[0,0,1] = 0.0
# # a[0,0,1] = 0.0

# n = 100
# x = np.linspace(0, T[0]/2, n)
# y = np.linspace(0, T[1]/2, n)

# z = np.zeros((n,n))

# for i in range(n):
#     for j in range(n):
#         z[i,j] = fourier(a,b,T,np.array([x[i],y[j]]))

# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# # fig.update_layout(title='Fourier', autosize=True,
# #                   width=1000, height=500,
# #                   margin=dict(l=65, r=50, b=65, t=90))
# fig.update_layout(title='Fourier', autosize=True,
#                   width=1000, height=500)
# pio.renderers.default='browser'
# fig.show()
