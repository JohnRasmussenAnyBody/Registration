# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:18:41 2023

@author: jr
"""

import numpy as np
import trimesh as tri
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fname = 'STL_Files_Pig/5_Left_Post.stl'
targname = 'STL_Files_Pig/5_Left_Pre.stl'

mesh = tri.load_mesh(fname)
mesh.visual.vertex_colors = [  2, 252,  66, 255]
# targ = tri.load_mesh(targname)
# targ.visual.vertex_colors = [  2,  81, 252, 255]

# A = tri.registration.mesh_other(mesh,targ,scale=False, icp_first=20)
# mesh = mesh.apply_transform(A[0])

# (mesh+targ).show()

vertices = mesh.vertices
vmin = np.min(vertices, axis=0)
vmax = np.max(vertices, axis=0)
d = vmax - vmin

origin = vmin
vec = (0.0, 0.0, 1.0)
heights = np.linspace(0, d[2], 170)

lines, to_3d, faces = tri.intersections.mesh_multiplane(mesh, origin, vec, heights)

random_rgb = np.random.rand(len(lines), 3)
for i,loop in enumerate(lines):
    color = mcolors.rgb2hex(random_rgb[i])
    fig = plt.figure(figsize=(6, 6))
    for i in range(loop.shape[0]):
        line = loop[i]
        plt.plot(line[:,0],line[:,1],color=color)
    plt.axis('equal')
    plt.xlim(-10, 200)
    plt.ylim(-10, 200)
    plt.savefig(fname.split('.')[0]+str(i)+'.png')
    plt.show()
    








# Save everything
# Mammr.visual.vertex_colors = [  200, 66,  48, 255]
# Mammr.export(mammrfname[:-3]+'morph.obj')
# original_stdout = sys.stdout
# with open('res.txt','wt') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(res)
#     sys.stdout = original_stdout
# radii = np.linalg.norm(mesh.vertices - mesh.center_mass, axis=1)

# # Assign colors to vertices based on curvature
# mesh.visual.vertex_colors = tri.visual.interpolate(radii, color_map='viridis')
# mesh.show()



