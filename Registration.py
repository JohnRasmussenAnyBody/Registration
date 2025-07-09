# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:00:35 2021

@author: jr
"""

import numpy as np
import os
import trimesh as tri
# import pymeshlab as ml
# import open3d as o3d

os.chdir(r'C:\Users\jr\OneDrive - Aalborg Universitet\Documents\Data\Registration')

mesh1 = tri.load_mesh('OsCoxaeLeftFemale.stl') + tri.load_mesh('OsCoxaeRightFemale.stl') + tri.load_mesh('SacrumFemale.stl')
mesh1.visual.vertex_colors = [  2, 252,  66, 255]
mesh2 = tri.load_mesh('OsCoxaeLeftMale.stl') + tri.load_mesh('OsCoxaeRightMale.stl') + tri.load_mesh('SacrumMale.stl')
mesh2.visual.vertex_colors = [  2,  81, 252, 255]

(mesh1+mesh2).show()

A = tri.registration.mesh_other(mesh1,mesh2)

mesh3 = mesh1.apply_transform(A[0])
mesh3.visual.vertex_colors = [158, 252,   2, 255]

(mesh2+mesh3).show()


