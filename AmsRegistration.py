# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:00:35 2021

@author: jr
"""

import numpy as np
import os
import trimesh as tri

os.chdir(r'C:\Users\jr\OneDrive - Aalborg Universitet\Documents\Data\Registration')

# Read AMMR bone from file
# Mammr = tri.load_mesh('OsCoxaeLeftMale.stl') + tri.load_mesh('OsCoxaeRightMale.stl') + tri.load_mesh('SacrumMale.stl')
Mammr = tri.load_mesh('OsCoxaeRightMale.stl')
Mammr.visual.vertex_colors = [  2,  81, 252, 255]

# Read user bone from file
# Mu = tri.load_mesh('OsCoxaeLeftFemale.stl') + tri.load_mesh('OsCoxaeRightFemale.stl') + tri.load_mesh('SacrumFemale.stl')
Mu = tri.load_mesh('OsCoxaeRightFemale.stl')
Mu.visual.vertex_colors = [  2, 252,  66, 255]

(Mammr+Mu).show()

A = tri.registration.mesh_other(Mu,Mammr,scale=True)

Mr = Mu.apply_transform(A[0])
Mr.visual.vertex_colors = [158, 252,   2, 255]

(Mammr+Mr).show()

# Mt = tri.registration.nricp_amberg(Mammr, Mr)
# Mt.visual.vertex_colors = [200, 100,   50, 255]

# (Mammr+Mt).show()

