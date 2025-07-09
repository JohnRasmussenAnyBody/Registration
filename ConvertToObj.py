# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:30:20 2023

@author: jr
"""

import os
import numpy as np
import trimesh as tri

# Search for all stl files and convert them to obj
# with different colors
directory = '.'     
for root, dirs, files in os.walk(directory):
    for filename in files:
        fname = filename.lower()
        if fname.split('.')[-1] == 'stl':
            fullname = os.path.join(root,fname[:-4]+'.obj')
            mesh = tri.load_mesh(os.path.join(root,fname))
            mesh.visual.vertex_colors = np.append(np.int_(np.random.rand(3)*255),128)
            mesh.export(fullname)
            print(fullname)

