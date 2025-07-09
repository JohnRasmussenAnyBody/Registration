# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 19:30:57 2023

@author: jr
"""

import numpy as np
import trimesh as tri

Mammr = tri.load_mesh('Mfinal.obj')
Mr = tri.load_mesh('Mr.obj')

print('Start')
vol = tri.boolean.boolean_automatic([Mammr, Mr], 'difference').volume
print('volume =',vol)

print('Start')
vol = tri.boolean.boolean_automatic([Mr, Mammr], 'difference').volume
print('volume =',vol)
