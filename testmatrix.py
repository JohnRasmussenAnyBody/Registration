# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 15:55:51 2025

@author: jr
"""

import numpy as np

A = np.array(
    [[-0.96463612, -0.2252663,  -0.13686578],
     [-0.23085721,  0.9726337,   0.02624179],
     [-0.12720888, -0.05691023,  0.99024195]]
    )

np.linalg.det(A)
