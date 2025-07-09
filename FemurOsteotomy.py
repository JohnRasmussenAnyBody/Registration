# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 2024

This program performs a virtual osteotomy on a Femur given specs
used in the statistical model

@author: jr
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import trimesh as tri
import meshtools as mt
from sillywalk import PCAModel

# Read existing list of femurs
femurs = pd.read_excel('Femurs/femurs.xlsx',index_col=0)
base = 'tlem2'
bone = '01L_Femur'

# Copy to dataframe Y minus columns that are not useful for PCA
droplist = ['Side']
Y = pd.DataFrame(femurs.drop(droplist, axis=1),dtype=float)
M1 = Y.loc[bone]
Mb = Y.loc[base]

pca = PCAModel(Y.mean(),Y.std(),Y.corr(numeric_only=True))
eigen = np.abs(pd.DataFrame(pca._eigvec)[2]).sort_values(ascending=False)

# Create a mapping from the base bone to the desired shape
f = {'HJr': 16.3540459838485,
     'KJr': 17.8220295718482,
     'Antetor': 5.0,
     'FL': 330.426647537482,
     'KEW': 90.6941519091907,
     'NA': 130,
     'Valgus': 0.0,
     }
M2 = pd.Series(pca.closest_pca(f))

# The mapping from bone to new bone is the difference between M2 and M1.
# However, only rigid transformations are allowed; no scaling and no shear!
minusset = ['ProxRx','ProxRy','ProxRz',
            'ProxTx','ProxTy','ProxTz',
            'DistRx','DistRy','DistRz',
            'DistTx','DistTy','DistTz']

wholeset = []
for key in Mb.index:
    if 'Prox' in key or 'Dist' in key:
        wholeset.append(key)

# Initiate with identity transformation, i.e., the base
M = Mb[wholeset]

# Subtract M1
for key in minusset:
    M[key] = M2[key] - M1[key]

# Load bone
B = tri.load_mesh('Femurs/'+bone+'.obj')

Bnew = mt.transform_femur(B, M, sharp=True, offset=-0.25)

mt.savemesh(Bnew,'Bnew.obj')
