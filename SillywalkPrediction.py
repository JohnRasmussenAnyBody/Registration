# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 2024

This program uses SillyWalk to make general predictions based on
a table of data

@author: jr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh as tri
from sillywalk import PCAModel

# Read existing list of femurs
data = pd.read_excel('StudentData.xlsx',index_col=0)

# Copy to dataframe Y minus columns that are not useful for PCA
droplist = []
Y = pd.DataFrame(data.drop(droplist, axis=1),dtype=float)

pca = PCAModel(Y.mean(),Y.std(),Y.corr(numeric_only=True))

# eigen = np.abs(pd.DataFrame(pca._eigvec,index=Y.columns)[2]).sort_values(ascending=False)
eigen = np.abs(pd.DataFrame(pca._eigvec)[2]).sort_values(ascending=False)

# Sum up cumulative values
evars = 100*pca._eigval/pca._eigval.sum()
ecum = [evars[0]]
for evalue in evars[1:]:
    ecum.append(evalue+ecum[-1])
ecum = np.array(ecum)

# Number of terms to 90% explained variance
N = ecum[np.array(ecum) < 90].shape[0]

# Make bar plot
fig, ax = plt.subplots()
ind = np.arange(1,N+1)
width = 0.35
ax.bar(ind, (100*pca._eigval/pca._eigval.sum())[:N], width, bottom=0, label='Component')
ax.bar(ind+width, ecum[:N], width, bottom=0, label='Cumulative')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(ind)
# plt.bar(range(pca._eigval.shape[0]),100*pca._eigval/pca._eigval.sum())
# plt.bar(range(len(ecum)),ecum)
plt.xlabel('Principal Component Number')
plt.ylabel('Explained variance [%]')
ax.legend()
ax.autoscale_view()
plt.grid()
plt.show()

# Create a new subject
f = {
        'Sex': 1,
        'Stature': 167,
        'Bodyweight': 65,
     }
row = pca.closest_pca(f)

for ix in row.keys():
    print(ix,': ',row[ix])
