# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 21:16:51 2025
@author: jr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import pickle

"""
Perform PCA on standardized data and return a DataFrame of principal components and the PCA object.
"""
def perform_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(data)
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pc_df = pd.DataFrame(pcs, index=data.index, columns=pc_columns)
    return pc_df, pca

"""
Extract integer labels from DataFrame index (last character assumed numeric).
"""
def extract_labels(index):
    return [int(ix[-1]) - 1 for ix in index]

"""
Fit LDA (linear discriminant analysis) in PC1-PC3 space and return separating 
direction in original PCA space.
"""
def project_lda_direction(pc_df, labels, pc1='PC1', pc2='PC3'):
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(pc_df[[pc1, pc2]], labels)
    return lda.coef_[0]

"""
Transform LDA direction from PCA space back to original standardized space.
"""
def compute_original_direction(pca_components, direction_pc, pcs=[0, 2]):
    return direction_pc[0] * pca_components[pcs[0]] + direction_pc[1] * pca_components[pcs[1]]

"""
Scatter plot of two principal components, with label highlighting.
"""
def plot_2d_scatter(pc_df, labels, label_strings, pc1='PC1', pc2='PC3', highlight_str='01L_Fe'):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pc_df[pc1], pc_df[pc2], c=labels, cmap='viridis', s=100)

    for i, label in enumerate(label_strings):
        if highlight_str in label:
            plt.text(pc_df.iloc[i][pc1], pc_df.iloc[i][pc2], label, fontsize=5, ha='right')

    plt.xlabel(pc1)
    plt.ylabel(pc2)
    plt.title(f'Scatter Plot of {pc1} vs {pc2} with Cluster Coloring')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.savefig('clusters.png', dpi=300)
    plt.show()

"""
Plot a 1D scatter projection with KDE curves for each cluster,
including a color legend and support for Matplotlib ≥3.7.
"""
def plot_1d_projection_with_density(projection, labels, index, highlight_str='01L_Fe'):
    projection = pd.Series(projection, index=index)
    labels = pd.Series(labels, index=index)

    unique_labels = sorted(labels.unique())
    n_labels = len(unique_labels)

    # Generate discrete colors from the colormap
    cmap = colormaps['viridis']
    colors = [cmap(i / (n_labels - 1)) if n_labels > 1 else cmap(0.5) for i in range(n_labels)]

    plt.figure(figsize=(10, 4))

    # 1. KDE curves
    for i, label_val in enumerate(unique_labels):
        sns.kdeplot(
            projection[labels == label_val],
            label=f'Label {label_val}',
            color=colors[i],
            fill=True,
            alpha=0.3,
            linewidth=2
        )

    # 2. 1D scatter
    plt.scatter(projection, [-0.01] * len(projection), c=labels, cmap='viridis', s=100)

    # 3. Annotate selected points
    for i, name in enumerate(index):
        if highlight_str in name:
            plt.text(projection.iloc[i], -0.03, name, fontsize=5, ha='right')

    # 4. Legend using same colors
    legend_elements = [Patch(facecolor=colors[i], label=f'Label {val}') for i, val in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, title='Cluster', loc='upper right')

    # Final touches
    plt.yticks([])
    plt.xlabel('Projection onto optimal separating direction (unscaled space)')
    plt.title('1D Scatter + Density Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('1d_projection_density.png', dpi=300)
    plt.show()

# --- MAIN ---
if __name__ == "__main__":

    # Load volume integral properties from bone ends for which we know the 
    # classification and standardize the data for forther procesing
    params = pd.read_excel('AllEndParameters.xlsx', index_col=0)
    scaler = StandardScaler()
    sparams = pd.DataFrame(scaler.fit_transform(params), index=params.index, columns=params.columns)
    
    # PCA and labeling
    pc_df, pca_obj = perform_pca(sparams, n_components=3)
    labels = extract_labels(params.index)
    
    # 2D PCA scatter
    plot_2d_scatter(pc_df, labels, params.index)
    
    # We assume that the points form two distinct clusters in the PC space.
    # To fnd the singl dimension that distinguishes the clusters, we
    # perform a linear discriminant analysis.
    direction_pc = project_lda_direction(pc_df, labels)
    direction_scaled = compute_original_direction(pca_obj.components_, direction_pc)
    
    # Wrap and print scaled weights
    direction_series = pd.Series(direction_scaled, index=sparams.columns)
    print("Linear combination of scaled original variables:")
    print(direction_series.sort_values(ascending=False))
    
    # Convert to unscaled space
    direction_unscaled = direction_series / scaler.scale_
    direction_unscaled = pd.Series(direction_unscaled, index=params.columns)
    
    # Save direction_unscaled to a pickle file. Classifications of new bones
    # are done by loading this object.
    with open('femur_classification_vector.pcl', 'wb') as f:
        pickle.dump(direction_unscaled, f)
    
    print("\nLinear combination of ORIGINAL variables:")
    print(direction_unscaled.sort_values(ascending=False))
    
    # Projection and 1D scatter plot with KDE curves
    projection_values = params @ direction_unscaled
    plot_1d_projection_with_density(projection_values, labels, params.index)
