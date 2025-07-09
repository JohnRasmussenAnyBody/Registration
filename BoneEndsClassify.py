# -*- coding: utf-8 -*-
"""
This program creates the foundation for classifying repectively
the proximal and distal ends of a femur. The program reads correctly
aligned femurs (prepared by the user) from a directory. For all these,
five integral properties of the two ends are computed, and the linear
combination of them that best distinguishes proximal from distal is
determined and saved.
Now, when a new femur appears, we can calculate the integral properties
of their ends, form thei linear combination and determine which is
which.

Created on Tue Sep  3 20:54:19 2024

@author: jr
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import pickle
import trimesh as tri
import meshtools as mt
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch

"""
This function is mainly used interactively when needed.
The automatic clustering offers no control of which cluster is the first, 
and which is the second. This can cause all bones to be swapped oppositely
of the base bone.
In this case, we must swap the cluster.
"""
def cluster_swap(pickle_file='kmeans_model.pcl'):
    """
    Load a KMeans model and scaler from a pickle file, swap the two clusters (0 <-> 1),
    and overwrite the pickle file with the modified model.
    
    Parameters:
    pickle_file (str): Path to the pickle file containing (kmeans, scaler).
    """
    with open(pickle_file, 'rb') as f:
        kmeans, scaler = pickle.load(f)

    if not isinstance(kmeans, KMeans) or not isinstance(scaler, StandardScaler):
        raise ValueError("Pickle file does not contain a (KMeans, StandardScaler) tuple.")

    if kmeans.n_clusters != 2:
        raise ValueError("This function only supports swapping for 2-cluster models.")

    # Swap the cluster centers
    kmeans.cluster_centers_ = kmeans.cluster_centers_[::-1]

    # If labels were ever assigned to data, also swap the internal label encoding
    kmeans.labels_ = 1 - kmeans.labels_

    # Swap the label-to-cluster-center mapping if it exists (used for prediction consistency)
    if hasattr(kmeans, 'predict'):
        # Remap the labels in the `kmeans._labels` if used internally
        # Prediction consistency is not guaranteed for custom label swaps
        kmeans._n_threads = getattr(kmeans, '_n_threads', 1)  # keep sklearn happy if needed

    # Save back to the pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump((kmeans, scaler), f)

    print(f"Swapped clusters and saved updated model to '{pickle_file}'.")

def perform_clustering(data):
    """
    Perform K-Means clustering to classify observations into two clusters,
    after standardizing the data.

    Parameters:
    data (pd.DataFrame): DataFrame containing the observations with parameters.

    Returns:
    kmeans (KMeans): Trained K-Means model with cluster centers.
    labels (pd.Series): Cluster labels for the original data.
    scaler (StandardScaler): The scaler used for normalization.
    """
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # Perform K-Means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(standardized_data)

    # Save the cluster centers, labels, and scaler
    data['Cluster'] = labels

    # Save the model and scaler to a file for future use
    with open('kmeans_model.pcl', 'wb') as f:
        pickle.dump((kmeans, scaler), f)

    return kmeans, pd.Series(labels, index=data.index), scaler

def classify_new_observations(new_data):
    """
    Classify new observations based on the existing K-Means clusters, using
    the saved model and scaler.

    Parameters:
    new_data (pd.DataFrame): DataFrame containing new observations with the same parameters.

    Returns:
    pd.Series: Cluster labels for the new observations.
    """
    # Load the trained K-Means model and scaler from file
    with open('kmeans_model.pcl', 'rb') as f:
        kmeans, scaler = pickle.load(f)

    # Standardize the new data using the same scaler
    standardized_new_data = scaler.transform(new_data)

    # Predict the cluster for the new data
    labels = kmeans.predict(standardized_new_data)
    
    # Take special action if the normal fit does not distinguish the ends.
    # In this case, we find the vector between cluster centers and the
    # vector between the bone end values and select based on the scalar product.
    if labels[0] == labels[1]:
        
        v1 = kmeans.cluster_centers_[1]-kmeans.cluster_centers_[0]
        v2 = standardized_new_data[1]-standardized_new_data[0]
    
        if np.linalg.vecdot(v1,v2) >= 0:
            labels[0] = 0
            labels[1] = 1
        else:
            labels[0] = 1
            labels[1] = 0
    
    # if labels[0] == labels[1]:
    #     distances = kmeans.transform(standardized_new_data)
    #     if np.max([distances[0,0],distances[1,1]]) < np.max([distances[1,0],distances[0,1]]):
    #         labels[0] = 0
    #         labels[1] = 1
    #     else:
    #         labels[0] = 1
    #         labels[1] = 0

    return pd.Series(labels, index=new_data.index)

def classify_new_observations_supervised(new_data, feature_columns=[]):
    with open('classifier_model.pcl', 'rb') as f:
        model, scaler = pickle.load(f)
        
    if len(feature_columns)==0:
        cols = new_data.columns
    else:
        cols = feature_columns

    X_new = new_data[cols]
    X_scaled = scaler.transform(X_new)

    predictions = model.predict(X_scaled)
    return pd.Series(predictions, index=new_data.index)

def train_classifier(data, feature_columns, labels):
    
    assert data.shape[0] == len(labels), "Mismatch of data and labels"
    
    """
    Trains a supervised classifier using ground truth cluster labels.

    Parameters:
    data (pd.DataFrame): Dataset including features and ground truth labels.
    feature_columns (list): Feature column names to use for classification.
    label_column (str): Name of the column with ground truth labels.

    Returns:
    model: Trained classification model.
    scaler: StandardScaler used to normalize the data.
    """
    X = data[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42) # 42 is just a random seed
    model.fit(X_scaled, labels)

    # Save model and scaler
    with open('classifier_model.pcl', 'wb') as f:
        pickle.dump((model, scaler), f)

    return model, scaler

def compute_end_params(mesh,femur=''):
    # Set up a dataframe with correct column and row names
    cols = []
    imoments = [2]
    for i in imoments:
        cols.append('I'+str(i))
    hmoments = [0,1]
    for i in hmoments:
        cols.append('H'+str(i))
    emoments = [0,1,2]
    for i in hmoments:
        cols.append('E'+str(i))
        
    indx = []
    indx.append(femur+'1')
    indx.append(femur+'2')
    
    params = pd.DataFrame(index=indx, columns=cols)

    m1, m2 = mt.splitmesh(mesh, 0.33)
    # hu_moments = mt.compute_hu_moments_3d(m1)
    for i in imoments:
         params.loc[femur+'1','I'+str(i)] = mt.nth_moment_about_center_of_mass_normalized(m1, i)
         params.loc[femur+'2','I'+str(i)] = mt.nth_moment_about_center_of_mass_normalized(m2, i)

    hu_moments1 = mt.compute_hu_moments_3d(m1)
    hu_moments2 = mt.compute_hu_moments_3d(m2)
    for i in hmoments:
         params.loc[femur+'1','H'+str(i)] = hu_moments1[i]
         params.loc[femur+'2','H'+str(i)] = hu_moments2[i]

    inertia1 = mt.compute_normalized_inertia_tensor_eigenvalues(m1)
    inertia2 = mt.compute_normalized_inertia_tensor_eigenvalues(m2)
    for i in emoments:
         params.loc[femur+'1','E'+str(i)] = inertia1[i]
         params.loc[femur+'2','E'+str(i)] = inertia2[i]
    
    return params

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


if __name__ == "__main__":
    
    # We predispose that directory contains a number of bones that we have
    # verified to be pre-aligned correctly
    directory = 'C:/Users/jr/Documents/GitHub/Registration/Femurs/aligned'

    objs = set()
    for filename in os.listdir(directory):
        if filename.endswith('.obj'):
            objs.add(filename[:-4])
            
    objs = sorted(objs)
        
    indx = []
    for bone in objs:
        indx.append(bone+'1')
        indx.append(bone+'2')
    
    for femur in objs:
        print(femur)
        mesh = tri.load_mesh(directory+'/'+femur+'.obj')
        if not mesh.is_watertight:
            print('Mesh ',femur,' is not watertight. Skipping.')
            continue
        
        # Compute the integral properties of the mesh's ends and
        # add them to the params dataframe. The column names come
        # from the function, so there is a bit of re-allocation the 
        # first time around
        par = compute_end_params(mesh,femur)
        
        # Allocate params the first time around
        if 'params' not in locals():
            params = pd.DataFrame(index=indx, columns=par.columns)
        else:
            if not isinstance(params, pd.DataFrame):
                params = pd.DataFrame(index=indx, columns=par.columns)
        
        for ix in par.index:
            params.loc[ix] = par.loc[ix]
    
    params.to_excel('AllEndParameters.xlsx')
    
    # Scale in preparation for PCA
    scaler = StandardScaler()
    sparams = pd.DataFrame(scaler.fit_transform(params), index=params.index, columns=params.columns)
    
    # PCA and labeling
    pc_df, pca_obj = perform_pca(sparams, n_components=3)
    labels = extract_labels(params.index)
    
    # 2D PCA scatter
    plot_2d_scatter(pc_df, labels, params.index)
    
    # We assume that the points form two distinct clusters in PC space.
    # To find the single dimension that distinguishes the clusters, we
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
