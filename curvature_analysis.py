#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:40:28 2024

@author: shima
"""

import numpy as np
import os
from PIL import Image
import trimesh as tri
import matplotlib.pyplot as plt
# import pymeshlab as ml
# import open3d as o3d
from scipy.linalg import polar
from scipy.spatial.transform import Rotation

def prepare(mesh):
    mesh_g = tri.load_mesh(mesh)

    # Temporarily align the mesh with the global coordinate system
    A = mesh_g.principal_inertia_transform
    mesh_g.apply_transform(A)
    return (mesh_g)

radius = 0.2
def calculate_curvature(mesh, radius=1):
    """Calculate the mean curvature of the mesh."""
    vertices = mesh.vertices
    return abs(tri.curvature.discrete_mean_curvature_measure(mesh, vertices, radius))

def save_mesh_image(mesh, filename):
    """Save an image of the mesh to disk."""
    scene = mesh.scene()
    png = scene.save_image(resolution=(1024, 1024), visible=True)
    with open(filename, 'wb') as f:
        f.write(png)
        
def plot_mesh_with_color(mesh, curvatures, C1, C2, C3, C4, filename):
    colors = np.zeros((len(curvatures), 4))  # RGBA colors
    for i, curv in enumerate(curvatures):
        if C1 <= curv < C2:
            colors[i] = [1, 0, 0, 1]  # Red
        elif C2 <= curv < C3:
            colors[i] = [0, 1, 0, 1]  # Green
        elif C3 <= curv <= C4:
            colors[i] = [0, 0, 1, 1]  # Blue
    mesh.visual.vertex_colors = colors
    save_mesh_image(mesh, filename)

def visualize_curvature_thresholds(mesh, mean_curvature, C1, C4, step_size):
    steps = int((C4 - C1) / step_size) + 1
    fig, axes = plt.subplots(steps - 1, steps - 1, figsize=(2 * (steps - 1), 2 * (steps - 1)))
    for i in range(1, steps):
        C2 = C1 + i * step_size
        for j in range(i + 1, steps):
            C3 = C1 + j * step_size
            ax = axes[j - 1, i - 1] if (steps - 1) > 1 else axes
            filename = f'mesh_{i}_{j}.png'
            plot_mesh_with_color(mesh, mean_curvature, C1, C2, C3, C4, filename)
            image = Image.open(filename)
            ax.imshow(image)
            ax.set_title(f'C2={C2:.1f}, C3={C3:.1f}')
            ax.axis('off')
            os.remove(filename)  # Optional: remove the file after loading
    plt.tight_layout()
    plt.show()
    
def color_mesh_by_curvature(mesh, curvatures, C1, C2, C3, C4):
    # Initialize the vertex colors array
    colors = np.zeros((len(curvatures), 4))  # RGBA colors

    # Assign colors based on curvature thresholds
    for i, curv in enumerate(curvatures):
        if C1 <= curv < C2:
            colors[i] = [1, 0, 0, 1]  # Red
        elif C2 <= curv < C3:
            colors[i] = [0, 1, 0, 1]  # Green
        elif C3 <= curv <= C4:
            colors[i] = [0, 0, 1, 1]  # Blue

    # Apply the colors to the mesh
    mesh.visual.vertex_colors = colors

    return mesh
    

## Example
mesh = prepare('ps_d.obj')
mean_curvature = calculate_curvature(mesh)
C1 = np.min(mean_curvature)
C4 = np.max(mean_curvature)
step_size = 0.2
# visualize_curvature_thresholds(mesh, mean_curvature, C1, C4, step_size)

## Visualize the mesh in 3D

#Example

colored_mesh = color_mesh_by_curvature(mesh, mean_curvature, C1, 0.25, 1.5, C4)
colored_mesh.show()
