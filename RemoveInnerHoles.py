# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 11:13:21 2025

This utility program identifies and removes inner cavities of the mesh
in preparation for 

@author: jr
"""
import os
import glob
import trimesh
import pymeshlab

directory = 'C:/Users/jr/Documents/GitHub/Registration/Femurs'
ply = os.path.join(directory, 'ply')
stl = os.path.join(directory, 'stl')

# Ensure output directory exists
os.makedirs(stl, exist_ok=True)

# Get all .ply files
files = glob.glob(os.path.join(ply, '*.ply'))

for file in files:
    print(f'Processing: {file}')
    
    # Load and filter with trimesh
    mesh = trimesh.load(file)
    components = mesh.split(only_watertight=True)
    # largest = max(components, key=lambda m: m.volume if m.is_volume else len(m.faces))
    largest = max(components, key=lambda m: m.volume)
    
    # Export largest component to a temporary .ply for PyMeshLab
    tmp_path = file[:-4] + '_cleaned.ply'
    largest.export(tmp_path)
    
    # Load into PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(tmp_path)
    
    # Apply isotropic remeshing
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(2.0),  # Absolute edge length in world units
        iterations=10,
        adaptive=True,
    )
    
    # Save remeshed result to STL
    basename = os.path.splitext(os.path.basename(file))[0]
    output_path = os.path.join(stl, basename + '.stl')
    ms.save_current_mesh(output_path)
    
    # Clean up temporary .ply if desired
    os.remove(tmp_path)

