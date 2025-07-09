# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:47:08 2024

This program identifies different features on trimesh meshes representing
femurs. The program is typically used on the base bone, and the mapping
of the base to the target bones will update the parameters for these bones.

It is not fully automated but requires the user to manually isolate relevant
parts of the base mesh, i.e., the head, the condyles, the neck, and the shaft,
and save them as templates for fitting of cylinder and spheres.

@author: jr
"""
import numpy as np
import pandas as pd
import trimesh as tri
import meshtools as mt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

"""
Generate a cylinder mesh using fit_cylinder_least_squares output.

Parameters:
params (array): Cylinder parameters [x0, y0, z0, ux, uy, uz, radius].
height (float): Height of the cylinder.
sections (int): Number of sections for the cylinder mesh (default is 32).

Returns:
trimesh.Trimesh: The generated cylinder mesh, oriented and translated.
"""
def generate_fitted_cylinder(params, height, sections=32):
    x0, y0, z0, ux, uy, uz, radius = params

    # Create a unit cylinder along the Z-axis
    cylinder = tri.creation.cylinder(radius=radius, height=height, sections=sections)

    # Axis of the cylinder
    axis = np.array([ux, uy, uz])
    axis /= np.linalg.norm(axis)  # Ensure it's a unit vector

    # Align the cylinder along the axis direction
    z_axis = np.array([0, 0, 1])  # Default cylinder direction in trimesh
    rotation_matrix = tri.geometry.align_vectors(z_axis, axis)
    cylinder.apply_transform(rotation_matrix)

    # Translate the cylinder to the center position
    cylinder.apply_translation([x0, y0, z0])

    return cylinder

"""
This function takes a set of points and assumes that they are located
on a sphere. The function returns the center and radius of the sphere.
"""
def fit_sphere_least_squares(p):
    # Prepare the data for least squares
    A = np.hstack((2 * p, np.ones((p.shape[0], 1))))
    b = np.sum(p**2, axis=1)

    # Solve the least squares problem A * [x0, y0, z0, r^2 - x0^2 - y0^2 - z0^2] = b
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Extract the center coordinates (x0, y0, z0) and compute radius
    x0, y0, z0, d = coeff
    r = np.sqrt(d + x0**2 + y0**2 + z0**2)

    return np.array([x0, y0, z0]), r

"""
This is the objective function of the fit_cylinder_least_squares
function. It computes the sum of squared distances between points and the cylinder.

Parameters:
params (array): Cylinder parameters [x0, y0, z0, ux, uy, uz, radius].
points (array): Points to be fitted, shape (n, 3).

Returns:
float: Sum of squared distances from points to the surface of the cylinder.
"""
def cylinder_distance(params, points):

    x0, y0, z0, ux, uy, uz, radius = params
    cylinder_axis = np.array([ux, uy, uz])
    cylinder_axis /= np.linalg.norm(cylinder_axis)  # Ensure it's a unit vector

    # Vector from the cylinder axis origin to the points
    vec_from_axis = points - np.array([x0, y0, z0])
    
    # Project vec_from_axis onto the cylinder axis to find the closest point on the axis
    projection_length = np.dot(vec_from_axis, cylinder_axis)
    closest_point_on_axis = np.outer(projection_length, cylinder_axis)
    
    # Perpendicular distance from the axis
    perp_vec = vec_from_axis - closest_point_on_axis
    perp_distance = np.linalg.norm(perp_vec, axis=1)
    
    # Distance from the cylinder surface
    distance_to_surface = np.abs(perp_distance - radius)
    
    return np.sum(distance_to_surface**2)

"""
This is a utility function for fit_cylinder_least_squares. It adresses the
problem that the location of the cylinder center point along the cylinder's
axis is undetermined by the optimization, so we forcefully align it with the 
centroid of the point cloud

Parameters:
centroid (array): The centroid of the point cloud.
axis_point (array): A point on the cylinder axis (initial guess for x0, y0, z0).
axis_direction (array): The direction of the cylinder axis (ux, uy, uz).

Returns:
array: Projected point (x0, y0, z0) on the cylinder axis.
"""
def project_centroid_to_axis(centroid, axis_point, axis_direction):

    axis_direction = axis_direction / np.linalg.norm(axis_direction)  # Ensure unit vector
    v = centroid - axis_point
    projection_length = np.dot(v, axis_direction)
    projected_point = axis_point + projection_length * axis_direction
    return projected_point

"""
Fit a cylinder to a 3D point cloud using least squares optimization, and ensure the
origin (x0, y0, z0) is as close as possible to the centroid of the points.

Parameters:
points (array): The point cloud with shape (n, 3).

Returns:
array: Fitted cylinder parameters [x0, y0, z0, ux, uy, uz, radius].
"""
def fit_cylinder_least_squares(points):

    # Initial guess: center of points, principal axis from PCA, and an estimated radius
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center)
    axis_guess = vh[0]  # First principal component as initial axis guess
    radius_guess = np.mean(np.linalg.norm(points - center, axis=1)) / 2

    initial_params = np.hstack([center, axis_guess, radius_guess])

    # Minimize the distance function to fit the cylinder
    result = minimize(cylinder_distance, initial_params, args=(points,))

    # Get optimized parameters
    x0, y0, z0, ux, uy, uz, radius = result.x

    # Project the centroid onto the cylinder axis
    axis_direction = np.array([ux, uy, uz])
    axis_point = np.array([x0, y0, z0])
    projected_point = project_centroid_to_axis(center, axis_point, axis_direction)

    # Update the origin of the cylinder with the projected point
    return np.hstack([projected_point, axis_direction, radius])

"""
Compute the sum of squared distances between points and the cylinder.
One end of the cylinder is anchored at 'anchor_point'.

Parameters:
params (array): Cylinder parameters [ux, uy, uz, radius].
points (array): Points to be fitted, shape (n, 3).
anchor_point (array): The anchor point for one end of the cylinder.

Returns:
float: Sum of squared distances from points to the surface of the cylinder.
"""
def cylinder_distance_anchor_point(params, points, anchor_point):

    ux, uy, uz, radius = params
    cylinder_axis = np.array([ux, uy, uz])
    cylinder_axis /= np.linalg.norm(cylinder_axis)  # Ensure it's a unit vector

    # Vector from the anchor point to the points
    vec_from_axis = points - anchor_point
    
    # Project vec_from_axis onto the cylinder axis to find the closest point on the axis
    projection_length = np.dot(vec_from_axis, cylinder_axis)
    closest_point_on_axis = np.outer(projection_length, cylinder_axis)
    
    # Perpendicular distance from the axis
    perp_vec = vec_from_axis - closest_point_on_axis
    perp_distance = np.linalg.norm(perp_vec, axis=1)
    
    # Distance from the cylinder surface
    distance_to_surface = np.abs(perp_distance - radius)
    
    return np.sum(distance_to_surface**2)

"""
This is used to fit a cylinder to the femoral neck, given knowledge of 
the hip joint center.

Parameters:
points (array): The array of 3D points with shape (n, 3).
anchor_point (array): The anchor point for one end of the cylinder.

Returns:
array: Fitted cylinder parameters [ux, uy, uz, radius].
"""
def fit_cylinder_anchor_point(points, anchor_point):

    # Estimate an initial guess using PCA (Principal Component Analysis)
    _, _, vh = np.linalg.svd(points - np.mean(points, axis=0))
    axis_guess = vh[0]  # First principal component as initial axis guess
    radius_guess = np.mean(np.linalg.norm(points - np.mean(points, axis=0), axis=1)) / 2

    initial_params = np.hstack([axis_guess, radius_guess])

    # Minimize the distance function to fit the cylinder, with the fixed anchor point
    result = minimize(cylinder_distance_anchor_point, initial_params, args=(points, anchor_point))

    return result.x

"""
Compute the sum of squared distances between points and the cylinder.
One end of the cylinder is anchored at 'anchor_point', and the other end
must lie on a line given by 'line_point' and 'line_direction'.

Parameters:
params (array): Cylinder parameters [radius, t], where t is the parameter on the line.
points (array): Points to be fitted, shape (n, 3).
anchor_point (array): The anchor point for one end of the cylinder.
line_point (array): A point on the line that constrains the other end.
line_direction (array): The unit direction vector of the line.

Returns:
float: Sum of squared distances from points to the surface of the cylinder.
"""
def cylinder_distance_constrained(params, points, anchor_point, line_point, line_direction):

    radius, t = params
    line_direction = line_direction / np.linalg.norm(line_direction)  # Ensure unit vector

    # Calculate the other end of the cylinder on the line
    other_end = line_point + t * line_direction

    # Compute the cylinder axis as the line from anchor_point to other_end
    cylinder_axis = other_end - anchor_point
    cylinder_axis /= np.linalg.norm(cylinder_axis)  # Normalize to get unit vector

    # Vector from the anchor point to the points
    vec_from_axis = points - anchor_point
    
    # Project vec_from_axis onto the cylinder axis to find the closest point on the axis
    projection_length = np.dot(vec_from_axis, cylinder_axis)
    closest_point_on_axis = np.outer(projection_length, cylinder_axis)

    # Perpendicular distance from the axis
    perp_vec = vec_from_axis - closest_point_on_axis
    perp_distance = np.linalg.norm(perp_vec, axis=1)
    
    # Distance from the cylinder surface
    distance_to_surface = np.abs(perp_distance - radius)

    return np.sum(distance_to_surface**2)

"""
Fit a cylinder to a set of 3D points with one end anchored to 'anchor_point'
and the other end constrained to lie on a line defined by 'line_point' and 'line_direction'.

Parameters:
points (array): The array of 3D points with shape (n, 3).
anchor_point (array): The anchor point for one end of the cylinder.
line_point (array): A point on the line that constrains the other end.
line_direction (array): The direction vector of the line.

Returns:
array: Fitted cylinder parameters [radius, t].
"""
def fit_cylinder_least_squares_constrained(points, anchor_point, line_point, line_direction):

    # Estimate initial guesses for radius and t (location on the line)
    radius_guess = np.mean(np.linalg.norm(points - np.mean(points, axis=0), axis=1)) / 2
    t_guess = np.linalg.norm(np.mean(points, axis=0) - line_point)

    initial_params = np.array([radius_guess, t_guess])

    # Minimize the distance function to fit the cylinder
    result = minimize(cylinder_distance_constrained, initial_params, 
                      args=(points, anchor_point, line_point, line_direction),
                      options={'disp': True})

    # Extract the radius and t parameter
    radius, t = result.x
    
    # Compute the other end of the cylinder based on the optimized t
    other_end = line_point + t * line_direction

    # Compute the cylinder axis as the line from anchor_point to other_end
    cylinder_axis = other_end - anchor_point
    cylinder_axis /= np.linalg.norm(cylinder_axis)  # Normalize to get unit vector

    return radius, t, cylinder_axis

def cylinder_distance_with_anchor_xy(params, points, line_point, line_direction):
    """
    Compute the sum of squared distances between points and the cylinder.
    The x and y coordinates of the anchor point are design variables, and the
    other end must lie on a line given by 'line_point' and 'line_direction'.
    
    Parameters:
    params (array): Cylinder parameters [x0, y0, radius, t], where t is the parameter on the line.
    points (array): Points to be fitted, shape (n, 3).
    line_point (array): A point on the line that constrains the other end.
    line_direction (array): The unit direction vector of the line.
    
    Returns:
    float: Sum of squared distances from points to the surface of the cylinder.
    """
    x0, y0, radius, t = params
    anchor_point = np.array([x0, y0, 0])  # z0 is fixed at 0
    line_direction = line_direction / np.linalg.norm(line_direction)  # Ensure unit vector

    # Calculate the other end of the cylinder on the line
    other_end = line_point + t * line_direction

    # Compute the cylinder axis as the line from anchor_point to other_end
    cylinder_axis = other_end - anchor_point
    cylinder_axis /= np.linalg.norm(cylinder_axis)  # Normalize to get unit vector

    # Vector from the anchor point to the points
    vec_from_axis = points - anchor_point
    
    # Project vec_from_axis onto the cylinder axis to find the closest point on the axis
    projection_length = np.dot(vec_from_axis, cylinder_axis)
    closest_point_on_axis = np.outer(projection_length, cylinder_axis)

    # Perpendicular distance from the axis
    perp_vec = vec_from_axis - closest_point_on_axis
    perp_distance = np.linalg.norm(perp_vec, axis=1)
    
    # Distance from the cylinder surface
    distance_to_surface = np.abs(perp_distance - radius)

    return np.sum(distance_to_surface**2)

def fit_cylinder_least_squares_with_anchor_xy(points, line_point, line_direction):
    """
    Fit a cylinder to a set of 3D points with the x and y coordinates of the anchor point as design variables,
    and the other end constrained to lie on a line defined by 'line_point' and 'line_direction'.

    Parameters:
    points (array): The array of 3D points with shape (n, 3).
    line_point (array): A point on the line that constrains the other end.
    line_direction (array): The direction vector of the line.

    Returns:
    array: Fitted cylinder parameters [x0, y0, radius, t].
    """
    # Estimate initial guesses for x0, y0, radius and t (location on the line)
    anchor_guess = np.mean(points, axis=0)[:2]  # Use mean x, y values of points for initial anchor guess
    radius_guess = np.mean(np.linalg.norm(points - np.mean(points, axis=0), axis=1)) / 2
    t_guess = np.linalg.norm(np.mean(points, axis=0) - line_point)

    initial_params = np.hstack([anchor_guess, radius_guess, t_guess])

    # Minimize the distance function to fit the cylinder
    result = minimize(cylinder_distance_with_anchor_xy, initial_params, 
                      args=(points, line_point, line_direction), options={'disp': True})

    # Extract the x0, y0, radius, and t parameter
    x0, y0, radius, t = result.x
    
    # Compute the other end of the cylinder based on the optimized t
    anchor_point = np.array([x0, y0, 0])  # z0 is fixed at 0
    other_end = line_point + t * line_direction

    # Compute the cylinder axis as the line from anchor_point to other_end
    cylinder_axis = other_end - anchor_point
    cylinder_axis /= np.linalg.norm(cylinder_axis)  # Normalize to get unit vector

    return x0, y0, radius, t, cylinder_axis


# Parameters and names of template meshes
directory = 'C:/Users/jr/Documents/GitHub/Registration/Femurs'
base = 'tlem2'
head = '/Templates/Tlem2Head'
condyles = '/Templates/Tlem2Condyles'
neck = '/Templates/Tlem2Neck'
shaft = '/Templates/Tlem2Shaft'
baseside = 'R'

# Fit a sphere to the head
Bb = tri.load_mesh(directory+'/'+head+'.obj')
c, r = fit_sphere_least_squares(Bb.vertices)
print('Fitted head parameters (x0, y0, z0, radius):', c, r)

# Create a center sphere
sphere = tri.creation.icosphere(subdivisions=3, radius=r)
sphere.apply_translation(c)
mt.savemesh(sphere,directory+'/head.obj')

# Fit a cylinder to the condyles
Bb = tri.load_mesh(directory+'/'+condyles+'.obj')
ecw = 92.5 #Epicondyle width measured manually.
cylinder_params = fit_cylinder_least_squares(Bb.vertices)
print("Fitted cylinder parameters (x0, y0, z0, ux, uy, uz, radius):", cylinder_params)
cylinder = generate_fitted_cylinder(cylinder_params, ecw)
mt.savemesh(cylinder,directory+'/cylinder.obj')

# Knee axis end points
ck = np.array(cylinder_params[0:3])
e = np.array(cylinder_params[3:6])
uknee = e/np.linalg.norm(e)
ckm = ck+uknee*ecw/2
ckl = ck-uknee*ecw/2

# Fit a cylinder to the femoral neck. One end of it is ficed to the hip center
Bb = tri.load_mesh(directory+'/'+neck+'.obj')
params = fit_cylinder_anchor_point(Bb.vertices,c)
cylinder_params = c.tolist()+list(params)
print("Fitted neck cylinder parameters (x0, y0, z0, ux, uy, uz, radius):", cylinder_params)
length = 150
uneck = -np.array(cylinder_params)[3:6]
p2 = c + length*uneck
pm = (c+p2)/2
new_params = cylinder_params
new_params[0:3] = pm
cylinder = generate_fitted_cylinder(new_params, length)
mt.savemesh(cylinder,directory+'/neckcylinder.obj')

# Fit a cylinder to the shaft. This cylinder is anchored to the knee joint
# center by one end. The other end is constrained to the neck cylinder center 
# line.
Bb = tri.load_mesh(directory+'/'+shaft+'.obj')
#r, t, axis  = fit_cylinder_least_squares_constrained(Bb.vertices, ck, c, u)
x0, y0, r, t, axis = fit_cylinder_least_squares_with_anchor_xy(Bb.vertices, c, uneck)
cn = c + t*uneck  # Neck kink point
ckk = np.array([x0,y0,ck[2]])
shaftcyl = mt.create_cyl(ckk,cn,radius=r,extend=1.0)
mt.savemesh(shaftcyl,directory+'/shaftcylinder.obj')

# The shaft cylinder does not begin at the knee joint point, so we must
# construct a new unit vector from ck to cn
ushaft = (cn-ck)/np.linalg.norm(cn-ck)

# Compute angles
rneck, rmsd = Rotation.align_vectors([ushaft], [uneck])
aneck = rneck.magnitude()

rantetor, rmsd = Rotation.align_vectors([uknee], [uneck])
antetor = rantetor.as_euler('zyx')[0]

# Update the femurs table with the identified parameters for the base bone
femurs = pd.read_excel('Femurs/femurs.xlsx',index_col=0)

# Hip
femurs.loc[base,'HJCx'] = c[0]
femurs.loc[base,'HJCy'] = c[1]
femurs.loc[base,'HJCz'] = c[2]
femurs.loc[base,'HJr'] = r

# Knee joint
femurs.loc[base,'KJCmx'] = ckm[0]
femurs.loc[base,'KJCmy'] = ckm[1]
femurs.loc[base,'KJCmz'] = ckm[2]
femurs.loc[base,'KJClx'] = ckl[0]
femurs.loc[base,'KJCly'] = ckl[1]
femurs.loc[base,'KJClz'] = ckl[2]
femurs.loc[base,'KJr'] = cylinder_params[6]

# Neck point
femurs.loc[base,'NCx'] = cn[0]
femurs.loc[base,'NCy'] = cn[1]
femurs.loc[base,'NCz'] = cn[2]

# Various parameters
femurs.loc[base,'FL'] = np.linalg.norm(c-(ckm+ckl)/2)  # Functional length
femurs.loc[base,'KEW'] = np.linalg.norm(ckm-ckl)  # Knee width
femurs.loc[base,'Antetor'] = antetor  # Antetorsion angle
femurs.loc[base,'NA'] = aneck  # Neck angle

femurs.to_excel('Femurs/femurs.xlsx')

# Add the functional axis
funcaxis = mt.create_cyl(ck,c,extend=1.2)
kneeaxis = mt.create_cyl(ckm,ckl,extend=1.2)
shaftaxis = mt.create_cyl(ck,cn,extend=1.2)
neckaxis = mt.create_cyl(c,cn,extend=1.2)

rig = funcaxis+kneeaxis+shaftaxis+neckaxis

rig.export(directory+'/rigs/'+base+'_rig.obj')
