# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:12:10 2024

Tools for mesh manipulation

@author: jr
"""
import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm, svd, eigvalsh
from scipy.optimize import minimize
from scipy.special import expit
import trimesh as tri

def heaviside(alpha, sharp=False, offset=0.0):
    if sharp:
        factor = 100.0
    else:
        factor = 12
    x = (alpha-0.5+offset)*factor
    return expit(x)

def randomcolor():
    color = (np.random.rand(3)*255).astype(int)
    np.append(color,255)  # Use opague color
    return color

"""
Apply a homogeneous affine transformation to a 3D point.

Parameters:
matrix (np.ndarray): A 4x4 homogeneous transformation matrix.
point (np.ndarray or list): A 3D point [x, y, z].

Returns:
np.ndarray: The transformed 3D point.
"""
def point_affine_transformation(matrix, point):
    # Convert the point to homogeneous coordinates by appending 1
    point_homogeneous = np.append(point, 1)

    # Apply the affine transformation matrix
    transformed_point_homogeneous = np.dot(matrix, point_homogeneous)

    # Convert back to 3D by discarding the homogeneous coordinate
    transformed_point = transformed_point_homogeneous[:3]

    return transformed_point

# Given a 4x4 affine transformation matrix A, this function returns the 
# inverse transformation matrix
def inverse_affine_transformation(A):
    # Separate the linear part (top-left 3x3) and the translation part (top-right 3x1)
    M = A[:3, :3]
    t = A[:3, 3]
    
    # Compute the inverse of the linear part
    M_inv = np.linalg.inv(M)
    
    # Compute the inverse translation
    t_inv = -np.dot(M_inv, t)
    
    # Construct the inverse affine transformation matrix
    A_inv = np.eye(4)
    A_inv[:3, :3] = M_inv
    A_inv[:3, 3] = t_inv
    
    return A_inv

"""
Extract the rigid transformation (rotation and translation) from an affine transformation matrix.

Parameters:
A (numpy.ndarray): 4x4 affine transformation matrix.

Returns:
numpy.ndarray: 4x4 rigid transformation matrix.
"""
def extract_rigid_transformation(A):

    # Extract the upper-left 3x3 submatrix (rotation + scaling/shearing)
    M = A[:3, :3]
    
    # Use SVD to extract the closest orthogonal matrix (rotation matrix)
    U, _, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    
    # Extract the translation vector
    t = A[:3, 3]
    
    # Construct the rigid transformation matrix
    rigid_transformation = np.eye(4)
    rigid_transformation[:3, :3] = R
    rigid_transformation[:3, 3] = t
    
    return rigid_transformation

"""
Decompose an affine transformation matrix into rigid and non-rigid components.

Parameters:
A (numpy.ndarray): 4x4 affine transformation matrix.

Returns:
numpy.ndarray: 4x4 rigid transformation matrix (rotation + translation).
numpy.ndarray: 4x4 non-rigid transformation matrix (scaling + shearing).
"""
def decompose_affine_transformation(A):
    # Extract the upper-left 3x3 submatrix (rotation + scaling/shearing)
    M = A[:3, :3]
    
    # Use SVD to decompose M into U, S, and Vt
    U, S, Vt = np.linalg.svd(M)
    
    # The rigid transformation (rotation)
    R = np.dot(U, Vt)
    
    # The non-rigid transformation (scaling + shearing)
    N = np.dot(np.linalg.inv(R), M)
    
    # Extract the translation vector
    t = A[:3, 3]
    
    # Construct the rigid transformation matrix
    rigid_transformation = np.eye(4)
    rigid_transformation[:3, :3] = R
    rigid_transformation[:3, 3] = t
    
    # Construct the non-rigid transformation matrix
    non_rigid_transformation = np.eye(4)
    non_rigid_transformation[:3, :3] = N
    
    return rigid_transformation, non_rigid_transformation

"""
Normalize the rotation vector to ensure the rotation angle is in [0, 2*pi].
If the angle is greater than 2*pi, subtract 2*pi until it is in the right 
interval

Parameters:
rotvec (numpy.ndarray): Rotation vector.

Returns:
numpy.ndarray: Normalized rotation vector.
"""
def normalize_rotation_vector(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 2*np.pi:
        return rotvec
    else:
        e = 2*np.pi*rotvec/angle # 2pi vector
        while angle > 2*np.pi:
            rotvec -= e
            angle = np.linalg.norm(rotvec)
        rotvec = (2 * np.pi - angle) * (-rotvec / angle)
    return rotvec

"""
Ensure that two rotation vectors are consistent in direction for interpolation.
This mitigates the risk of interpolation errors due to the cyclic nature of angles.

Parameters:
rotvec1 (numpy.ndarray): First rotation vector.
rotvec2 (numpy.ndarray): Second rotation vector.

Returns:
numpy.ndarray: The adjusted second rotation vector.
"""
# def consistent_rotation_vector(rotvec1, rotvec2):
#     scalprod = np.dot(rotvec1, rotvec2)
#     if scalprod < 0:
#         angle = np.linalg.norm(rotvec2)
#         e = 2*np.pi*rotvec2/angle # 2pi vector
#         while angle > 2*np.pi or scalprod < 0:
#             rotvec2 -= e
#             angle = np.linalg.norm(rotvec2)
#             scalprod = np.dot(rotvec1, rotvec2)
#     return rotvec2

def coordinate_rotation_vectors(v1,v2):
    vlist = [v1,v2]

    # First, make sure that each vector is as short as possible
    for v in vlist:
        angle = np.linalg.norm(v)
        if angle > np.pi:
            v = v - 2*np.pi*v/angle
    
    # Then we search for the combination with smallest distance between the
    # vector ends
    angle1 = np.linalg.norm(v1)
    e1 = 2*np.pi*v1/angle1
    u1 = v1-e1
    angle2 = np.linalg.norm(v2)
    e2 = 2*np.pi*v2/angle2
    u2 = v2-e2
    dist = np.zeros((3,3))
    for i in range(3):
        uu1 = u1 + i*e1
        for j in range(3):
            uu2 = u2 + j*e2
            dist[i,j] = np.linalg.norm(uu1 - uu2)

    min_index_flat = np.argmin(dist)
    imin,jmin = np.unravel_index(min_index_flat, dist.shape)

    u1 = u1 + imin*e1
    u2 = u2 + jmin*e2
    return u1, u2

def interpolate_rotation_vectors(rotvec1, rotvec2, alpha):
    """
    Interpolate between two rotation vectors with consistency.
    
    Parameters:
    rotvec1 (numpy.ndarray): First rotation vector.
    rotvec2 (numpy.ndarray): Second rotation vector.
    alpha (float): Interpolation parameter between 0 and 1.
    
    Returns:
    numpy.ndarray: Interpolated rotation vector.
    """
    # Normalize and ensure consistency in the rotation vector direction
    # rotvec1 = normalize_rotation_vector(rotvec1)
    # rotvec2 = normalize_rotation_vector(rotvec2)

    u1, u2 = coordinate_rotation_vectors(rotvec1,rotvec2)
    
    # rotvec2 = consistent_rotation_vector(rotvec1, rotvec2)
    
    # Interpolate between the rotation vectors
    return (1 - alpha)*u1 + alpha*u2

def decompose_affine_matrix(A):
    """
    Decomposes a 4x4 affine transformation matrix into its scaling, rotation (as a vector),
    shearing (as a vector), and translation components.
    
    Parameters:
    A (numpy.ndarray): 4x4 affine transformation matrix.
    
    Returns:
    tuple: (scaling, rotation_vector, shearing_vector, translation)
    """
    # Extract the translation component
    translation = A[:3, 3]
    
    # Extract the linear transformation matrix
    linear_part = A[:3, :3]
    
    # Perform polar decomposition to separate rotation and shearing
    rotation_matrix, upper_triangular_matrix = polar(linear_part)
    
    # Convert the rotation matrix to a rotation vector (axis-angle representation)
    rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
    
    
    # Shearing can be represented by a vector of three independent parameters
    shear_xy = upper_triangular_matrix[0, 1] / upper_triangular_matrix[1, 1]
    shear_xz = upper_triangular_matrix[0, 2] / upper_triangular_matrix[2, 2]
    shear_yz = upper_triangular_matrix[1, 2] / upper_triangular_matrix[2, 2]
    shearing_vector = np.array([shear_xy, shear_xz, shear_yz])
    
    # Scaling factors are the diagonal elements of the upper triangular matrix
    scaling_factors = np.diag(upper_triangular_matrix)
    
    return scaling_factors, rotation_vector, shearing_vector, translation

def decompose_affine_matrix_svd(A):
    """
    Decomposes a 4x4 affine transformation matrix into its scaling, rotation (as a vector),
    shearing (as a vector), and translation components using SVD for improved stability.
    """
    # Extract the translation component
    translation = A[:3, 3]

    # Extract the linear transformation matrix
    linear_part = A[:3, :3]

    # Perform SVD to separate scaling and rotation
    U, S, Vt = svd(linear_part)
    rotation_matrix = np.dot(U, Vt)
    scaling_factors = S

    # Calculate shearing component (residual part after removing rotation and scaling)
    shear_matrix = np.dot(U.T, linear_part) - np.diag(S)
    shearing_vector = np.array([shear_matrix[0, 1], shear_matrix[0, 2], shear_matrix[1, 2]])

    # Convert the rotation matrix to a rotation vector (axis-angle representation)
    rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()

    return scaling_factors, rotation_vector, shearing_vector, translation


def recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation):
    """
    Recombines scaling, rotation (from a vector), shearing (from a vector), and translation into a 4x4 affine matrix.
    
    Parameters:
    scaling (numpy.ndarray): Scaling factors (3x1 vector).
    rotation_vector (numpy.ndarray): Rotation vector (3x1 vector).
    shearing_vector (numpy.ndarray): Shearing vector (3x1 vector).
    translation (numpy.ndarray): Translation vector (3x1).
    
    Returns:
    numpy.ndarray: 4x4 affine transformation matrix.
    """
    # Convert the rotation vector back to a rotation matrix
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    
    # Recreate the shearing matrix
    shearing_matrix = np.eye(3)
    shearing_matrix[0, 1] = shearing_vector[0] * scaling[1]
    shearing_matrix[0, 2] = shearing_vector[1] * scaling[2]
    shearing_matrix[1, 2] = shearing_vector[2] * scaling[2]
    
    # Recreate the linear part by combining scaling, rotation, and shearing
    linear_part = np.dot(rotation_matrix, np.dot(np.diag(scaling), shearing_matrix))
    
    # Create the full 4x4 affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = linear_part
    affine_matrix[:3, 3] = translation
    
    return affine_matrix

"""
Ensure that the 4x4 affine matrix A has a proper rotation component (det > 0).
If the rotation part is improper (i.e., det < 0), correct it by flipping one axis.
"""
def regularize_affine(A):
    R = A[:3, :3]
    det = np.linalg.det(R)
    if det < 0:
        # Flip the last column to correct the handedness
        R[:, -1] *= -1
        A[:3, :3] = R
    return A

"""
Wrapper for trimesh.registration.mesh_other that ensures the returned transform
has a proper rotation matrix (det > 0).

Returns:
    transform (4x4 ndarray): Corrected affine transformation matrix.
    aligned (ndarray): Aligned points (optional, same as ICP).
"""
def registration_mesh_other(*args, **kwargs):
    transform, aligned = tri.registration.mesh_other(*args, **kwargs)
    transform = regularize_affine(transform)
    return transform, aligned

"""
Wrapper for trimesh.registration.icp that ensures the returned transform
has a proper rotation matrix (det > 0).

Returns:
    transform (4x4 ndarray): Corrected affine transformation matrix.
    aligned (ndarray): Aligned points (optional, same as ICP).
"""
def registration_icp(*args, **kwargs):
    result = tri.registration.icp(*args, **kwargs)

    # Handle multiple return types
    if isinstance(result, np.ndarray):
        # Only transform returned
        transform = regularize_affine(result)
        return transform
    elif isinstance(result, (list, tuple)) and len(result) >= 1:
        # Multiple values returned
        transform = regularize_affine(result[0])
        return (transform,) + tuple(result[1:])
    else:
        raise ValueError("trimesh.registration.icp failed")

"""
Returns a 4x4 homogeneous transformation matrix for a given 3D translation vector.

Parameters:
trans (np.ndarray or list): A 3D translation vector [tx, ty, tz].

Returns:
np.ndarray: A 4x4 homogeneous transformation matrix.
"""
def translation_matrix(trans):

    # Create a 4x4 identity matrix
    T = np.eye(4)
    
    # Set the translation components in the matrix
    T[0, 3] = trans[0]
    T[1, 3] = trans[1]
    T[2, 3] = trans[2]
    
    return T

def interpolate_affine_matrices(A1, A2, alpha):
    """
    Interpolates between two 4x4 affine transformation matrices.
    
    Parameters:
    A1, A2 (numpy.ndarray): 4x4 affine transformation matrices.
    alpha (float): Interpolation parameter between 0 and 1.
    
    Returns:
    numpy.ndarray: 4x4 interpolated affine transformation matrix.
    """
    # Decompose both matrices into scaling, rotation (as a vector), shearing (as a vector), and translation
    scaling1, rotation_vector1, shearing_vector1, translation1 = decompose_affine_matrix(A1)
    scaling2, rotation_vector2, shearing_vector2, translation2 = decompose_affine_matrix(A2)
    
    # Interpolate the translation component
    translation_interp = (1 - alpha) * translation1 + alpha * translation2
    
    # Interpolate the scaling component
    scaling_interp = (1 - alpha) * scaling1 + alpha * scaling2
    
    # Interpolate the shearing component
    shearing_interp = (1 - alpha) * shearing_vector1 + alpha * shearing_vector2
    
    # Interpolate the rotation vector with normalization and consistency
    rotation_vector_interp = interpolate_rotation_vectors(rotation_vector1, rotation_vector2, alpha)
    
    # Recombine the components into a full affine matrix
    A_interp = recompose_affine_matrix(scaling_interp, rotation_vector_interp, shearing_interp, translation_interp)
    
    return A_interp

"""
Average two rotation matrices using SVD.

Parameters:
R1, R2 (numpy.ndarray): 3x3 rotation matrices.

Returns:
numpy.ndarray: 3x3 average rotation matrix.
"""
def average_rotations(R1, R2):
    # Compute the rotation matrix using SVD on the sum of the rotation matrices
    R = R1 + R2
    U, _, Vt = np.linalg.svd(R)
    R_avg = np.dot(U, Vt)
    
    return R_avg

"""
Compute an average of two affine transformation matrices.

Parameters:
A1, A2 (numpy.ndarray): 4x4 affine transformation matrices.

Returns:
numpy.ndarray: 4x4 average affine transformation matrix.
"""
def average_affine_transformations(A1, A2):
    # Extract the rotation and translation components
    R1 = A1[:3, :3]
    R2 = A2[:3, :3]
    t1 = A1[:3, 3]
    t2 = A2[:3, 3]
    
    # Average the translation components
    t_avg = (t1 + t2) / 2.0
    
    # Average the rotation matrices using SVD
    R_avg = average_rotations(R1, R2)
    
    # Construct the average affine transformation matrix
    A_avg = np.eye(4)
    A_avg[:3, :3] = R_avg
    A_avg[:3, 3] = t_avg
    
    return A_avg

"""
Logarithmic interpolation between two affine transformation matrices 
using matrix logarithm and exponential. The algorithm takes the
logarithms of two transformation matrices before interpolation.
This tends to make nonlinear properties linear and will create
a better interpolation.

Parameters:
A1, A2 (numpy.ndarray): 4x4 affine transformation matrices.
alpha (float): Interpolation factor between 0 and 1.

Returns:
numpy.ndarray: 4x4 interpolated affine transformation matrix.
"""
def logarithmic_interpolation(A1, A2, alpha):
    # Compute the matrix logarithms of the transformation matrices
    log_A1 = logm(A1)
    log_A2 = logm(A2)
    
    # Interpolate in the logarithmic space
    log_A_interp = (1 - alpha) * log_A1 + alpha * log_A2
    
    # Compute the matrix exponential of the interpolated matrix
    A_interp = expm(log_A_interp)
    
    return A_interp

def safe_logarithmic_interpolation(A1, A2, alpha):
    """
    Interpolates between two affine transformation matrices using matrix logarithm and exponential.
    
    Parameters:
    A1, A2 (numpy.ndarray): 4x4 affine transformation matrices.
    alpha (float): Interpolation factor between 0 and 1.
    
    Returns:
    numpy.ndarray: 4x4 interpolated affine transformation matrix.
    """
    # Compute the matrix logarithms of the transformation matrices
    log_A1 = logm(A1)
    log_A2 = logm(A2)
    
    # Ensure the signs are consistent in the diagonal for interpolation
    for i in range(3):
        if np.sign(log_A1[i, i]) != np.sign(log_A2[i, i]):
            log_A2[i, i] = -log_A2[i, i]  # Flip to match signs

    # Interpolate in the logarithmic space
    log_A_interp = (1 - alpha) * log_A1 + alpha * log_A2
    
    # Compute the matrix exponential of the interpolated matrix
    A_interp = expm(log_A_interp)
    
    return A_interp


# This function splits a mesh into two parts by intersection with the xy plane
# def splitmesh(mesh, cap=True):
#     plane_origin = [0, 0, 0]
#     plane_normal = np.array([0, 0, 1])
    
#     # Use trimesh's slice_plane to intersect the mesh with the XY plane
#     # This function will return the intersection curve and the two resulting submeshes
#     side1 = mesh.slice_plane(plane_origin, plane_normal, cap=cap)        
#     side2 = mesh.slice_plane(plane_origin, -plane_normal, cap=cap)
    
#     return side1, side2

# This function splits a mesh into two parts by intersection with a plane
# parallel with the XY plane and skipping the middle alpha'th section
def splitmesh(mesh, alpha=0.0, cap=True):
    minz = mesh.bounds[0,2]
    maxz = mesh.bounds[1,2]
    middle = 0.5*(minz+maxz)
    lz = maxz-minz

    plane1_origin = [0, 0, middle+lz*alpha/2]
    plane2_origin = [0, 0, middle-lz*alpha/2]
    plane_normal = np.array([0, 0, 1])
    
    # Use trimesh's slice_plane to intersect the mesh with the XY plane
    # This function will return the intersection curve and the two resulting submeshes
    side1 = mesh.slice_plane(plane1_origin, plane_normal, cap=cap)        
    side2 = mesh.slice_plane(plane2_origin, -plane_normal, cap=cap)
    
    return side1, side2

"""
Computation of the squared sum of distances between corresponding vertices
in two meshes Bb and Bbt. The meshes must have identical structures. This works
as the objective function for optimization-based mesh registration.

Parameters:
B1 (trimesh.Trimesh): The first mesh.
B2 (trimesh.Trimesh): The second mesh.

Returns:
float: The squared sum of distances between corresponding vertices.
"""
def squared_sum_of_distances(B1, B2):

    # Ensure both meshes have the same number of vertices
    assert B1.vertices.shape == B2.vertices.shape, "Meshes must have the same number of vertices"

    # Compute the difference between corresponding vertices
    differences = B1.vertices - B2.vertices

    # Compute the squared distances
    squared_distances = np.sum(differences ** 2, axis=1)

    # Compute the sum of the squared distances
    squared_sum = np.sum(squared_distances)
    
    # print('Objective = ',squared_sum)

    return squared_sum

"""
An affine transformation given not by a 4x4 matrix but by separate vectors
for scaling, rotation, shearing and translation is applied to mesh. The
resulting mesh is returned. mesh is not modified.
"""
def apply_affine_transformation(mesh, scaling, rotation_vector, shearing_vector, translation):
    # Convert rotation vector to rotation matrix
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()

    # Create shearing matrix
    shearing_matrix = np.eye(3)
    shearing_matrix[0, 1] = shearing_vector[0] * scaling[1]
    shearing_matrix[0, 2] = shearing_vector[1] * scaling[2]
    shearing_matrix[1, 2] = shearing_vector[2] * scaling[2]

    # Apply scaling, rotation, and shearing
    linear_part = np.dot(rotation_matrix, np.dot(np.diag(scaling), shearing_matrix))

    # Apply the affine transformation to the vertices of mesh
    transformed_vertices = np.dot(mesh.vertices, linear_part.T) + translation

    # Create a new mesh with the transformed vertices
    transformed_mesh = tri.Trimesh(vertices=transformed_vertices, faces=mesh.faces, process=False)

    return transformed_mesh

"""
This is just an interface function that accepts a design variable vector,
separates it into affine mapping vectors, applies it to Bb and computes returns
the squared distance between Bb and Bbt.
"""
def objective_function(params, Bb, Bbt):
    # Decompose the params into scaling, rotation_vector, shearing_vector, translation
    scaling = params[0:3]
    rotation_vector = params[3:6]
    shearing_vector = params[6:9]
    translation = params[9:12]

    # Apply the affine transformation to Bb
    transformed_Bb = apply_affine_transformation(Bb, scaling, rotation_vector, shearing_vector, translation)

    # Compute the squared sum of distances
    return squared_sum_of_distances(transformed_Bb, Bbt)

"""
This function determines the optimum affine mapping of the base mesh Bb onto the
target Bt. The returned vector contains in sequence
3 scaling factors
3 components of a rotation vector
3 components of a shearing vector
3 translation vector components
We use this representation rather than the 4x4 affine matrix to better
separate the different mapping components.
"""
def optimize_affine_transformation(Bb, Bbt):
    assert Bb.vertices.shape == Bbt.vertices.shape, "optimize_affine_transformation: Meshes must have the same number of vertices"

    # Initial guess for parameters: identity transformation
    initial_params = np.zeros(12)
    initial_params[:3] = 1.0  # Scaling factors initialized to 1

    # Define bounds if necessary (e.g., to constrain scaling or translation)
    # bounds = [(0.5, 2.0)] * 3 + [(-np.pi, np.pi)] * 3 + [(-1, 1)] * 3 + [(-5, 5)] * 3

    # Minimize the objective function
    # Set options for the optimizer, including verbose output
    options = {
        'disp': True,  # Enable verbose output
        'maxiter': 1000  # Set the maximum number of iterations (optional)
    }
    # result = minimize(objective_function, initial_params, args=(Bb, Bbt), bounds=bounds, method='L-BFGS-B', options=options)
    result = minimize(objective_function, initial_params, args=(Bb, Bbt), method='L-BFGS-B')

    return result

"""
Assign a random color to a mesh and save it
"""
def savemesh(mesh, filename, color=None):
    # Assign color and export
    if isinstance(color, np.ndarray) and color.shape == (3,) and color.dtype == int:
        mesh.visual.vertex_colors = color
    else:
        mesh.visual.vertex_colors = randomcolor()
    mesh.export(filename)
    return

"""
Compute the normalized eigenvalues of the inertia tensor of a trimesh mesh.

Parameters:
mesh (trimesh.Trimesh): The input mesh.

Returns:
np.ndarray: Normalized eigenvalues of the inertia tensor (3,).
"""
def compute_normalized_inertia_tensor_eigenvalues(mesh):

    # Ensure the mesh is watertight for accurate mass properties
    assert mesh.is_watertight, "Mesh must be watertight to compute inertia tensor."

    # Compute the inertia tensor and mass properties
    inertia_tensor = mesh.principal_inertia_components
    volume = mesh.mass

    # Normalize the inertia tensor by the volume (to be scale-invariant)
    normalized_inertia_tensor = inertia_tensor / (volume ** (5/3))

    return normalized_inertia_tensor

"""
Aligns the mesh's principal axes with the coordinate system.

Parameters:
mesh (trimesh.Trimesh): The mesh to be aligned.

Returns:
trimesh.Trimesh: The aligned mesh.
    """
def align_principal_axes(mesh):
    A = mesh.principal_inertia_transform
    mesh.apply_transform(A)
    
    # The trans below is necessary because principal_inertia_transform
    # aligns to the centroud rather than the CoM
    A = tri.transformations.translation_matrix(-mesh.center_mass)
    mesh.apply_transform(A)
    return mesh

"""
Compute the n-th moment of a watertight mesh about its center of mass
with principal axes aligned with the coordinate system.

Parameters:
mesh (trimesh.Trimesh): The mesh representing the volume.
n (int): The order of the moment to compute.

Returns:
float: The n-th moment of the mesh distribution.
"""
def nth_moment_about_center_of_mass(mesh, n):
    assert mesh.is_watertight, "Mesh must be watertight to compute moments."

    # Compute and align the mesh to the principal axes
    aligned_mesh = mesh.copy()
    align_principal_axes(aligned_mesh)  # Align principal axes

    # Compute the n-th moment
    moment = 0.0

    for face in aligned_mesh.faces:
        vertices = aligned_mesh.vertices[face]
        
        # Calculate the normal and the area of the face
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        area = np.linalg.norm(normal) / 2.0
        normal = normal / np.linalg.norm(normal)
        
        # Centroid of the triangle (face)
        centroid = np.mean(vertices, axis=0)
        
        # Contribution of this face to the n-th moment
        moment += np.dot(centroid**(n+1), normal) * area
        
    moment /= 3*(n+1)
    
    return moment

def nth_moment_about_center_of_mass_normalized(mesh, n):
    moment = nth_moment_about_center_of_mass(mesh, n)
    # return moment
    return moment/(mesh.volume**(n/3+1))

def compute_3d_central_moments(mesh, order=3):
    """
    Compute the central moments of a 3D mesh up to a given order, 
    using the divergence theorem and summing over surface triangles.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.
    order (int): The highest order of moments to compute.

    Returns:
    np.ndarray: Central moments up to the specified order.
    """
    # Compute the center of mass (volume-based) for translation invariance
    center_of_mass = mesh.center_mass

    # Initialize central moments array
    central_moments = np.zeros((order + 1, order + 1, order + 1))

    # Iterate through the faces (triangles) of the mesh
    for face in mesh.faces:
        # Get the vertices of the triangle
        vertices = mesh.vertices[face]
        
        # Calculate the normal and the area of the face
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        area = np.linalg.norm(normal) / 2.0
        normal = normal / np.linalg.norm(normal)

        # Centroid of the triangle (face)
        centroid = np.mean(vertices, axis=0)

        # Shift the centroid to be relative to the center of mass
        xc, yc, zc = centroid - center_of_mass

        F = np.zeros(3)
        # Compute the contribution of this triangle to the central moments
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    F[0] = (xc**(i+1))*(yc**j)*(zc**k)/(i+1)
                    F[1] = (xc**i)*(yc**(j+1))*(zc**k)/(j+1)
                    F[2] = (xc**i)*(yc**j)*(zc**(k+1))/(k+1)
                    central_moments[i, j, k] += np.dot(F, normal) * area

    return central_moments

def normalize_moments(central_moments):
    """
    Normalize moments to achieve scale invariance.

    Parameters:
    central_moments (np.ndarray): The central moments of the mesh.

    Returns:
    np.ndarray: Normalized moments.
    """
    mu_000 = central_moments[0, 0, 0]  # The zeroth moment (volume)

    # Get the shape of the central moments
    order = central_moments.shape[0] - 1

    # Initialize normalized moments array
    normalized_moments = np.zeros_like(central_moments)

    # Normalize each moment based on its degree (i + j + k)
    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(order + 1):
                p = 1 + (i + j + k) / 3.0
                normalization_factor = mu_000**p
                normalized_moments[i, j, k] = central_moments[i, j, k] / normalization_factor

    return normalized_moments

def compute_hu_moments_3d(mesh):
    """
    Computes 3D Hu-like moments (rotation and scale invariant) for a mesh.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.

    Returns:
    np.ndarray: Array of Hu-like moments (5 invariants).
    """
    # Step 1: Compute central moments up to order 3
    central_moments = compute_3d_central_moments(mesh, order=3)

    # Step 2: Normalize moments for scale invariance
    normalized_moments = normalize_moments(central_moments)

    # Extract relevant moments for building Hu-like invariants
    mu_200 = normalized_moments[2, 0, 0]
    mu_020 = normalized_moments[0, 2, 0]
    mu_002 = normalized_moments[0, 0, 2]
    mu_110 = normalized_moments[1, 1, 0]
    mu_101 = normalized_moments[1, 0, 1]
    mu_011 = normalized_moments[0, 1, 1]
    mu_300 = normalized_moments[3, 0, 0]
    mu_030 = normalized_moments[0, 3, 0]
    mu_003 = normalized_moments[0, 0, 3]

    # Step 3: Compute Hu-like invariants
    I1 = mu_200 + mu_020 + mu_002  # Sum of second-order moments
    I2 = mu_200 * mu_020 + mu_020 * mu_002 + mu_002 * mu_200 - mu_110**2 - mu_101**2 - mu_011**2
    I3 = mu_300**2 + mu_030**2 + mu_003**2  # Higher-order moments

    # Combine moments to get invariants
    hu_moments = np.array([I1, I2, I3])

    return hu_moments

"""
Computes the normalized inertia tensor eigenvalues and Hu-like moments for a 3D mesh.

Parameters:
mesh (trimesh.Trimesh): The input mesh.

Returns:
dict: A dictionary containing the normalized inertia tensor eigenvalues and Hu-like moments.
"""
def compute_invariants(mesh):
    eigenvalues = compute_normalized_inertia_tensor_eigenvalues(mesh)
    hu_moments = compute_hu_moments_3d(mesh)

    return {
        'inertia_eigenvalues': eigenvalues,
        'hu_moments': hu_moments
    }

# This function maps the base femur to the target femur given by a dict of
# mapping maparemeters
# If sharp = True, the heaviside interpolation function becomes sharp.
# Offset moves the cut to either side compared with the middle
def transform_femur(Bb, row, sharp=False, offset=0):

    minz = Bb.bounds[0,2]
    maxz = Bb.bounds[1,2]
    lz = maxz-minz    
    
    # Form proximal matrix
    scaling = np.array([row['ProxSx'],row['ProxSy'],row['ProxSz']])
    rotation_vector = np.array([row['ProxRx'],row['ProxRy'],row['ProxRz']])
    shearing_vector = np.array([row['ProxSH1'],row['ProxSH2'],row['ProxSH3']])
    translation = np.array([row['ProxTx'],row['ProxTy'],row['ProxTz']])
    Aprox = recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)
    
    # Form distal matrix
    scaling = np.array([row['DistSx'],row['DistSy'],row['DistSz']])
    rotation_vector = np.array([row['DistRx'],row['DistRy'],row['DistRz']])
    shearing_vector = np.array([row['DistSH1'],row['DistSH2'],row['DistSH3']])
    translation = np.array([row['DistTx'],row['DistTy'],row['DistTz']])
    Adist = recompose_affine_matrix(scaling, rotation_vector, shearing_vector, translation)
    
    # Perform an interpolated mapping between the two ends
    vertices = Bb.vertices
    transformed_vertices = np.zeros_like(vertices)

    # Loop over each vertex
    for i, vertex in enumerate(vertices):
        # Convert the vertex to homogeneous coordinates
        vertex_homogeneous = np.append(vertex, 1)
        alpha = heaviside((vertex[2]-minz)/lz, sharp, offset)
        A = interpolate_affine_matrices(Adist, Aprox, alpha)
        
        # Transform the vertex using the modified matrix
        transformed_vertex_homogeneous = A @ vertex_homogeneous
        
        # Convert back to 3D coordinates
        transformed_vertices[i] = transformed_vertex_homogeneous[:3]

    # Create a new mesh with the transformed vertices and the same faces
    Bbtrans = tri.Trimesh(vertices=transformed_vertices, faces=Bb.faces)
    
    return(Bbtrans)

"""
Create a (thin) cylinder between two points. It will look as
though it is a line
"""
def create_cyl(point1, point2, radius=2, extend=1.0):
    
    # Create a cylinder along the Z-axis
    height = extend*np.linalg.norm(np.array(point2) - np.array(point1))
    cylinder = tri.creation.cylinder(radius=radius, height=height)
    
    # Calculate the direction vector of the line
    direction = np.array(point2) - np.array(point1)
    direction /= np.linalg.norm(direction)  # Normalize
    
    # Align the Z-axis of the cylinder with the direction vector
    z_axis = np.array([0, 0, 1])
    rotation_matrix = tri.geometry.align_vectors(z_axis, direction)
    cylinder.apply_transform(rotation_matrix)
    
    # Translate the cylinder to the correct position
    midpoint = (np.array(point1) + np.array(point2)) / 2
    cylinder.apply_translation(midpoint)
    cylinder.visual.vertex_colors = (255,0,0)
    
    return cylinder