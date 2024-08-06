import os
import sys
import glob
import torch
import pickle
import logging
import subprocess
import imageio
import scipy.stats as stats
import numpy as np
import pandas as pd
import scipy.linalg
from tqdm import tqdm
from Bio import Phylo
from pathlib import Path
import torch.optim as optim
from datetime import datetime

from .config import LEARNING_RATE_INIT, TOTAL_EPOCHS, DIMENSION, NO_WEIGHT_RATIO, CURV_RATIO, WINDOW_RATIO, INCREASE_FACTOR,INCREASE_COUNT_RATIO,DECREASE_FACTOR, RESULTS_DIR
from torch import Tensor
from typing import Optional, Any, Tuple

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_mds(dataset_name, dimension=None):
    # Log and print the start of the Naive Hyperbolic Embedding Step
    logging.info("Naive Hyperbolic Embedding Step.")
    tqdm.write("Naive Hyperbolic Embedding Step.")

    # Log and print the name of the dataset being processed
    tqdm.write(f"Dataset: {dataset_name}")
    logging.info(f"Dataset: {dataset_name}")

    # Define the input directory for distance matrices
    input_directory = f'datasets/{dataset_name}/distance_matrices'
    # Check if the input directory exists
    if not os.path.exists(input_directory):
        tqdm.write("The input directory does not exist.")
        logging.info("The input directory does not exist.")
        return

    # Define the output directory for hyperbolic points
    output_directory = f'datasets/{dataset_name}/hyperbolic_points'
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_directory) if f.endswith(".npy")]

    # If no .npy files are found, log and exit
    if len(npy_files) == 0:
        tqdm.write("No .npy files found in the directory.")
        logging.info("No .npy files found in the directory.")
        return

    # Set up a progress bar for processing files
    with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            file_path = os.path.join(input_directory, filename)
            # Load the distance matrix from the .npy file
            matrix = np.load(file_path)
            
            # Compute the scale factor to normalize the matrix
            scale = 10 / np.max(matrix)
            output_scale_file_path = os.path.join(output_directory, f"scale_{filename[7:-4]}.npy")
            # Save the scale factor
            np.save(output_scale_file_path, np.array([scale]))

            # Scale the distance matrix
            matrix_scaled = scale * matrix
            # Compute the Gramian matrix
            gramian = -np.cosh(matrix_scaled)
            
            # Determine the number of points (N) and the embedding dimension
            N = np.shape(gramian)[0]
            if dimension is None:
                dimension = np.shape(gramian)[1] - 1  # Assume Gramian is square, and dimension is one less
            
            # Compute points from the Gramian matrix
            X = lgram_to_points(dimension, gramian)

            # Transform and project each vector in X to the hyperbolic space
            for n in range(N):
                x = X[:, n]
                X[:, n] = project_vector_to_hyperbolic_space(x)
            
            # Save the resulting X matrix
            output_X_file_path = os.path.join(output_directory, f"X_{filename[7:-4]}.npy")
            np.save(output_X_file_path, X)
            
            # Log the processed file
            logging.info(f"Processed file: {filename}")
            pbar.set_postfix({"file": filename})
            pbar.update(1)
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_log_torch(X):
    """
    Compute the hyperbolic logarithm map for given points in hyperbolic space.

    Parameters:
    X (Tensor): Points in hyperbolic space (DxN).

    Returns:
    Tensor: The tangent vectors at the base point (DxN).
    """
    
    # Get dimensions
    D, N = X.shape
    D -= 1  # Adjust D to match the context of the hyperbolic space
    
    # Define the base point as [1, 0, 0, ..., 0]
    base_point = torch.zeros(D + 1, dtype=X.dtype, device=X.device)
    base_point[0] = 1.0
    
    # Initialize the result matrix for tangent vectors
    tangent_vectors = torch.zeros((D + 1, N), dtype=X.dtype, device=X.device)
    
    
    for n in range(N):
        x = X[:, n]
        
        # Compute the theta value
        theta = -lorentzian_product(x,base_point)
        theta = torch.maximum(theta, torch.tensor(1.0, dtype=X.dtype, device=X.device))
        theta = torch.acosh(theta)
        
        # Compute the tangent vector
        if theta != 0:
            theta_over_sinh_theta = theta / torch.sinh(theta)
        else:
            theta_over_sinh_theta = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        
        tangent_vectors[:, n] = theta_over_sinh_theta * (x - base_point * torch.cosh(theta))
    
    return tangent_vectors[1:]
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_exponential_torch(V):
    """
    Compute the hyperbolic exponential map for given tangent vectors.

    Parameters:
    V (Tensor): Tangent vectors (DxN).

    Returns:
    Tensor: The resulting points in the hyperbolic space (DxN).
    """
    
    # Get dimensions
    D,N = V.shape

    zero_row = torch.zeros(1, V.shape[1], dtype=V.dtype, device=V.device)
    V = torch.cat((zero_row, V), dim=0)

    
    # Initialize the result matrix
    result_matrix = torch.zeros((D + 1, N), dtype=V.dtype, device=V.device)
    
    # Define the base point as [1, 0, 0, ..., 0]
    base_point = torch.zeros(D + 1, dtype=V.dtype, device=V.device)
    base_point[0] = 1.0

    
    for n in range(N):
        v = V[:, n]
        norm_v = torch.sqrt(J_norm_torch(v))
        

        if norm_v != 0:
            sinh_norm_v_over_norm_v = torch.sinh( norm_v) / norm_v
        else:
            sinh_norm_v_over_norm_v = torch.tensor(1.0, dtype=V.dtype, device=V.device)

        x = torch.cosh( norm_v) * base_point + sinh_norm_v_over_norm_v * v
        norm_x = J_norm_torch(x)
        
        if norm_x > 0:
            print(x)
            raise ValueError('Error: Norm should not be positive')
        
        # Normalize the result vector
        result_matrix[:, n] = x / torch.sqrt(-norm_x)
    
    return result_matrix
###########################################################################
def cost_function(tangent_vectors: Tensor, distance_matrix: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    """
    Compute the cost function for hyperbolic embeddings.

    Args:
        tangent_vectors (Tensor): Tangent vectors for optimization.
        distance_matrix (Tensor): The matrix of distances between points.
        **kwargs: Optional keyword arguments:
            - enable_scale_learning (bool): Flag to indicate if scale learning is enabled. Default is False.
            - scale_factor (Optional[Tensor]): Current scale factor. Default is None.
            - use_unweighted_cost (bool): Flag to indicate if the cost should be unweighted. Default is True.
            - weight_matrix (Optional[Tensor]): Weight matrix for weighted cost computation. Default is None.

    Returns:
        Tuple[Tensor, Tensor]: Computed cost value and updated scale factor.
    """
    enable_scale_learning = kwargs.get('enable_scale_learning', False)
    scale_factor = kwargs.get('scale_factor', torch.tensor(1.0, dtype=distance_matrix.dtype, device=distance_matrix.device))
    use_unweighted_cost = kwargs.get('use_unweighted_cost', True)
    weight_matrix = kwargs.get('weight_matrix', None)
    enable_save = kwargs.get('enable_save', False)
    
    embeddings = hyperbolic_exponential_torch(tangent_vectors)
    flipped_embeddings = embeddings.clone()
    flipped_embeddings[0, :] *= -1

    gram_matrix = torch.matmul(embeddings.t(), flipped_embeddings).clamp(max=-1)
    param_dist_matrix = torch.arccosh(-gram_matrix)
    off_diagonal_mask = ~torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)

    if enable_scale_learning:
        scale_factor = (param_dist_matrix.pow(2).sum() / 
                        (param_dist_matrix * distance_matrix).sum())

    if use_unweighted_cost:
        norm_diff = torch.norm(param_dist_matrix[off_diagonal_mask] - scale_factor * distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm(scale_factor * distance_matrix[off_diagonal_mask], p='fro') ** 2
    elif weight_matrix is not None:
        weighted_param_dist_matrix = param_dist_matrix * weight_matrix
        weighted_distance_matrix = distance_matrix * weight_matrix

        norm_diff = torch.norm(weighted_param_dist_matrix[off_diagonal_mask] - scale_factor * weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm(scale_factor * weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        print('norm scale is:', norm_scale)
    else:
        norm_diff = torch.tensor(0.0, dtype=distance_matrix.dtype, device=distance_matrix.device)
        norm_scale = torch.tensor(1.0, dtype=distance_matrix.dtype, device=distance_matrix.device)

    cost = norm_diff / norm_scale

    if enable_save:
        numerator = param_dist_matrix
        numerator.fill_diagonal_(1)
        
        denominator = scale_factor * distance_matrix
        denominator.fill_diagonal_(1)
        
        relative_error = torch.pow( torch.div(numerator,denominator )-1 ,2)
        relative_error[torch.eye(distance_matrix.size(0), dtype=torch.bool, device=distance_matrix.device)] = 0

        norm_diff = torch.norm(weighted_param_dist_matrix[off_diagonal_mask] - scale_factor * weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2
        norm_scale = torch.norm(scale_factor * weighted_distance_matrix[off_diagonal_mask], p='fro') ** 2


        return cost, scale_factor, relative_error

    return cost, scale_factor, None
###########################################################################
def lorentzian_product(x, y):
    return -x[0] * y[0] + torch.dot(x[1:], y[1:])
###########################################################################
def calculate_multipliers(loss_list, total_epochs):
    """
    Calculate learning rate multipliers based on loss trends.

    Args:
        loss_list (list): List of loss values.
        total_epochs (int): The total number of epochs.

    Returns:
        float: The product of calculated multipliers.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")
    
    WINDOW_SIZE = int(WINDOW_RATIO * total_epochs)
    INCREASE_COUNT_MAX = int(INCREASE_COUNT_RATIO * WINDOW_SIZE)
    multipliers = []
    for i in range(1, len(loss_list) + 1):
        if i < WINDOW_SIZE:
            multipliers.append(1.0)
            continue
        recent_losses = loss_list[i - WINDOW_SIZE:i]
        increase_count = sum(1 for x, y in zip(recent_losses, recent_losses[1:]) if y > x)
        if increase_count > INCREASE_COUNT_MAX:
            multipliers.append(DECREASE_FACTOR)
        elif all(x > y for x, y in zip(recent_losses, recent_losses[1:])):
            multipliers.append(INCREASE_FACTOR)
        else:
            multipliers.append(1.0)

    return np.prod(multipliers)
###########################################################################
def distance_range(matrix):
    """
    Calculate the range of distances in the matrix.

    Args:
        matrix (torch.Tensor): The distance matrix.

    Returns:
        torch.Tensor: The calculated scale range.
    """
    off_diagonals = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return torch.mean(torch.log10(matrix[off_diagonals])) - torch.min(torch.log10(matrix[off_diagonals]))
###########################################################################
def default_learning_rate(epoch, total_epochs, loss_list, scale_range=None):
    """
    Default learning rate scheduler.

    Args:
        epoch (int): The current epoch.
        total_epochs (int): The total number of epochs.
        loss_list (list): List of loss values.
        scale_range (float, optional): The range for scaling. Defaults to None.

    Returns:
        float: The calculated learning rate.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")
    if scale_range is None:
        raise ValueError("Scale range must be provided.")

    NO_WEIGHT_EPOCHS = int(NO_WEIGHT_RATIO * total_epochs)
    lr = calculate_multipliers(loss_list[:NO_WEIGHT_EPOCHS] if epoch >= NO_WEIGHT_EPOCHS else loss_list, total_epochs)
    
    if epoch >= NO_WEIGHT_EPOCHS:
        for i in range(NO_WEIGHT_EPOCHS, epoch):
            r = (i - NO_WEIGHT_EPOCHS) / (total_epochs - 1 - NO_WEIGHT_EPOCHS)
            p = 10 ** (-scale_range / (total_epochs - 1 - NO_WEIGHT_EPOCHS))
            lr *= 10 ** (2 * r * np.log10(p)) 
    
    return lr
###########################################################################
def default_scale_learning(epoch, total_epochs):
    """
    Determine if scale learning should occur based on the current epoch and total epochs.

    Args:
        epoch (int): The current epoch.
        total_epochs (int): The total number of epochs.

    Returns:
        bool: True if scale learning should occur, False otherwise.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")

    return epoch < int(CURV_RATIO * total_epochs)
###########################################################################
def default_weight_exponent(epoch, total_epochs):
    """
    Calculate the weight exponent based on the current epoch and total epochs.

    Args:
        epoch (int): The current epoch.
        total_epochs (int): The total number of epochs.

    Returns:
        float: The calculated weight exponent.
    """
    if total_epochs <= 1:
        raise ValueError("Total epochs must be greater than 1.")

    no_weight_epochs = int(NO_WEIGHT_RATIO * total_epochs)
    return 0 if epoch < no_weight_epochs else -(epoch - no_weight_epochs) / (total_epochs - 1 - no_weight_epochs)
###########################################################################
def optimize_embedding(distance_matrix, dimension, **kwargs):
    """
    Optimize the embedding of points based on a given distance matrix.
    
    Parameters:
    - distance_matrix (torch.Tensor): The matrix of distances between points.
    - dimension (int): Dimensionality of the embedding space.
    - **kwargs: Optional parameters for the optimization.
        - initial_tangents (torch.Tensor or None): Initial tangents for optimization.
        - total_epochs (int): Number of optimization iterations (default: 2000).
        - log_function (function or None): Function to log optimization progress.
        - learning_rate (function or None): Function to compute learning rate.
        - scale_learning (function or None): Function to determine scale learning status.
        - weight_exponent (function or None): Function to compute weight exponent.
        - initial_lr (float): Initial learning rate (default: 0.1).
    
    Returns:
    - tangents (torch.Tensor): Optimized tangents.
    - scale (float): Final scale factor used in the optimization.
    """

    if not isinstance(distance_matrix, torch.Tensor):
        raise ValueError("distance_matrix must be a torch.Tensor")
    
    # Default values for kwargs
    total_epochs = kwargs.get('total_epochs', TOTAL_EPOCHS)
    initial_lr = kwargs.get('initial_lr', LEARNING_RATE_INIT)
    initial_tangents = kwargs.get('initial_tangents', None)
    log_function = kwargs.get('log_function', None)
    learning_rate = kwargs.get('learning_rate', None)
    scale_learning = kwargs.get('scale_learning', None)
    weight_exponent = kwargs.get('weight_exponent', None)
    enable_save = kwargs.get('enable_save', False)
    timestamp = kwargs.get('time', None)


    
    
    if initial_tangents is None:
        n = distance_matrix.size(0)
        tangents = torch.rand(dimension, n, requires_grad=True)
        tangents.data.mul_(0.01)
    else:
        tangents = initial_tangents.clone().detach().requires_grad_(True)

    if learning_rate is None:
        learning_rate_fn = lambda epoch, total_epochs, loss_list=None: default_learning_rate(epoch, total_epochs, loss_list, scale_range=(distance_range(distance_matrix)).item())
        # print((distance_range(distance_matrix)).item())
    if scale_learning is None:
        scale_learning_fn = lambda epoch, total_epochs, loss_list=None: default_scale_learning(epoch, total_epochs)
    if weight_exponent is None:
        weight_exponent_fn = lambda epoch, total_epochs, loss_list=None: default_weight_exponent(epoch, total_epochs)
    
    loss_history = []
    scale_factor = torch.tensor(1.0, requires_grad = True)  # Initialize scale
    latest_scale_grad_free = torch.tensor(scale_factor.item())


    optimizer_tangents = optim.Adam([tangents], lr=initial_lr)
    weight_history = []
    lr_history = []
    scale_history = []
    for epoch in range(total_epochs):
        weight_exp = weight_exponent_fn(epoch, total_epochs, loss_history)
        weight_history.append(weight_exp)

        weight_matrix = torch.pow(latest_scale_grad_free * distance_matrix, weight_exp)
        weight_matrix.fill_diagonal_(1)

        is_scale_learning = scale_learning_fn(epoch, total_epochs, loss_history)
        scale_history.append(is_scale_learning)
        if is_scale_learning:
            loss, scale_factor, relative_error = cost_function(tangents, distance_matrix, enable_scale_learning=True, weight_matrix = weight_matrix, enable_save = enable_save)
            latest_scale_grad_free = torch.tensor(scale_factor.item())
        else:
            loss, scale_factor, relative_error = cost_function(tangents, distance_matrix, enable_scale_learning=False, scale_factor=latest_scale_grad_free, weight_matrix=weight_matrix, enable_save = enable_save)

        loss.backward()
        loss_history.append(loss.item())
        with torch.no_grad():
            tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)
        optimizer_tangents.step()

        lr = learning_rate_fn(epoch, total_epochs, loss_history)
        lr_history.append(lr*initial_lr)
        
        

        for param_group in optimizer_tangents.param_groups:
            param_group['lr'] = lr*initial_lr

        current_lr = optimizer_tangents.param_groups[0]['lr']
        
        num_digits = len(str(total_epochs))
        formatted_epoch = f"{epoch + 1:0{num_digits}d}"
        message = (f"[Epoch {formatted_epoch}/{total_epochs}] "
                   f"Loss: {loss.item():.8f}, Scale: {scale_factor.item():.8f}, Learning Rate: {current_lr:.10f}, "
                   f"Weight Exponent: {weight_exp:.8f}, Scale Learning: {'Yes' if is_scale_learning else 'No'}")
        if log_function:
            log_function(message)
        else:
            print(message)

        if enable_save and (relative_error is not None):
            path = f'{RESULTS_DIR}/{timestamp}'
            os.makedirs(path, exist_ok=True)
            # Save relative error to a file
            relative_error_path = os.path.join(path, f"RE_{epoch+1}.npy")
            np.save(relative_error_path, relative_error.detach().cpu().numpy())

    if enable_save:
        path = f'{RESULTS_DIR}/{timestamp}'
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, f"weight_history.npy")
        np.save(weight_path, weight_history)
        lr_path = os.path.join(path, f"lr_history.npy")
        np.save(lr_path, lr_history)
        scale_path = os.path.join(path, f"scale_history.npy")
        np.save(scale_path, scale_history)
        return tangents, scale_factor
    else:
        return tangents, scale_factor

    
###########################################################################
###########################################################################
###########################################################################
class subspace:
    def __init__(self, H = 0, Hp = 0, p = 0,subspace=0):
        self.H = H
        self.Hp = Hp
        self.p = p
        self.subspace = subspace
###########################################################################
###########################################################################
###########################################################################
def J_norm(v):
    """
    Compute the norm of a vector in the Lorentzian space defined by matrix J.

    Parameters:
    v (ndarray): The input vector.

    Returns:
    float: The computed norm.
    """
    # Construct the vector v*J
    vJ = v.copy()
    vJ[0] = -v[0]

    # Compute the norm in the Lorentzian space
    norm_value = np.dot(vJ.ravel(),v.ravel())
    
    return norm_value
###########################################################################
###########################################################################
###########################################################################
def J_norm_torch(v):
    """
    Compute the norm of a vector in the Lorentzian space defined by matrix J.

    Parameters:
    v (Tensor): The input vector.

    Returns:
    Tensor: The computed norm.
    """
    # Construct the vector v*J
    vJ = v.clone()
    vJ[0] = -v[0]

    # Compute the norm in the Lorentzian space
    norm_value = torch.dot(vJ.view(-1), v.view(-1))
    
    return norm_value
###########################################################################
###########################################################################
###########################################################################
###########################################################################
def compute_hyperbolic_exponential(Vt, S):
    """
    Compute the hyperbolic exponential map for given tangent vectors.

    Parameters:
    Vt (ndarray): Tangent vectors (DxN).
    S (object): An object containing the base point 'p'.

    Returns:
    ndarray: The resulting points in the hyperbolic space (DxN).
    """
    
    base_point = np.squeeze(S.p)
    
    # Get dimensions
    D = np.shape(Vt)[0] - 1
    N = np.shape(Vt)[1]
    
    # Initialize the result matrix
    result_matrix = np.zeros((D + 1, N))
    
    for n in range(N):
        v = Vt[:, n]
        norm_v = np.sqrt(J_norm(v))

        if norm_v != 0:
            sinh_norm_v_over_norm_v = np.sinh(norm_v) / norm_v
        else:
            sinh_norm_v_over_norm_v = 1

        x = np.cosh(norm_v) * base_point + sinh_norm_v_over_norm_v * v

        norm_x = J_norm(x)
        
        if norm_x > 0:
            print(x)
            print('Error: Norm should not be positive')
        
        # Normalize the result vector
        result_matrix[:, n] = x / np.sqrt(-norm_x)
    
    return result_matrix
###########################################################################
###########################################################################
###########################################################################
def hyperbolic_log(X, S):
    """
    Compute the hyperbolic logarithm map for given points in hyperbolic space.

    Parameters:
    X (ndarray): Points in hyperbolic space (DxN).
    S (object): An object containing the base point 'p'.

    Returns:
    ndarray: The tangent vectors at the base point 'p' (DxN).
    """
    base_point = S.p
    
    # Get dimensions
    D, N = np.shape(X)
    D -= 1  # Adjust D to match the context of the hyperbolic space
    
    # Initialize the result matrix for tangent vectors
    tangent_vectors = np.zeros((D + 1, N))
    
    # Construct the J matrix for the hyperbolic space
    J = np.eye(D + 1)
    J[0, 0] = -1
    
    for n in range(N):
        x = X[:, n]
        
        # Compute the theta value
        theta = -np.matmul(np.matmul(x.T, J), base_point)
        theta = np.maximum(theta, 1)
        theta = np.arccosh(theta)
        
        # Compute the tangent vector
        if theta != 0:
            theta_over_sinh_theta = theta / np.sinh(theta)
        else:
            theta_over_sinh_theta = 1
        
        tangent_vectors[:, n] = theta_over_sinh_theta * (x - base_point * np.cosh(theta))
    
    return tangent_vectors
###########################################################################
###########################################################################
###########################################################################
def find_first_max_index(array):
    """
    Find the first index of the maximum value in the array.

    Parameters:
    array (ndarray): Input array.

    Returns:
    int: The first index of the maximum value, or -1 if the array is empty.
    """
    if array.size == 0:
        return -1  # Return -1 to indicate that the array is empty.
    
    max_index = np.argmax(array)  # Get the index of the maximum value
    
    return max_index
###########################################################################
###########################################################################
###########################################################################
def estimate_hyperbolic_subspace(X, d = None):
    """
    Estimate the hyperbolic subspace for the given data.

    Parameters:
    X (ndarray): Input data matrix (DxN).
    parameters (object): An object containing parameters 'd', 'N', and 'D'.

    Returns:
    tuple: A tuple containing the subspace object and eigenvalues.
    """
    D,N = np.shape(X)
    D -= 1
    if d is None:
        d = D
    
    # Compute the covariance matrix
    covariance_matrix = np.matmul(X, X.T) / N
    D = np.shape(covariance_matrix)[0] - 1

    # Compute the eigenvalues and eigenvectors in the hyperbolic space
    eigenvalues, eigen_signs, eigenvectors = compute_j_eigenvalues_accurate(covariance_matrix, d)
    
    # Filter and sort the eigenvalues and corresponding eigenvectors
    positive_eigenvalue_indices = eigen_signs > 0
    positive_eigenvalues = eigenvalues[positive_eigenvalue_indices]
    sorted_indices = np.argsort(positive_eigenvalues)[::-1]
    
    # Extract the base point (eigenvectors with negative eigenvalue)
    base_point = np.squeeze(eigenvectors[:, ~positive_eigenvalue_indices])

    # Extract and sort the hyperbolic subspace basis vectors
    hyperbolic_subspace_basis = eigenvectors[:, positive_eigenvalue_indices]
    hyperbolic_subspace_basis = hyperbolic_subspace_basis[:, sorted_indices]

    # Handle multiple potential base points
    if len(np.shape(base_point)) > 1:
        base_point_index = find_first_max_index(eigenvalues[~positive_eigenvalue_indices])
        base_point = base_point[:, base_point_index]
    
    # Ensure the first component of the base point is positive
    if base_point[0] < 0:
        base_point = -base_point
    
    # Reshape base_point and concatenate with the hyperbolic subspace basis
    base_point_reshaped = base_point.reshape(D+1, 1)
    H_matrix = np.concatenate((base_point_reshaped, hyperbolic_subspace_basis), axis=1)
    
    # Create the subspace object and set its attributes
    S = subspace()
    S.H = H_matrix
    S.p = base_point
    S.Hp = hyperbolic_subspace_basis
    S.eigenvalues = eigenvalues
    
    return S
###########################################################################
###########################################################################
###########################################################################
def gram_schmidt(components, n_components):
    def inner(u, v):
        return torch.sum(u * v)
    Q = []
    for k in range(n_components):
        v_k = components[k]
        proj = 0.0
        for v_j in Q:
            v_j = v_j[0]
            coeff = inner(v_j, v_k) / inner(v_j, v_j).clamp_min(1e-15)
            proj += coeff * v_j
        v_k = v_k - proj
        v_k = v_k / torch.norm(v_k).clamp_min(1e-15)
        Q.append(torch.unsqueeze(v_k, 0))
    return torch.cat(Q, dim=0)
###########################################################################
###########################################################################
###########################################################################
def compute_distance_matrix(X, Q_or_S, mu_ref=None, dimension=None, method=None):
    """
    Compute the distance matrix using the specified method.

    Parameters:
    X (ndarray): Input data matrix (D+1, N).
    Q_or_S (ndarray or object): Projection matrix Q or an object containing the transformation matrix 'H'.
    mu_ref (ndarray, optional): Reference point for reflections (required for certain methods).
    dimension (int): The target dimension for the subspace.
    method (str): The method to use ('bsa', 'horopca', 'pga', or other).

    Returns:
    tuple: The hyperbolic distance matrix (HDM) and a boolean indicating if the result is inaccurate.
    """
    D, N = np.shape(X)
    D -= 1  # Adjust D to match the context of the hyperbolic space

    # Construct the J matrix for the hyperbolic space
    J = np.eye(D + 1)
    J[0, 0] = -1

    inaccurate = False

    if method in ['bsa', 'horopca']:
        Q = Q_or_S
        # Transpose X for easier indexing
        X = X.T

        # Extract Y from X and normalize
        Y = X[:, 1:].copy()
        for n in range(N):
            y = Y[n, :]
            Y[n, :] = y / (1 + X[n, 0].copy())

        # Convert Y to tensor and reflect at zero in Poincare ball
        z = torch.from_numpy(Y)  # N by D in Poincare
        x = poincare.reflect_at_zero(z, mu_ref)

        # Gram-Schmidt process to orthonormalize Qd
        Qd = gram_schmidt(Q[:dimension, :], dimension)

        if method == 'bsa':
            proj = poincare.orthogonal_projection(x, Qd, normalized=True)
            Q_orthogonal = euclidean.orthonormal(Qd)
            x_p = proj @ Q_orthogonal.transpose(0, 1)
        else:
            if dimension == 1:
                proj = project_kd(Qd, x)[0]
            else:
                hyperboloid_ideals = hyperboloid.from_poincare(Qd, ideal=True)
                hyperboloid_x = hyperboloid.from_poincare(x)
                hyperboloid_proj = hyperboloid.horo_projection(hyperboloid_ideals, hyperboloid_x)[0]
                proj = hyperboloid.to_poincare(hyperboloid_proj)

            # Orthonormalize Qd in Euclidean space
            Q_orthogonal = euclidean.orthonormal(Qd)

            # Project x to orthogonal Q
            x_p = proj @ Q_orthogonal.transpose(0, 1)

        # Compute pairwise distances in Poincare ball
        distance_matrix = poincare.pairwise_distance(x_p)
        distance_matrix = distance_matrix.numpy()
    else:
        S = Q_or_S
        if method == 'pga':
            # Compute the hyperbolic logarithm map of the data
            tangent_vectors = hyperbolic_log(X, S)

            # Extract the first `dimension` columns of the hyperbolic subspace basis
            H_matrix = S.Hp[:, :dimension]

            # Project the tangent vectors onto the subspace
            projection_matrix = np.matmul(H_matrix, H_matrix.T)
            projected_tangent_vectors = np.matmul(projection_matrix, tangent_vectors)

            # Compute the hyperbolic exponential map of the projected tangent vectors
            projected_data = compute_hyperbolic_exponential(projected_tangent_vectors, S)
            Gd = np.matmul(np.matmul(projected_data.T, J), projected_data)
        else:
            # Extract the first d+1 columns of the transformation matrix H
            H = S.H[:, :dimension + 1]

            # Construct the projection matrix
            Jk = np.eye(dimension + 1)
            Jk[0, 0] = -1

            # Project the data into the new subspace
            projection_matrix = np.matmul(np.matmul(H, Jk), H.T)
            X_projected = np.matmul(np.matmul(projection_matrix, J), X)

            # Normalize the projected data
            for n in range(N):
                x = X_projected[:, n]
                norm_value = -J_norm(x)

                if norm_value < 0:
                    inaccurate = True
                else:
                    X_projected[:, n] = x / np.sqrt(norm_value)

            # Compute the Gram matrix in the new subspace
            Gd = np.matmul(np.matmul(X_projected.T, J), X_projected)
        
        # Compute the distance matrix in the projected space
        Gd[Gd >= -1] = -1  # Ensure the values are within the valid range
        distance_matrix = np.arccosh(-Gd)
        np.fill_diagonal(distance_matrix, 0)

    return distance_matrix, inaccurate
###########################################################################
###########################################################################
###########################################################################
def load_subspace_instance(directory, file_index):
    try:
        filename = f'subspace_{file_index}.pkl'
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            instance = pickle.load(file)
        return instance
    except Exception as e:
        tqdm.write(f"Error occurred while loading: {e}")
        logging.info(f"Error occurred while loading: {e}")
        return None
###########################################################################
###########################################################################
###########################################################################
def load_subspace_instance_qmu(subspace_directory, file_index):
    try:
        Q_path = os.path.join(subspace_directory, f'Q_{file_index}.pt')
        mu_ref_path = os.path.join(subspace_directory, f'mu_{file_index}.pt')
        
        Q = torch.load(Q_path)
        mu_ref = torch.load(mu_ref_path)
        
        return Q, mu_ref
    except Exception as e:
        tqdm.write(f"Error occurred while loading: {e}")
        logging.info(f"Error occurred while loading: {e}")
        return None, None
###########################################################################
###########################################################################
###########################################################################
def compute_hyperbolic_distance_matrix(X):
    """
    Compute the hyperbolic distance matrix for the given data.

    Parameters:
    X (ndarray): Input data matrix (DxN).

    Returns:
    ndarray: The hyperbolic distance matrix (distance_matrix).
    """
    
    D, N = np.shape(X)
    D -= 1  # Adjust D to match the context of the hyperbolic space
    
    # Construct the J matrix for the hyperbolic space
    J = np.eye(D + 1)
    J[0, 0] = -1
    
    # Compute the Gram matrix in the hyperbolic space
    G = np.matmul(np.matmul(X.T, J), X)
    
    # Clamp values to ensure they are suitable for arccosh
    G[G >= -1] = -1
    
    # Compute the distance matrix using arccosh
    distance_matrix = np.arccosh(-G)
    
    # Set the diagonal to zero (distance from a point to itself)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix
###########################################################################
###########################################################################
###########################################################################
def normalize_v(v, threshold):
    """
    Normalize a vector and check if further normalization is needed based on a threshold.

    Parameters:
    v (ndarray): The input vector to be normalized.
    threshold (float): The threshold value to determine if further normalization is necessary.

    Returns:
    tuple: A tuple containing the normalized vector and a boolean indicating if further normalization was performed.
    """
    normalized = False
    
    # Compute the Euclidean norm of the vector
    norm_vector = np.linalg.norm(v)
    
    # Check if the norm exceeds the threshold
    if norm_vector > threshold:
        # Normalize the vector
        v = v / norm_vector
        
        # Compute the hyperbolic norm of the normalized vector
        D = len(v) - 1
        norm_hyperbolic = np.sqrt(np.abs(J_norm(v)))
        
        # Check if the hyperbolic norm exceeds the threshold
        if norm_hyperbolic > threshold:
            # Further normalize the vector
            v = v / norm_hyperbolic
            normalized = True
    
    return v, normalized
###########################################################################
###########################################################################
###########################################################################
def j_orthonormalize(v, subspace_basis, eigenvalue_signs):
    """
    Perform J-orthonormalization of a vector with respect to a subspace.

    Parameters:
    v (ndarray): The vector to orthonormalize.
    subspace_basis (ndarray): Basis vectors of the subspace.
    eigenvalue_signs (list): Signs of eigenvalues used in projection.

    Returns:
    ndarray: The J-orthonormalized vector.
    """
    D = len(v) - 1
    J = np.eye(D + 1)
    J[0, 0] = -1
    
    # Reshape the vector for matrix operations
    vector = np.reshape(v, (D + 1, 1))
    
    # Check if eigenvalue signs are provided
    if len(eigenvalue_signs) > 0:
        # Construct projection matrix to ensure J-orthonormality
        diag_matrix = np.diag(eigenvalue_signs)
        projection_matrix = np.eye(D + 1) - np.matmul(np.matmul(subspace_basis, diag_matrix), np.matmul(subspace_basis.T, J))
        vector = np.matmul(projection_matrix, vector)

    return vector
###########################################################################
###########################################################################
###########################################################################
def expected_sgn(count):
    if count ==1:
        sgn = -1
    else:
        sgn = 1
    return sgn
###########################################################################
###########################################################################
###########################################################################
def compute_residual_matrix(Cx, eigenvalues, eigenvectors):
    """
    Compute the residual matrix by removing the contribution of eigenvalues and eigenvectors.

    Parameters:
    Cx (ndarray): The original matrix (DxD).
    eigenvalues (ndarray): Array of eigenvalues.
    eigenvectors (ndarray): Matrix of eigenvectors (DxD).

    Returns:
    ndarray: The residual matrix.
    """
    residual_matrix = Cx.copy()  # Initialize the residual matrix as a copy of the original matrix
    
    # Subtract the contribution of each eigenvalue and corresponding eigenvector
    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        eigenvector_i = eigenvectors[:, i]
        residual_matrix -= lambda_i * np.outer(eigenvector_i, eigenvector_i)
    
    # Normalize the residual matrix by its maximum absolute value
    residual_matrix /= np.max(np.abs(residual_matrix))
    
    return residual_matrix
###########################################################################
###########################################################################
########################################################################### 
def find_good_initial_j_eigenvector(Cx, true_sign, eigenvalues, eigenvectors, eigen_signs, threshold):
    """
    Find a good initial J-eigenvector for the given residual matrix.

    Parameters:
    Cx (ndarray): covariance matrix (DxD).
    true_sign (float): The true sign of the eigenvector.
    eigenvectors (ndarray): Matrix of previously computed eigenvectors (DxD).
    eigen_signs (ndarray): Array of eigenvalue signs.
    threshold (float): Threshold for normalization.

    Returns:
    ndarray: The initial J-eigenvector.
    """

    Cx_residual = compute_residual_matrix(Cx, eigenvalues, eigenvectors)

    D = np.shape(Cx_residual)[0] - 1

    # Construct J matrix
    J = np.eye(D + 1)
    J[0, 0] = -1

    # Compute the leading eigenvector of the J-transformed residual matrix
    _, v = eigs(np.matmul(Cx_residual, J), k=1, which='LM')
    v = np.real(v)

    # Orthonormalize and normalize the leading eigenvector
    v = j_orthonormalize(v, eigenvectors, eigen_signs)
    v, status = normalize_v(v, threshold)

    # Check the sign of the normalized eigenvector
    sign = J_norm(v)

    # Add noise and re-normalize if the vector is not suitable
    while (not status) or (sign * true_sign < 0):
        noise = threshold * np.random.randn(D + 1, 1)
        v = j_orthonormalize(v + noise, eigenvectors, eigen_signs)
        v, status = normalize_v(v, threshold)
        sign = J_norm(v)

    return v
###########################################################################
###########################################################################
###########################################################################
def find_random_j_eigenvector(D, true_sign, eigenvectors, eigen_signs, threshold):
    """
    Find a valid random J-eigenvector

    Parameters:
    D (int): Dimension of the vectors.
    true_sign (float): The true sign of the eigenvector.
    eigenvectors (ndarray): Matrix of previously computed eigenvectors (DxD).
    eigen_signs (ndarray): Array of eigenvalue signs.
    threshold (float): Threshold for normalization.

    Returns:
    ndarray: A suitable random J-eigenvector.
    """

    status = False
    sign = -1
    while not status or sign * true_sign < 0:
        # Generate a random vector
        random_vector = np.random.randn(D + 1, 1)
        
        # Orthonormalize the random vector
        random_vector = j_orthonormalize(random_vector, eigenvectors, eigen_signs)
        
        # Normalize the orthonormalized vector
        random_vector, status = normalize_v(random_vector, threshold)
        
        # Calculate the sign of the normalized vector
        sign = J_norm(random_vector)

    return random_vector
###########################################################################
###########################################################################
########################################################################### 
def perturb_to_find_valid_j_eigenvector(v, true_sign, D, eigenvectors, eigenvalue_signs, threshold):
    """
    Perturbs the initial vector to find a valid J eigenvector.
    
    Parameters:
        v (np.ndarray): The initial vector to start perturbing from.
        true_sign (float): The true sign for the J norm.
        D (int): The dimension of the problem.
        eigenvectors (np.ndarray): The current set of eigenvectors.
        eigenvalue_signs (np.ndarray): The signs of the current eigenvalues.
        threshold (float): The perturbation threshold.

    Returns:
        np.ndarray: A valid J eigenvector.
    """
    is_valid = False
    current_sign = J_norm(v)
    
    while not is_valid or (current_sign * true_sign < 0):
        # Add small random noise to the initial vector to perturb it
        noise = threshold * np.random.randn(D + 1, 1)
        perturbed_vector = v + noise

        # Orthonormalize the perturbed vector against the current eigenvectors
        perturbed_vector = j_orthonormalize(perturbed_vector, eigenvectors, eigenvalue_signs)
        
        # Normalize the perturbed vector
        perturbed_vector, is_valid = normalize_v(perturbed_vector, threshold)
        
        # Compute the sign of the J norm of the perturbed vector
        current_sign = J_norm(perturbed_vector)
    
    return perturbed_vector
###########################################################################
###########################################################################
###########################################################################
def update_j_eigenvectors(eigenvectors, new_eigenvector):
    """
    Updates the set of eigenvectors with a new eigenvector.

    Parameters:
        eigenvectors (np.ndarray): Array of existing eigenvectors.
        new_eigenvector (np.ndarray): New eigenvector to be added.

    Returns:
        np.ndarray: Updated array of eigenvectors.
    """
    if np.shape(eigenvectors)[0] == 0:
        # If no eigenvectors exist yet, initialize with the new eigenvector
        updated_eigenvectors = new_eigenvector
    else:
        # Concatenate the new eigenvector to the existing set of eigenvectors
        updated_eigenvectors = np.concatenate((eigenvectors, new_eigenvector), axis=1)
    
    return updated_eigenvectors
###########################################################################
###########################################################################
###########################################################################
def perform_multiplications(Cxj, v, threshold, accuracy_factor, eigenvectors, eigenvalue_signs):
    """
    Performs one iteration of matrix-vector multiplications with normalization and orthonormalization steps.
    
    Parameters:
        Cxj (np.ndarray): The matrix used for multiplication.
        v (np.ndarray): The input vector to be multiplied.
        threshold (float): The threshold for normalization.
        accuracy_factor (int): Number of times orthonormalization is applied.
        eigenvectors (np.ndarray): The set of eigenvectors for orthonormalization.
        eigenvalue_signs (np.ndarray): Signs of the eigenvalues for orthonormalization.

    Returns:
        np.ndarray: The resulting vector after the iteration.
    """

    # Step 1: Matrix-vector multiplication
    v_out = np.matmul(Cxj, v)
    
    # Step 2: Normalize the resulting vector
    v_out, _ = normalize_v(v_out, threshold)
    
    # Step 3: Apply orthonormalization and normalization iteratively
    for _ in range(accuracy_factor):
        v_new = j_orthonormalize(v_out, eigenvectors, eigenvalue_signs)
        v_new, _ = normalize_v(v_out, threshold)

    return v_out
###########################################################################
###########################################################################
###########################################################################
def compute_one_j_eigenvalue(Cx, J, j_eigenvector):
    """
    Computes the j-th eigenvalue for given matrices and eigenvector.
    
    Parameters:
        Cx (np.ndarray): The covariance matrix.
        J (np.ndarray): The matrix J.
        j_eigenvector (np.ndarray): The j-eigenvector.

    Returns:
        float: The computed j_eigenvalue.
    """
    
    # Step 1: Perform the first matrix multiplication: (C * J) * j_eigenvector
    intermediate_vector = np.matmul(np.matmul(Cx, J), j_eigenvector)
    
    # Step 2: Perform the second matrix multiplication: (C * J) * intermediate_vector
    result_vector = np.matmul(np.matmul(Cx, J), intermediate_vector)
    
    # Step 3: Calculate the eigenvalue as the square root of the ratio of norms
    j_eigenvalue = np.sqrt(np.linalg.norm(result_vector) / np.linalg.norm(j_eigenvector))
    
    return j_eigenvalue
###########################################################################
###########################################################################
###########################################################################
def compute_j_eigenvalues(Cx,d):
    D = np.shape(Cx)[0]-1
    J = np.eye(D+1)
    J[0,0] = -1
    evals = []
    eval_signs = []
    condition = True
    count = 0
    evecs = []
    while condition:
        count = count + 1 

        _, v = eigs(np.matmul(Cx,J), k=1, which = 'LM')
        v = v/np.sqrt(np.abs(J_norm(v))) 
        v = np.real(v)
        sgn = np.sign(J_norm(v))
        if count == 1:
            evecs = v
        else:
            evecs = np.concatenate( (evecs,v) , axis = 1)
        lmbd = np.matmul(np.matmul(v.T,J), np.matmul(np.matmul(Cx,J),v))

        lmbd = np.squeeze(lmbd)
        evals = np.append(evals,lmbd)
        eval_signs = np.append(eval_signs,sgn)
        Cx = Cx - lmbd*np.matmul(v,v.T)
        condition = are_eigenvalues_valid(eval_signs,d)
    return evals, eval_signs, evecs
###########################################################################
###########################################################################
###########################################################################
def compute_j_eigenvalues_accurate(Cx,d):
    condition = True
    count = 0
    threshold = 10**(-20)
    ev_threshold = 10**(-30)

    evals = []
    eval_signs = []
    evecs = []
    line = []

    D = np.shape(Cx)[0]-1
    J = np.eye(D+1)
    J[0,0] = -1

    Cxj = np.matmul(Cx,J)
    accuracy_factor = 1
    while condition:
        count = count + 1 
        true_sgn = expected_sgn(count)
        ###################################################################################  
        v = find_good_initial_j_eigenvector(Cx,true_sgn,evals,evecs,eval_signs,threshold)
        flag = True
        errors = [1]
        while flag:
            norm_v = np.linalg.norm(v)
            v2 = perform_multiplications(Cxj, v, threshold, accuracy_factor, evecs, eval_signs)
            norm_v2 = np.linalg.norm(v2)
            sgn = J_norm(v2)
            #####################################################
            if norm_v2 < ev_threshold:
                line = 1
                ev = 1
                errors = [1]
                flag = False
                v2 = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
            else:
                ev = np.linalg.norm(v/norm_v - v2/norm_v2)
                ev = min(ev ,np.linalg.norm(v/norm_v + v2/norm_v2))
            #####################################################
            if (ev < ev_threshold) and (sgn*true_sgn >=0):
                flag = False
                errors = [1]
                line = 2
            else:
                ev_inf = min(errors[-1], ev)
                errors.append(ev_inf)
                K = 1000
                if len(errors) > K+1 and np.mod(len(errors),K) == 0: 
                    last_K = errors[-K:]
                    mean_err = np.mean(last_K)
                    std_err = np.std(last_K)

                    sgn = J_norm(v2)
                    good_enough = (mean_err == 0) or (std_err < ev_threshold/count) 
                    if  good_enough and (sgn*true_sgn >=0):
                        flag = False
                        errors = [1]
                        line = 3
                    elif len(evals) >= 1:
                        if sgn*true_sgn < 0:
                            accuracy_factor = accuracy_factor + 1
                            if accuracy_factor > 100:
                                flag = False
                                line = 4
                                errors = [1]
                                v2 = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
                        else:
                            min_val = np.min(evals)
                            no_of_remaining_vals = D+1-len(eval_signs)
                            remaining_energy = min_val/2 * no_of_remaining_vals
                            total_energy = np.sum(evals)+min_val/2 * no_of_remaining_vals

                            accurate_enough = remaining_energy/total_energy < 10**(-4)
                            iterate_upperb = len(errors) > (D+1)*K*np.log(1+len(evals))
                            if accurate_enough or (len(evals) >= 2 and iterate_upperb):
                                flag = False
                                #status = False
                                line = 5
                                errors = [1]
                                v2 = perturb_to_find_valid_j_eigenvector(v2, true_sgn, D, evecs, eval_signs, threshold)

            v = j_orthonormalize(v2,evecs,eval_signs)
            v,_ = normalize_v(v,threshold)
            norm_v = np.linalg.norm(v)
            if norm_v < ev_threshold:
                status = False
                errors = [1]
                flag = False
                line = 6
                v = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
                
        v = j_orthonormalize(v,evecs,eval_signs)
        v,status = normalize_v(v,threshold)
        sgn = np.sign(J_norm(v))
        if (not status) or (sgn*true_sgn < 0):
            status = False
            line = 7
            v = find_random_j_eigenvector(D, true_sgn, evecs, eval_signs, threshold)
            sgn = J_norm(v)
                    
        lmbd = compute_one_j_eigenvalue(Cx, J, v)
        sgn = np.sign(J_norm(v))
        
        evecs = update_j_eigenvectors(evecs,v)
        evals = np.append(evals,lmbd)
        eval_signs = np.append(eval_signs,sgn)
        condition = are_eigenvalues_valid(eval_signs,d)
        #print(count,evals)
    return evals, eval_signs, evecs
###########################################################################
###########################################################################
###########################################################################
def are_eigenvalues_valid(eigenvalue_signs, d):
    """
    Checks if the set of eigenvalues meets the specified conditions.
    
    Parameters:
        eigenvalue_signs (np.ndarray): Array of signs of the current eigenvalues.
        target_positive_count (int): Desired number of positive eigenvalue signs.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    # Count the number of positive and negative eigenvalue signs
    positive_sign_count = np.sum(eigenvalue_signs > 0)
    negative_sign_count = np.sum(eigenvalue_signs < 0)
    
    # Determine if the condition is met
    condition = not (positive_sign_count == d and negative_sign_count >= 1)
    
    return condition
###########################################################################
###########################################################################
###########################################################################
def project_vector_to_hyperbolic_space(vector):
    """
    Project a given vector to hyperbolic space using iterative optimization.

    Parameters:
    vector (np.ndarray): The input vector to be projected.

    Returns:
    np.ndarray: The projected vector in hyperbolic space.
    """
    # Determine the dimension from the input vector
    #######################################
    dimension = len(vector)-1

    # Define the Lorentzian metric tensor
    lorentzian_metric = np.ones(dimension+1)
    lorentzian_metric[0] = -1
    
    # Check if the vector is above the Lorentzian sheet
    above_sheet = vector[0] > 0
    
    # Set the tolerance level for convergence
    tolerance = 10**(-16)
    center = 0
    
    # Initialize the optimal error and projection values
    optimal_error = 10**(10)
    optimal_projector = np.eye(dimension+1)
    projected_point = vector
    for i in range(70):
        lambda_range = 10**(-i)

        if above_sheet:
            lambda_min = max(center-lambda_range, -1+tolerance)
            lambda_max = min(center+lambda_range, 1-tolerance)
            number = 100
        else:
            lambda_min = max(center-lambda_range*10000, 1+tolerance)
            lambda_max = center+lambda_range*10000
            number = 100
        lambda_values = np.linspace(lambda_min,lambda_max,num=number)
        
        for lambda_val in lambda_values:
            projection_operator_lambda = 1./(1+lambda_val*lorentzian_metric) 
            projected_point_lambda = projection_operator_lambda  * vector # elementwise product
            if abs(J_norm(projected_point_lambda)+1) < optimal_error:
                optimal_projector = projection_operator_lambda
                optimal_error = abs(J_norm(projected_point_lambda)+1)
                center = lambda_val
        
        new_projected_point = optimal_projector * vector
        if abs(J_norm(new_projected_point)+1) < tolerance:
            return new_projected_point
        projected_point = new_projected_point

    if optimal_error > 10**(-5):
        print('Warning: High projection error:', optimal_error)
    return projected_point
###########################################################################
###########################################################################
########################################################################### 
def lgram_to_points(gram_matrix, dimension):
    """
    Convert a Lorentzian Gram matrix to points using eigen decomposition.

    Parameters:
    dimension (int): Dimension of the target space.
    gram_matrix (np.ndarray): Gram matrix to be decomposed.

    Returns:
    np.ndarray: Coordinates corresponding to the input Gram matrix.
    """
    min_dimension = min(dimension, np.shape(gram_matrix)[0]-1)
    # Perform eigen decomposition of the Gram matrix
    eigenvalues, eigenvectors = np.linalg.eig(gram_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Identify the smallest eigenvalue and its index
    min_eigenvalue = np.amin(eigenvalues)
    min_index = np.argmin(eigenvalues)
    min_eigenvector = eigenvectors[:, min_index]
    
    
    
    # Remove the smallest eigenvalue and sort the rest in descending order
    eigenvalues = np.delete(eigenvalues, min_index)
    eigenvectors = np.delete(eigenvectors, min_index, axis=1)
    
    sorted_indices = np.argsort(-eigenvalues)

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]
    
    top_eigenvalues = eigenvalues[:dimension]
    top_eigenvectors = eigenvectors[:,:dimension]
    
    
    # Create the final set of eigenvalues for constructing the coordinate matrix
    final_eigenvalues = np.concatenate((abs(min_eigenvalue), top_eigenvalues), axis=None)
    final_eigenvalues[final_eigenvalues <= 0] = 0
    final_eigenvalues = np.sqrt(final_eigenvalues)

    # Adjust eigenvectors to match the selected eigenvalues
    final_eigenvectors = np.column_stack((min_eigenvector, top_eigenvectors))
    
    # Construct the coordinate matrix
    X = np.matmul(np.diag(final_eigenvalues), final_eigenvectors.T)
    
    # Ensure the first coordinate is positive
    if X[0, 0] < 0:
        X = -X

    if min_dimension < dimension:
        zero_rows = np.zeros((dimension-min_dimension, np.shape(gram_matrix)[0]))
        X = np.vstack((X, zero_rows))
    
    return X
###########################################################################
###########################################################################
###########################################################################
def estimate_hyperbolic_subspace_pga(X, d = None):
    """
    Estimate the hyperbolic subspace using Principal Geodesic Analysis (PGA).

    Parameters:
    X (ndarray): The input data matrix (DxN).
    parameters (object): An object containing parameters such as dimension (d), dimensionality (D), and number of samples (N).

    Returns:
    tuple: A tuple containing the transformed data matrix and the subspace object (X_, S).
    """
    S = subspace()  # Initialize the subspace object
    D, N = np.shape(X)
    D -= 1
    if d is None:
        d = D
    
    # Unpack parameters
    tau = 0.1
    tolerance = 10**(-10)
    
    # Compute initial mean vector p
    p = np.mean(X, axis=1)
    
    # Normalize p
    p /= np.sqrt(-J_norm(p))
    S.p = p
    
    # Initialize error and condition for convergence
    error = 1
    condition = True
    cnt = 0
    # Iteratively update p until convergence
    while condition:
        # Compute hyperbolic logarithm of the data with respect to current subspace
        cnt = cnt +1 
        tangent_vectors = hyperbolic_log(X, S)
        
        # Compute update delta_p
        delta_p = tau * np.mean(tangent_vectors, axis=1)
        delta_p = delta_p.reshape(D + 1, 1)
        
        # Perform hyperbolic exponential map to update p
        p = compute_hyperbolic_exponential(delta_p, S).ravel()
        
        # Calculate error between current p and previous p
        new_error = np.linalg.norm(p - S.p) / np.sqrt(D + 1)
        
        # Check convergence condition
        condition = np.abs(error - new_error) > tolerance
        
        # Update error and subspace.p
        error = new_error
        S.p = p
    
    # Finalize subspace estimation
    tangent_vectors = hyperbolic_log(X, S)
    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(tangent_vectors, tangent_vectors.T))
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort eigenvalues and eigenvectors
    index = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    
    # Extract top d eigenvectors as Hp
    Hp = eigenvectors[:, 0:d]
    S.Hp = Hp
    
    # Construct H matrix
    p = S.p
    H = np.concatenate((p.reshape(D + 1, 1), Hp), axis=1)
    S.H = H
    
    # Project data onto the estimated subspace and compute exponential map
    #Vt = np.matmul(np.matmul(Hp, Hp.T), V)
    #transformed_data = compute_hyperbolic_exponential(Vt, S)
    
    return S
###########################################################################
###########################################################################
###########################################################################
def run_dimensionality_reduction(model_type, X):
    lr=5e-2
    n_runs=5

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    torch.set_default_dtype(torch.float64)

    pca_models = {
        'pca': {'class': EucPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'tpca': {'class': TangentPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'pga': {'class': PGA, 'optim': True, 'iterative': True, "n_runs": n_runs},
        'bsa': {'class': BSA, 'optim': True, 'iterative': False, "n_runs": n_runs},
        'horopca': {'class': HoroPCA, 'optim': True, 'iterative': False, "n_runs": n_runs},
    }

    
    D, N = np.shape(X)
    D -= 1

    X = X.T
    Y = X[:, 1:].copy()
    for n in range(N):
        y = Y[n, :]
        Y[n, :] = y / (1 + X[n, 0].copy())

    z = torch.from_numpy(Y)  # N by D in poincare
    # Compute the mean and center the data
    #logging.info("Computing the Frechet mean to center the embeddings")
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(z, return_converged=True)
    #logging.info(f"Mean computation has converged: {has_converged}")
    x = poincare.reflect_at_zero(z, mu_ref)

    # Run dimensionality reduction methods
    #logging.info(f"Running {model_type} for dimensionality reduction")
    metrics = []
    dist_orig = poincare.pairwise_distance(x)
    k = 0
    if model_type in pca_models.keys():
        model_params = pca_models[model_type]
        model_params["n_runs"] = 1
        for _ in range(model_params["n_runs"]):
            model = model_params['class'](dim=D, n_components=D-1, lr=lr, max_steps=500)
            if torch.cuda.is_available():
                model.cuda()
            model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
            while np.isnan(model.compute_metrics(x)['distortion']):
                k += 1
                model = model_params['class'](dim=D, n_components=D-k, lr=lr, max_steps=500)
                if torch.cuda.is_available():
                    model.cuda()
                model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
                if k == D-1:
                    break
            if not np.isnan(model.compute_metrics(x)['distortion']):
                embeddings = model.map_to_ball(x).detach().cpu().numpy()
                Q = model.get_components()
            else:
                Q = np.nan
        return Q.detach().cpu(), mu_ref.detach().cpu()
    else:
        logging.info(f"Model {model_type} is not implemented.")
        return np.nan, np.nan
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

######################################################################
######################################################################
######################################################################
def distance_distortion(DM1, DM2):
    distortion = np.linalg.norm(DM1 - DM2)/ np.linalg.norm(DM2) * 100
    return distortion
######################################################################
######################################################################
######################################################################
def total_wasserstein_distance(X, Y):
    X = X.T
    Y = Y.T
    X = X**2
    Y = Y**2
    # Initialize total distance
    total_distance = 0.0
    for x, y in zip(X, Y):
        distance = wasserstein_distance(x, y)
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def total_variation_distance(X, Y):
    X = X.T
    Y = Y.T
    X = X**2
    Y = Y**2
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of distributions
    for x, y in zip(X, Y):
        # Compute Wasserstein distance between the distributions
        distance = total_variation(x, y)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def total_distance(X, Y):
    X = X.T
    Y = Y.T
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of distributions
    for x, y in zip(X, Y):
        # Compute Wasserstein distance between the distributions
        g = np.matmul(x.T, y)
        g = max(g,-1)
        g = min(g,1)
        distance = np.arccos(g)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def aitchison_distance(x, y):
    D = len(x)
    x = np.array(x)+min(10**(-6), 0.01/D)
    y = np.array(y)+min(10**(-6), 0.01/D)
    # Pseudo-inverse of the compositions
    inv_x = np.log(x / stats.gmean(x))
    inv_y = np.log(y / stats.gmean(y))
    # Compute the Aitchison distance
    dist = np.sqrt(np.sum((inv_x - inv_y) ** 2))
    return dist
######################################################################
######################################################################
######################################################################
def kl_divergence(p, q):
    D = len(p)
    p = np.array(p, dtype=np.float64) + min(10**(-6), 0.01/D)
    q = np.array(q, dtype=np.float64) + min(10**(-6), 0.01/D)
    
    # Compute the KL divergence
    return np.sum(p * np.log(p / q))
######################################################################
######################################################################
######################################################################
def total_variation(p, q):
    """Compute the Total Variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))
######################################################################
######################################################################
######################################################################
def total_aitchison_distance(X, Y):
    X = X.T
    Y = Y.T
    X = X**2
    Y = Y**2
    # Initialize total distance
    total_distance = 0.0
    # Iterate over each pair of compositions
    for x, y in zip(X, Y):
        # Compute Aitchison distance between the compositions
        distance = aitchison_distance(x, y)
        # Add the distance to the total
        total_distance += distance
    return total_distance
######################################################################
######################################################################
######################################################################
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
######################################################################
######################################################################
######################################################################
def compute_jsdm(X):
    X = X.T
    X = X ** 2
    N = np.shape(X)[0]
    JSDM = np.zeros((N, N))
    for i in range(N):
        x = X[i, :]
        for j in range(i + 1, N):
            y = X[j, :]
            JSDM[i, j] = jensen_shannon_divergence(x, y)
            JSDM[j, i] = JSDM[i, j]
    return JSDM
######################################################################
######################################################################
######################################################################
def compute_tvdm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    DM = np.zeros((N,N))
    for i in range(N):
        x = X[i,:]
        for j in range(i+1,N):
            y = X[j,:]
            DM[i,j] = total_variation(x, y)
            DM[j,i] = DM[i,j]
    return DM
######################################################################
######################################################################
######################################################################
def compute_kldm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    DM = np.zeros((N,N))
    for i in range(N):
        x = X[i,:]
        for j in range(i+1,N):
            y = X[j,:]
            DM[i,j] = kl_divergence(x, y)
            DM[j,i] = DM[i,j]
    return DM
######################################################################
######################################################################
######################################################################
def compute_aidm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    D = np.shape(X)[1]

    Y = np.zeros((N,D))
    gmean = np.zeros((N,1))
    for n in range(N):
        x = X[n,:]+min(10**(-6), 0.01/D)
        Y[n,:] = np.log(x/ stats.gmean(x))
    G = np.matmul(Y,Y.T)
    dG = np.reshape(np.diag(G), (N,1))
    dG = np.matmul(dG,np.ones((1,N)) )
    DM= np.sqrt(-2*G + dG + dG.T)
    return DM
######################################################################
######################################################################
######################################################################
def compute_mutual_information(X, y):
    X = X**2
    # Compute mutual information
    mutual_infos = []
    for label in np.unique(y):
        y_binary = (y == label).astype(int)
        mutual_info = mutual_info_classif(X, y_binary)
        mutual_infos.append(mutual_info)
    mean_mutual_information = np.mean(mutual_infos)
    return mean_mutual_information
######################################################################
######################################################################
######################################################################
def compute_wdm(X):
    X = X.T
    X = X**2
    N = np.shape(X)[0]
    DM = np.zeros((N,N))
    for i in range(N):
        x = X[i,:]
        for j in range(i+1,N):
            y = X[j,:]
            DM[i,j] = wasserstein_distance(x, y)
            DM[j,i] = DM[i,j]
    return DM
######################################################################
######################################################################
######################################################################
# Define the compute_sdm function
def compute_sdm(X):
    G = np.matmul(X.T, X)
    np.clip(G, -1, 1, out=G)
    DM = np.arccos(G)
    return DM
######################################################################
######################################################################
######################################################################
def estimate_spherical_subspace(X,param):
    S = subspace()
    d = param.d
    N = param.N
    #####################################
    Cx = np.matmul(X,X.T) #Cx = (Cx + Cx.T)/2
    evals , evecs = np.linalg.eig(Cx)
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    H = evecs[:,0:d+1]
    S.H = H
    S.p = evecs[:,0]
    p = S.p
    
    S.Hp = evecs[:,1:d+1]
    #####################################
    PH = np.matmul(H,H.T)
    X_ = np.matmul(PH,X)

    for n in range(N):
        x_ = X_[:,n]
        X_[:,n] =  x_/np.linalg.norm(x_)
    #####################################
    return X_,S
######################################################################
######################################################################
######################################################################
def estimate_spherical_subspace_liu(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    normX = np.linalg.norm(X,'fro') 
    #####################################
    ############# mode 1 ################
    # I = np.eye(D+1)
    # H = np.random.randn(D+1,d+1)#I[:,0:d+1]
    # H = I[:,0:d+1]
    # V = np.random.randn(d+1,N)
    ############# mode 1 ################

    ############# mode 2 ################
    X_, S_ = estimate_spherical_subspace(X,param)
    H = S_.H
    V = np.matmul(H.T, X)
    ############# mode 2 ################

    err = 1
    #####################################
    lambd = 10*N
    mu = 10*N

    condition = True
    err_diff = 0
    count = 0
    while condition:
        count = count + 1
        X1 = np.matmul(H,V) 
        err1 = np.linalg.norm(X-X1,'fro')/normX

        V_ = (lambd-2)*V+2*np.matmul(H.T,X)
        for n in range(N):
            V_[:,n] = V_[:,n]/np.linalg.norm(V_[:,n])
        M = 2*np.matmul( (X-np.matmul(H,V_)), V_.T)+mu*H
        le,_,re = np.linalg.svd(M,full_matrices=False)
        H_ = np.matmul(le,re)

        X2 = np.matmul(H_,V_)
        err2 = np.linalg.norm(X-X2,'fro')/normX
        err_diff = err2-err1
        err_diff_normal = err_diff*lambd
        #print(err_diff_normal)
        # print(lambd)
        # if np.abs(err_diff_normal) < 1:
        #     if err_diff_normal < 0:
        #         lambd = max( lambd + np.sqrt(N/10)*err_diff_normal , N/np.log(N))
        #         mu = max( mu + np.sqrt(N/10)*err_diff_normal , N/np.log(N))

        
        
        condition = (np.abs(err_diff_normal)  > 10**(-3)) or (err_diff > 0)

        V = V_
        H = H_
        #print(count)
        #print(lambd)
    S.H = H 
    S.p = H[:,0]
    S.Hp = H[:,1:d+1]
    return X2,S
######################################################################
######################################################################
######################################################################





def estimate_spherical_subspace_pga(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    tau = 0.1
    ########## new line ###################
    p = np.mean(X,1)
    ########## new line ###################
    S.p  = p/np.linalg.norm(p)
    err = 1
    condition = True
    while condition:
        V = spherical_log(X,S)
        delta_p = tau * np.mean(V,1)
        delta_p = delta_p.reshape(D+1,1)
        p = np.concatenate( spherical_exp(delta_p,S) )
        err_ = np.linalg.norm(p-S.p)/np.sqrt(D+1)
        ########## new line ###################
        condition = np.abs(err-err_)  > 10**(-10)
        ########## new line ###################
        err = err_
        S.p = p
    #####################################    
    V = spherical_log(X,S)
    evals , evecs = np.linalg.eig( np.matmul(V,V.T))
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    Hp = evecs[:,0:d]
    S.Hp = Hp
    p = S.p
    H = np.concatenate( (p.reshape(D+1,1),Hp),axis = 1)
    S.H = H
    Vt = np.matmul( np.matmul(Hp,Hp.T) , V)
    X_ = spherical_exp(Vt,S)
    return X_,S
########################################################################### 
def estimate_hyperbolic_subspace_pga(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    tau = 0.1
    ########## new line ###################
    p = np.mean(X,1)
    ########## new line ###################
    S.p  = p/np.sqrt(-J_norm(p,D))
    err = 1
    condition = True
    while condition:
        V = hyperbolic_log(X,S)
        delta_p = tau * np.mean(V,1)
        delta_p = delta_p.reshape(D+1,1)
        p = np.concatenate( hyperbolic_exp(delta_p,S) )
        err_ = np.linalg.norm(p-S.p)/np.sqrt(D+1)
        ########## new line ###################
        condition = np.abs(err-err_)  > 10**(-3)
        ########## new line ###################
        err = err_
        S.p = p
    #####################################    
    V = hyperbolic_log(X,S)
    evals , evecs = np.linalg.eig( np.matmul(V,V.T))
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    Hp = evecs[:,0:d]
    S.Hp = Hp
    p = S.p
    H = np.concatenate( (p.reshape(D+1,1),Hp),axis = 1)
    S.H = H
    Vt = np.matmul( np.matmul(Hp,Hp.T) , V)
    X_ = hyperbolic_exp(Vt,S)
    return X_,S
########################################################################### 
def estimate_spherical_subspace_pga_2(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    tau = 0.1

    p = np.mean(X,1)
    S.p  = p/np.linalg.norm(p)

    p = S.p

    #p = X[:,0]+0.1
    #S.p  = p/np.linalg.norm(p)
    err = 1
    condition = True
    while condition:
        V = spherical_log(X,S)
        delta_p = tau * np.mean(V,1)
        delta_p = delta_p.reshape(D+1,1)
        p = np.concatenate( spherical_exp(delta_p,S) )
        err_ = np.linalg.norm(p-S.p)/np.sqrt(D+1)
        condition = np.abs(err-err_)  > 10**(-3)
        err = err_
        S.p = p
    #####################################    
    V = spherical_log(X,S)
    evals , evecs = np.linalg.eig( np.matmul(V,V.T))
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    Hp = evecs[:,0:d]
    S.Hp = Hp
    p = S.p
    H = np.concatenate( (p.reshape(D+1,1),Hp),axis = 1)
    S.H = H
    Vt = np.matmul( np.matmul(Hp,Hp.T) , V)
    X_ = spherical_exp(Vt,S)
    return X_,S
########################################################################### 
def estimate_spherical_subspace_dai(X,param):
    S = subspace()
    d = param.d
    D = param.D
    N = param.N
    #####################################    
    e1 = np.zeros((D+1,))
    e1[0] = 1
    #####################################    
    tau = 0.1
    p = X[:,0]+0.1
    S.p  = p/np.linalg.norm(p)
    err = 1
    eps = 10**(-8)
    condition = True
    #####################################    
    while condition:
        V = spherical_log(X,S)
        delta_p = tau * np.mean(V,1)
        delta_p = delta_p.reshape(D+1,1)
        p = np.concatenate( spherical_exp(delta_p,S) )
        err_ = np.linalg.norm(p-S.p)/np.sqrt(D+1)
        condition = np.abs(err-err_)  > 10**(-3)
        err = err_
        S.p = p
    #####################################    
    p = S.p
    S.p = e1
    cost = 10**(10)
    condition = True
    cnt = 0
    step = 1
    while condition:
        cnt = cnt + 1
        tmp = np.matmul(X.T,p)
        tmp = np.minimum(tmp,1)
        tmp = np.maximum(tmp,-1)
        cost_ = np.mean( np.arccos( tmp )**2 )
        ########## new line ###################
        step  = 1/cnt 
        ########## new line ###################
        cost = cost_
        p_v = spherical_log( p.reshape(D+1,1),S )
        g = np.zeros((D+1,1))
        #compute gradient
        for i in range(D):
            x = p_v
            x[i+1,0] = x[i+1,0] + eps
            p_ = spherical_exp(x,S) 
            tmp = np.matmul( X.T,p_ )
            tmp = np.minimum(tmp,1)
            tmp = np.maximum(tmp,-1)
            cost_i = np.mean( np.arccos( tmp )**2 )
            g[i+1,0] = (cost_i - cost)/eps
        norm_g = np.linalg.norm(g)
        condition = norm_g/D > 10**(-2)
        #print( norm_g/D)
        p = spherical_exp(p_v-step*g/norm_g,S) 
    S.p = np.concatenate(p)
    V = spherical_log(X,S)
    evals , evecs = np.linalg.eig( np.matmul(V,V.T))
    evals = np.real(evals)
    evecs = np.real(evecs)
    index = np.argsort(-evals)
    evals = evals[index]
    evecs = evecs[:,index]
    #####################################
    Hp = evecs[:,0:d]
    S.Hp = Hp
    p = S.p
    H = np.concatenate( (p.reshape(D+1,1),Hp),axis = 1)
    S.H = H
    Vt = np.matmul( np.matmul(Hp,Hp.T) , V)
    X_ = spherical_exp(Vt,S)
    return X_,S
###########################################################################
###########################################################################
###########################################################################
def random_orthogonal_matrix(param):
    D = param.D
    d = param.d
    
    # Initialize the matrix H
    H = np.zeros((D + 1, d + 1))
    
    # Iterate over each column of H
    for i in range(d + 1):
        # Compute the orthogonal complement
        PH_perp = np.eye(D + 1) - np.matmul(H, H.T)
        
        # Generate a random vector
        v = np.random.normal(0, 1, (D + 1, 1))
        
        # Project the vector onto the orthogonal complement
        v = np.matmul(PH_perp, v)
        
        # Normalize the vector
        v /= np.linalg.norm(v)
        
        # Assign the normalized vector to the corresponding column of H
        H[:, i] = v[:, 0]
    
    return H
###########################################################################
###########################################################################
###########################################################################
def random_J_orthogonal_matrix(param):
    D = param.D
    d = param.d
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    #####################################
    H = np.zeros((D+1,d))
    #####################################
    # generate p
    condition = True
    while condition:
        v = np.random.normal(0, 10, (D+1,1))
        v[0] = 100*np.random.normal(0, 1)
        #v[0] = np.sqrt(1+np.linalg.norm(v[1:D+1])**2)
        norm_v = J_norm(v,D)
        if norm_v < 0:
            p = v/np.sqrt(-norm_v)
            if p[0] < 0:
                p = -p
            condition = False
    #####################################
    condition = True
    Pp_perp = np.eye(D+1)+np.matmul(np.matmul(p,p.T),J)
    i = 0
    while condition:
        v = np.random.normal(0, 1, (D+1,1))
        v = np.matmul(Pp_perp,v)
        PH_perp = np.eye(D+1)-np.matmul(np.matmul(H,H.T),J)
        v = np.matmul(PH_perp,v)
        norm_v = J_norm(v,D)
        if norm_v > 0:
            v = v/np.sqrt(norm_v)
            H[:,i] = v[:,0]
            i = i +1
        condition = (i < d)
    return H,p
###########################################################################
###########################################################################
def random_spherical_subspace(param):
    H = random_orthogonal_matrix(param)
    #####################################
    p = H[:,0]
    Hp = np.delete(H,0,1)
    #####################################
    S = subspace()
    S.H = H
    S.Hp = Hp
    S.p = p
    #####################################
    return S
###########################################################################
def random_hyperbolic_subspace(param):
    Hp,p = random_J_orthogonal_matrix(param)
    #####################################
    S = subspace()
    S.Hp = Hp
    S.p = p
    S.H = np.concatenate((p,Hp),1)
    #####################################
    #J = np.eye(101)
    #J[0,0] = -1
    #H_ = S.H
    #print(np.matmul(H_.T, np.matmul(J,H_)) )
    return S
###########################################################################
def random_spherical_tangents(S,param):
    N = param.N
    sigma = param.sigma
    #####################################
    Hp = S.Hp
    p = S.p
    #####################################
    D = np.shape(Hp)[0]-1
    d = np.shape(Hp)[1]
    #####################################
    y = np.random.normal(0, np.pi/4, (d,N))
    Vt = np.matmul(Hp,y)
    #####################################
    p_perp = np.eye(D+1)-np.outer(p,p.T)
    noise = sigma*np.random.normal(0, np.pi/4, (D+1,N))
    noise = np.matmul(p_perp,noise)
    #####################################
    Vt = Vt + noise
    #####################################
    return Vt
###########################################################################
def random_hyperbolic_tangents(S,param):
    N = param.N
    sigma = param.sigma
    #####################################
    Hp = S.Hp
    p = S.p
    #####################################
    D = np.shape(Hp)[0]-1
    d = np.shape(Hp)[1]
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    #####################################
    y = np.random.normal(0, 1, (d,N))
    Vt = np.matmul(Hp,y)
    #####################################
    Pp_perp = np.eye(D+1)+np.matmul(np.matmul(p,p.T),J)
    noise = sigma*np.random.normal(0, 1, (D+1,N))
    noise = np.matmul(Pp_perp,noise)
    #####################################
    Vt = Vt + noise
    #####################################
    return Vt
###########################################################################
def spherical_exp(Vt,S):
    p = S.p
    #####################################
    D = np.shape(Vt)[0]-1
    N = np.shape(Vt)[1]
    X = np.zeros( (D+1, N) )
    for n in range(N):
      v = Vt[:,n]
      norm_v = np.linalg.norm(v)
      x = np.cos(norm_v)*p+(np.sin(norm_v)/norm_v)*v
      X[:,n] = x #/np.linalg.norm(x)
    return X
###########################################################################
###########################################################################
def spherical_log(X,S):
    p = S.p
    #####################################
    D = np.shape(X)[0]-1
    N = np.shape(X)[1]
    V = np.zeros( (D+1,N) )
    for n in range(N):
      x = X[:,n]
      theta = np.arccos( np.matmul(x.T,p) )
      V[:,n] = ( theta/np.sin(theta) ) * ( x-p*np.cos(theta) )
    return V
###########################################################################
###########################################################################
def random_spherical_data(param):
    S = random_spherical_subspace(param)
    #####################################
    Vt = random_spherical_tangents(S,param)
    #####################################
    X = spherical_exp(Vt,S)
    #####################################
    noise_lvl = compute_noise_lvl(X,S)
    return X,S,noise_lvl
###########################################################################
def random_hyperbolic_data(param):
    S = random_hyperbolic_subspace(param)
    #####################################
    Vt = random_hyperbolic_tangents(S,param)
    #####################################
    X = hyperbolic_exp(Vt,S)
    #####################################
    noise_lvl = compute_H_noise_lvl(X,S)
    return X,S,noise_lvl
###########################################################################
def compute_H_noise_lvl(X,S):
    H = S.H
    D = np.shape(H)[0]-1
    d = np.shape(H)[1]-1
    N = np.shape(X)[1]
    #####################################
    J = np.eye(D+1)
    J[0,0] = -1
    J_ = np.eye(d+1)
    J_[0,0] = -1
    #####################################
    X_ = np.matmul(np.matmul(H.T, J),X)
    noise_lvl = 0
    for n in range(N):
        x = X_[:,n]
        tmp = np.sqrt(-np.matmul(x.T,np.matmul(J_,x)))
        tmp = np.maximum(tmp,1)
        noise_lvl = noise_lvl + np.arccosh(tmp)/N
    return noise_lvl
###########################################################################
def compute_noise_lvl(X,S):
    H = S.H
    noise_lvl = np.linalg.norm(np.matmul(H.T,X),2,axis = 0)
    noise_lvl = np.minimum(noise_lvl,1)
    noise_lvl = np.arccos(noise_lvl)
    noise_lvl = np.mean(noise_lvl)
    return noise_lvl
###########################################################################
###########################################################################
def compute_J_evs(Cx,d):
    # D = np.shape(Cx)[0]-1
    # J = np.eye(D+1)
    # J[0,0] = -1
    # evals = []
    # eval_signs = []
    # condition = True
    # count = 0
    # evecs = []
    # while condition:
    #     v = np.random.randn(D+1,1) 
    #     condition_i = True
    #     count_i = 0
    #     count = count + 1 
    #     #print(count)
    #     #print(J_norm(v,D))
    #     while condition_i:
    #         count_i = count_i + 1
    #         v_ = np.matmul(np.matmul(Cx,J),v)
    #         v_ = np.matmul(np.matmul(Cx,J),v_)
    #         v_ = v_/np.sqrt(np.abs(J_norm(v_,D))) 
    #         #print( np.linalg.norm(v-v_)/(D+1) )
    #         condition_i = np.linalg.norm(v-v_)/(D+1) > count*(10**(-10))
    #         v = v_
    #         #if count_i > 100000:
    #         #    count_i = 0
    #         #    v = np.random.randn(D+1,1) 
    #         #print(np.linalg.norm(v-v_)/(D+1) > count*(10**(-10)) )
    #         #print(count)
    #         #print(np.linalg.norm(v-v_))
    #     #print(v)
    #     sgn = np.sign(J_norm(v,D))
    #     #print(sgn)
    #     if count == 1:
    #         evecs = v
    #     else:
    #         evecs = np.concatenate( (evecs,v) , axis = 1)
    #     lmbd = np.matmul(np.matmul(v.T,J), np.matmul(np.matmul(Cx,J),v))
    #     lmbd = np.squeeze(lmbd)
    #     evals = np.append(evals,lmbd)
    #     eval_signs = np.append(eval_signs,sgn)
    #     Cx = Cx - lmbd*np.matmul(v,v.T)
    #     #print(Cx)
    #     condition = check_evals(eval_signs,d)
    #     #print(Cx)
    D = np.shape(Cx)[0]-1
    J = np.eye(D+1)
    J[0,0] = -1
    evals = []
    eval_signs = []
    condition = True
    count = 0
    evecs = []
    while condition:
        v = np.random.randn(D+1,1) 
        condition_i = True
        count_i = 0
        count = count + 1 
        # while condition_i:
        #     count_i = count_i + 1
        #     v_ = np.matmul(np.matmul(Cx,J),v)
        #     v_ = np.matmul(np.matmul(Cx,J),v_)
        #     v_ = v_/np.sqrt(np.abs(J_norm(v_,D))) 
        #     condition_i = np.linalg.norm(v-v_)/(D+1) > count*(10**(-10))
        #     v = v_
        # sgn = np.sign(J_norm(v,D))

        _, v = eigs(np.matmul(Cx,J), k=1)
        v = v/np.sqrt(np.abs(J_norm(v,D))) 
        v = np.real(v)
        sgn = np.sign(J_norm(v,D))
        if count == 1:
            evecs = v
        else:
            evecs = np.concatenate( (evecs,v) , axis = 1)
        lmbd = np.matmul(np.matmul(v.T,J), np.matmul(np.matmul(Cx,J),v))

        lmbd = np.squeeze(lmbd)
        evals = np.append(evals,lmbd)
        eval_signs = np.append(eval_signs,sgn)
        Cx = Cx - lmbd*np.matmul(v,v.T)
        #print(Cx)
        condition = check_evals(eval_signs,d)
        #print(Cx)
    return evals, eval_signs, evecs
########################################################################### 
def check_evals(eval_signs,d):
    evals_p = np.sum(eval_signs>0)
    evals_n = np.sum(eval_signs<0)
    condition = True
    if (evals_p == d) and (evals_n >= 1):
        condition = False
    return condition
########################################################################### 
def subspace_dist(S,S_):
    H = S.H
    H_ = S_.H
    #####################################
    T = np.matmul(H.T,H_)
    #####################################
    SVs = scipy.linalg.svdvals(T)
    #print(SVs)
    #print(T)
    #print(SVs)
    SVs = np.minimum(SVs,1)
    #print(SVs)
    #####################################
    dist =  np.sqrt( np.sum(np.arccos(SVs)**2) )
    return dist
###########################################################################
def subspace_dist_H(S,S_,param):
    H = S.H
    H_ = S_.H
    #####################################
    J = np.eye(param.D+1)
    J[0,0] = -1
    #####################################
    Jd = np.eye(param.d+1)
    J[0,0] = -1
    T = np.matmul(np.matmul(H.T,J), H_)
    T = np.matmul(np.matmul(T.T,Jd),T)
    #####################################
    evals, eval_signs,evecs = compute_J_evs(T,np.shape(T)[0]-1)
    print(evals)
    print(eval_signs)
    return 0
###########################################################################
######################################################################
######################################################################
######################################################################
