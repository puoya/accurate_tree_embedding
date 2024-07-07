import os
import sys
import glob
import torch
import pickle
import logging
import subprocess
from sklearn.cluster import KMeans

import imageio



import scipy.stats as stats

import numpy as np
import pandas as pd
import scipy.linalg
from tqdm import tqdm

from Bio import Phylo
from pathlib import Path

from scipy.stats import wasserstein_distance
import geom.poincare as poincare
import geom.euclidean as euclidean
from learning.frechet import Frechet
from scipy.sparse.linalg import eigs
import geom.hyperboloid as hyperboloid
from geom.horo import busemann, project_kd
from learning.pca import EucPCA, TangentPCA,PGA, HoroPCA, BSA
from utils.metrics import compute_metrics


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import spaceform_pca_lib as sfpca


from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr


import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
###########################################################################
###########################################################################
###########################################################################
def extract_trees(dataset_name):
    # Define the output directory and create it if it doesn't exist
    output_dir = "datasets/" + dataset_name + "/output_trees"
    os.makedirs(output_dir, exist_ok=True)

    # Log and print information about parsing the trees
    tqdm.write("Parse tree files.")
    logging.info("Parse tree files.")

    # Define the source file directory
    source_file_path = "datasets/" + dataset_name + "/tree_directory"
    
    # Check if the source file directory exists
    if not os.path.exists(source_file_path):
        tqdm.write("The source file directory does not exist.")
        logging.info("The source file directory does not exist.")
        return

    directory = Path(source_file_path)
    # Get all .tre and .trees files from the directory
    all_files = list(directory.glob('*.tre')) + list(directory.glob('*.trees'))
    file_names = [file.stem for file in all_files] 

    # If no .tre or .trees files are found, log and print the information
    if not all_files:
        tqdm.write(f"No .tre or .trees files found in the directory: {dataset_name}")
        logging.info(f"No .tre or .trees files found in the directory: {dataset_name}")
        return

    # Open each file and perform the desired operation
    i = 0
    for source_file_path in all_files:
        i += 1
        tree_name = "tree_" + str(i)
        n = 0

        # Count the number of lines (trees) in the source file
        with source_file_path.open("r") as source_file:
            num_lines = sum(1 for _ in source_file)

        # Initialize the progress bar for the current source file
        with tqdm(total=num_lines, unit="tree", dynamic_ncols=True) as pbar:
            with source_file_path.open("r") as source_file:
                # Log and print the filename and number of trees
                logging.info(f"Filename: {source_file_path.stem}")
                logging.info(f"Number of trees: {num_lines}")
                tqdm.write(f"Filename: {source_file_path.stem}")
                tqdm.write(f"Number of trees: {num_lines}")
                
                # Read each line from the file
                for line in source_file:
                    # Increment the counter
                    n += 1
                    # Construct the filename for each tree
                    filename = os.path.join(output_dir, f"{tree_name}_{n}.tre")
                    # Open a new file for writing the tree
                    with open(filename, "w") as tree_file:
                        # Write the tree (line) to the file
                        tree_file.write(line.strip() + '\n')  # strip() removes any extra whitespace/newline and '\n' adds a clean newline

                    # Update the progress bar
                    pbar.set_postfix({"file": n})
                    pbar.update(1)
###########################################################################
###########################################################################
###########################################################################
def process_all_trees(dataset_name):
    # Log and print information about the processing
    tqdm.write("Compute distance matrices for trees.")
    logging.info("Compute distance matrices for trees.")

    tqdm.write(f"Dataset: {dataset_name}")
    logging.info(f"Dataset: {dataset_name}")
    
    # Define input and output directories
    input_directory = "datasets/" + dataset_name + "/output_trees"

    # Check if the input directory exists
    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    output_directory = "datasets/" + dataset_name + "/distance_matrices"
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Get list of all .tre files in the input directory
    tree_files = [f for f in os.listdir(input_directory) if f.endswith(".tre")]

    # Check if there are any .tre files to process
    if len(tree_files) == 0:
        tqdm.write(f"No .tre files found in the directory.")
        logging.info(f"No .tre files found in the directory.")
        return

    # Loop through all .tre files with a progress bar
    with tqdm(total=len(tree_files), unit="file", dynamic_ncols=True) as pbar:
        for filename in tree_files:
            tree_path = os.path.join(input_directory, filename)
            # Calculate the distance matrix from the tree file
            distance_matrix = tree_distance_matrix(tree_path)
            # Prepare the output filename
            output_filename = "D_" + os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_directory, output_filename)
            # Save the distance matrix to a file
            np.save(output_path, distance_matrix)
            # Log and update progress bar
            pbar.set_postfix({"file": filename})
            logging.info(f"Processed file: {filename}")
            pbar.update(1)
###########################################################################
###########################################################################
###########################################################################
def tree_distance_matrix(tree_file):
    # Read the phylogenetic tree from the given file in Newick format
    tree = Phylo.read(tree_file, "newick")
    
    # Get all the terminal (leaf) nodes from the tree
    terminals = tree.get_terminals()
    
    # Determine the number of terminal nodes
    n = len(terminals)
    
    # Initialize an n x n matrix to store the distances between terminals
    distance_matrix = np.zeros((n, n))
    
    # Iterate over each pair of terminals to compute distances
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the distance between terminal i and terminal j
            distance = tree.distance(terminals[i], terminals[j])
            
            # Since the distance matrix is symmetric, set both [i][j] and [j][i] to the computed distance
            distance_matrix[i][j] = distance_matrix[j][i] = distance
    
    # Return the completed distance matrix
    return distance_matrix
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
def precise_hyperbolic_mds(dataset_name, dimension=None):
    # Log and print the start of the Naive Hyperbolic Embedding Step
    logging.info("Precise Hyperbolic Embedding Step.")
    tqdm.write("Precise Hyperbolic Embedding Step.")

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
    output_directory = f'datasets/{dataset_name}/exact_hyperbolic_points'
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
    cnt = 0
    with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            file_path = os.path.join(input_directory, filename)
            # Load the distance matrix from the .npy file
            distance_matrix = np.load(file_path)
            distance_matrix = torch.tensor(distance_matrix)
            scale = 10 / torch.max(distance_matrix)
            matrix_scaled = scale * distance_matrix

            grammian_matrix = -torch.cosh(matrix_scaled)
            
            initial_embedding = np.load(f'datasets/{dataset_name}/hyperbolic_points/X_{filename[7:-4]}.npy')
            initial_embedding = torch.tensor(initial_embedding)
            initial_tangents = hyperbolic_log_torch(initial_embedding)
            
            # save_dir = "training_frames"
            # video_filename = "training_video.mp4"

            # # Create a list of image file paths
            # epochs = 2000
            # image_files = [os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png") for epoch in range(epochs)]
            
            # p = 1/2

            # total_time = 1
            # x = 0
            # # for epoch in range(epochs):
            # #     x += int((epochs + 1)**p / (epoch + 1)**p)
            # for epoch in range(epochs):
            #     x += int(1)

            # x /= (total_time*60)
            # x = int(x)


            # # Create a video writer object
            # with imageio.get_writer(video_filename, fps=x) as writer:
            #     for epoch, image_file in enumerate(image_files):
            #         if epoch == 0:
            #             continue
            #         image = imageio.imread(image_file)
            #         # Calculate frame count based on 1/sqrt(epoch)
            #         frame_count = int((epochs + 1)**p / (epoch + 1)**p)
            #         for _ in range(frame_count):
            #             writer.append_data(image)
            # print(f"Video saved as {video_filename}")

            # matrix_scaled,initial_tangents=initial_tangents, epochs=2000, dim = dimension
            # if cnt == 0:
            tangents, scale = optimize_embedding(matrix_scaled,initial_tangents=initial_tangents, epochs=2000, dim = dimension)
            embedding = hyperbolic_exponential_torch(tangents)
            if scale.requires_grad:
                s = scale.detach().numpy()
            else:
                s = scale.numpy()
            np.save(f'{output_directory}/X_{filename[7:-4]}.npy', embedding.detach().numpy())
            np.save(f'{output_directory}/scale_{filename[7:-4]}.npy', s)
            
            

            # np.save(output_X_file_path, X)
            
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
###########################################################################
###########################################################################
def compute_binned_costs(distance_matrix_paramterized, scale, distance_matrix, num_bins = 10):
    # Compute element-wise operation
    element_wise_cost = (distance_matrix_paramterized - scale * distance_matrix) ** 2 / (scale * distance_matrix) ** 2

    # Extract upper triangular elements
    triu_indices = torch.triu_indices(distance_matrix.size(0), distance_matrix.size(1), offset=1)
    upper_triangular_values = element_wise_cost[triu_indices[0], triu_indices[1]]

    # Sort the values
    sorted_values, _ = torch.sort(upper_triangular_values)

    # Compute bin size
    bin_size = sorted_values.size(0) // num_bins

    # Initialize list to store mean values of bins
    bin_means = []

    for i in range(num_bins):
        if i == num_bins - 1:
            # Last bin takes the remaining elements
            bin_values = sorted_values[i * bin_size:]
        else:
            bin_values = sorted_values[i * bin_size:(i + 1) * bin_size]
        bin_means.append( torch.log(bin_values.mean()).item())

    return bin_means
###########################################################################
###########################################################################
###########################################################################
def generate_weighted_mask(distance_matrix,portion):
    # Compute the probability matrix (weights are inverse of distance)
    prob_matrix = 1 / (distance_matrix)  # Add small value to avoid division by zero
    prob_matrix.fill_diagonal_(0)
    
    prob_matrix /= prob_matrix.sum()  # Normalize to make it a probability distribution

    # Flatten the probability matrix
    prob_flat = prob_matrix.flatten().cpu().numpy()
    nonzero_prob_flat = np.sum(prob_flat>0)


    # Number of elements to select
    num_elements = prob_flat.size
    num_to_select = int(num_elements * portion)
    # print(nonzero_prob_flat,num_to_select,num_elements)
    if num_to_select > nonzero_prob_flat:
        num_to_select = nonzero_prob_flat

    # Generate random indices based on the probability distribution
    selected_indices = np.random.choice(num_elements, num_to_select, replace=False, p=prob_flat)

    # Initialize mask
    mask = torch.zeros(num_elements, dtype=torch.bool, device=distance_matrix.device)
    mask[selected_indices] = True
    
    # Reshape mask to original distance matrix shape
    mask = mask.view_as(distance_matrix)
    prob_matrix[~mask] = 0

    prob_matrix = (prob_matrix + prob_matrix.t())/2
    prob_matrix /= prob_matrix.sum()

    return prob_matrix
###########################################################################
###########################################################################
###########################################################################
def create_bin_masks(distance_matrix, num_bins):
    # Get the number of elements
    n = distance_matrix.size(0)

    # Flatten the upper triangular part (excluding the diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1)
    upper_triangular_values = distance_matrix[triu_indices[0], triu_indices[1]]
    
    # Sort the upper triangular values and get the sorted indices
    sorted_values, sorted_indices = torch.sort(upper_triangular_values,descending= True)

    
    # Compute the size of each bin
    bin_size = len(sorted_values) // num_bins
    remainder = len(sorted_values) % num_bins
    
    # Initialize list to hold mask matrices
    masks = []
    
    start_idx = 0
    for i in range(num_bins):
        end_idx = start_idx + bin_size + (1 if i < remainder else 0)
        
        # Create a mask for the current bin
        mask = torch.zeros(n, n, dtype=torch.bool)
        mask_values = sorted_indices[start_idx:end_idx]
        mask[triu_indices[0][mask_values], triu_indices[1][mask_values]] = True
        mask = mask | mask.t()  # Make the mask symmetric
        
        masks.append(mask)
        
        start_idx = end_idx
    
    return masks
###########################################################################
###########################################################################
###########################################################################
def create_clustered_masks(distance_matrix, num_clusters):
    # Get the number of elements
    n = distance_matrix.size(0)

    # Flatten the upper triangular part (excluding the diagonal)
    triu_indices = torch.triu_indices(n, n, offset=1)
    upper_triangular_values = distance_matrix[triu_indices[0], triu_indices[1]].cpu().numpy()

    # Reshape the data for clustering
    upper_triangular_values = upper_triangular_values.reshape(-1, 1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict( np.log10(upper_triangular_values) )
    centroids = kmeans.cluster_centers_.flatten()
    print(centroids)

    # Sort clusters by centroid values in descending order
    sorted_clusters = sorted(range(num_clusters), key=lambda x: centroids[x], reverse=True)
    print(sorted_clusters)


    # Initialize list to hold mask matrices
    masks = []

    for cluster in sorted_clusters:
        mask = torch.zeros(n, n, dtype=torch.bool)
        mask_indices = (cluster_labels == cluster)
        mask_values = torch.tensor(mask_indices, dtype=torch.bool)

        mask[triu_indices[0][mask_values], triu_indices[1][mask_values]] = True
        mask = mask | mask.t()  # Make the mask symmetric

        masks.append(mask)

    return masks
###########################################################################
###########################################################################
###########################################################################
def cost_function(tangents, distance_matrix, scale_learning= False, scale = torch.tensor(1), unweighted = True, W = None):
    n = tangents.size(1)

    grammian_matrix = -torch.cosh(distance_matrix)
    embedding = hyperbolic_exponential_torch(tangents)

    embedding_ = embedding.clone()
    embedding_[0, :] *= -1
    G = torch.matmul(embedding.t(), embedding_)
    G_clipped = torch.clamp(G, max=-1)
    distance_matrix_paramterized = torch.arccosh(-G_clipped)

    size = distance_matrix.size()
    off_diagonals = torch.ones(size, dtype=torch.bool)
    off_diagonals.fill_diagonal_(False)

    cost = 0

    if scale_learning is not False:
        scale = torch.div( torch.sum(torch.pow(distance_matrix_paramterized,2)),torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix)))

    if unweighted is True:
        cost = (torch.norm( distance_matrix_paramterized[off_diagonals] - scale*distance_matrix[off_diagonals],p='fro')**2 )/ (torch.norm(scale*distance_matrix[off_diagonals],p='fro')**2) 
    else:
        if W is not None:
            distance_matrix_paramterized_w = torch.mul(distance_matrix_paramterized,W)
            distance_matrix_w = torch.mul(distance_matrix,W)
            cost = (torch.norm( distance_matrix_paramterized_w[off_diagonals] - scale*distance_matrix_w[off_diagonals],p='fro')**2 )/ (torch.norm(scale*distance_matrix_w[off_diagonals],p='fro')**2) 
        else:
            print('ERROR')
            return None, None, None
        
    return cost, scale
###########################################################################
###########################################################################
###########################################################################
def cost_function2(tangents, distance_matrix, clusteres_mask = None, scale_learning= False, scale = torch.tensor(1), unweighted = True, W = None):
    n = tangents.size(1)

    grammian_matrix = -torch.cosh(distance_matrix)
    embedding = hyperbolic_exponential_torch(tangents)

    embedding_ = embedding.clone()
    embedding_[0, :] *= -1
    G = torch.matmul(embedding.t(), embedding_)
    G_clipped = torch.clamp(G, max=-1)
    distance_matrix_paramterized = torch.arccosh(-G_clipped)

    size = distance_matrix.size()
    off_diagonals = torch.ones(size, dtype=torch.bool)
    off_diagonals.fill_diagonal_(False)

    cost = 0
    costs = []

    if scale_learning is not False:
        scale = torch.div( torch.sum(torch.pow(distance_matrix_paramterized,2)),torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix)))

    if unweighted is True:
        cost = (torch.norm( distance_matrix_paramterized[off_diagonals] - scale*distance_matrix[off_diagonals],p='fro')**2 )/ (torch.norm(scale*distance_matrix[off_diagonals],p='fro')**2) 
    else:
        if W is not None:
            distance_matrix_paramterized_w = torch.mul(distance_matrix_paramterized,W)
            distance_matrix_w = torch.mul(distance_matrix,W)
            cost = (torch.norm( distance_matrix_paramterized_w[off_diagonals] - scale*distance_matrix_w[off_diagonals],p='fro')**2 )/ (torch.norm(scale*distance_matrix_w[off_diagonals],p='fro')**2) 
        else:
            print('ERROR')
            return None, None, None
    
    for i in range(len(clusteres_mask)):
        mask_i = clusteres_mask[i]
        cost_i = (torch.norm( distance_matrix_paramterized[mask_i] - scale*distance_matrix[mask_i],p='fro')**2 )/ (torch.norm(scale*distance_matrix[mask_i],p='fro')**2)
        costs.append(torch.log10(cost_i).item())
        
        
    return cost, scale, costs
###########################################################################
###########################################################################
###########################################################################
def lorentzian_product(x, y):
    return -x[0] * y[0] + torch.dot(x[1:], y[1:])
###########################################################################
###########################################################################
###########################################################################
class CustomLRScheduler:
    def __init__(self, optimizer, increase_factor=1.1, decrease_factor=0.5,window_size=50, max_increase_count=10):
        self.optimizer = optimizer
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.window_size = window_size
        self.max_increase_count = max_increase_count  # Add this line
        self.losses = []

    def step(self, loss):
        self.losses.append(loss.item())

        # Keep only the latest `window_size` losses
        if len(self.losses) > self.window_size:
            self.losses.pop(0)

        # If we don't have enough data yet, don't change the learning rate
        if len(self.losses) < self.window_size:
            return

        increase_count = sum(1 for x, y in zip(self.losses, self.losses[1:]) if y > x)

        if increase_count > self.max_increase_count:
            self._adjust_learning_rate(self.decrease_factor)
        else:
            # If the loss is consistently decreasing
            if all(x > y for x, y in zip(self.losses, self.losses[1:])):
                self._adjust_learning_rate(self.increase_factor)

    def _adjust_learning_rate(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
###########################################################################
###########################################################################
###########################################################################
# Function to optimize the embedding
def optimize_embedding(distance_matrix, initial_tangents=None, learning_rate=0.1, epochs=1000, dim = 2):

    n = distance_matrix.size(0)
    # Initialize embedding
    if initial_tangents is None:
        tangents = torch.rand(dim,n, requires_grad=True)
        tangents.data.mul_(0.01)
    else:
        tangents = initial_tangents.clone().detach().requires_grad_(True)

    # Define optimizer
    optimizer_tangents = optim.Adam([tangents], lr=learning_rate)

    ratio_curv = 0.3
    ratio_window = 0.025
    num_bins = 200

    epochs_to_estimate_curvature = int(ratio_curv*epochs)
    window_size = int(ratio_window*epochs)

    increase_factor = 1.001
    decrease_factor = 0.98
    max_increase_count = window_size//10

    # masks = create_bin_masks(distance_matrix, num_bins)
    # masks = create_clustered_masks(distance_matrix, num_bins)
    custom_scheduler = CustomLRScheduler(optimizer_tangents, increase_factor, decrease_factor, window_size, max_increase_count)

    # means = []
    # for i in range(len(masks)):
    #     W = masks[i]
    #     means.append(torch.mean(distance_matrix[W]).item())
    

    # Initialize the figure and axes outside the loop
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # fig.set_size_inches(12, 8)  # Increase figure size as needed (in inches)
    # fig.set_dpi(600)  # Increase DPI for higher resolution

    # # Reduce the space between subplots
    # plt.subplots_adjust(hspace=0.1)  # Adjust hspace as needed
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the figure
    # display(fig)

    # save_dir = "training_frames"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    size = distance_matrix.size()
    off_diagonals = torch.ones(size, dtype=torch.bool)
    off_diagonals.fill_diagonal_(False)

    scale_range = torch.log10(torch.mean(distance_matrix[off_diagonals]))-torch.log10(torch.min(distance_matrix[off_diagonals]))

    p = 10**(-scale_range.item() / (epochs-epochs_to_estimate_curvature) )
    # average p to cover the scale_range
    
    scale = torch.tensor(1)
    for epoch in range(epochs):
        # Optimize tangents
        optimizer_tangents.zero_grad()
        if epoch < epochs_to_estimate_curvature:
            # if epoch % 100 == 99:
            #     loss, scale, costs = cost_function2(tangents, distance_matrix, clusteres_mask = masks, scale_learning = True, unweighted = True)
            # else:
            loss, scale = cost_function(tangents, distance_matrix, scale_learning = True, unweighted = True)
            loss.backward()
            with torch.no_grad():
                tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)
            optimizer_tangents.step()
            custom_scheduler.step(loss)      
        else:
            scale_grad_free = torch.tensor(scale.item())
            r =  (epoch-epochs_to_estimate_curvature)/(epochs-1-epochs_to_estimate_curvature)
            W = torch.pow(scale_grad_free*distance_matrix,-r)
            W.fill_diagonal_(1)
            # if epoch % 100 == 99:
            #     loss, _, costs = cost_function2(tangents, distance_matrix, clusteres_mask = masks, scale_learning = False, scale = scale_grad_free, unweighted = False, W= W)
            # else:
            loss, _ = cost_function(tangents, distance_matrix, scale_learning = False, scale = scale_grad_free, unweighted = False, W = W)
            loss.backward()
            with torch.no_grad():
                tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)
            optimizer_tangents.step()
            for param_group in optimizer_tangents.param_groups:
                # param_group['lr'] *= p # start slow, then speed up
                param_group['lr'] *= 10**(2*r*np.log10(p)) # start slow, then speed up
        
        # if epoch % 100 == 99:
        #     ax1.clear()
        #     ax2.clear()

        #     # Plot costs on the first subplot
        #     colors = ['#6495ED'] * num_bins  # Use medium blue for default color
        #     # colors[cnt] = '#FF0000'  # Use vibrant red for the bin at index `cnt`
        #     ax1.bar(range(num_bins), costs, color=colors)
        #     ax1.set_ylabel('Costs', color='blue')
        #     ax1.set_ylim(-5, 6)  # Set consistent y-axis limits
        #     ax1.tick_params(axis='y', labelcolor='blue')

        #     # Plot means on the second subplot
        #     ax2.bar(range(num_bins), np.log10(means), color='green', alpha=0.5)
        #     ax2.set_xlabel('Bin')
        #     ax2.set_ylabel('Means', color='green')
        #     ax2.set_ylim(-6, 1)  # Set consistent y-axis limits
        #     ax2.tick_params(axis='y', labelcolor='green')

        #     # Title and adjust layout
        #     fig.suptitle(f'Epoch {epoch + 1}')
        #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        #     # Save figure
        #     save_dir = "training_frames"
        #     fig_path = os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png")
        #     fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)  # Use tight layout and minimal padding

        #     # Update the figure display
        #     clear_output(wait=True)
        #     display(fig)
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        #     # Optionally, pause for a short time to allow display to update
        #     plt.pause(0.01)

        current_lr = optimizer_tangents.param_groups[0]['lr']
        print(f"Epoch {epoch}, Loss: {loss.item()}, Scale: {scale.item()}, Learning Rate: {current_lr}")

    return tangents,scale
###########################################################################
###########################################################################
###########################################################################
#optimizer = optim.Adam([tangents], lr=learning_rate)
def update_scale(tangents, distance_matrix,scale):
    n = tangents.size(1)
    scaled_distance_matrix = scale*distance_matrix
    embedding = hyperbolic_exponential_torch(tangents)

    embedding_1 = embedding.clone()
    embedding_1[0, :] *= -1
    XTX = torch.matmul(embedding.t(), embedding_1)
    XTX_clipped = torch.clamp(XTX, max=-1)
    distance_matrix_paramterized = torch.arccosh(-XTX_clipped)

    scale = torch.div( torch.sum(torch.mul(distance_matrix_paramterized,scaled_distance_matrix)), 
            torch.sum(torch.mul(scaled_distance_matrix,scaled_distance_matrix)))

    return scale
###########################################################################
###########################################################################
###########################################################################
    #print('scaled',distance_matrix_paramterized)
    #print(scale)

    #cost = torch.norm(XTX_clipped - grammian_matrix) / torch.norm(grammian_matrix)
    #cost = torch.sum( torch.pow(D_X - grammian_matrix,2) )/ torch.sum(torch.pow(grammian_matrix,2) )
    #print(cost)
    

def compute_pca_results(dataset_name, method):
    """
    Compute PCA results for a given dataset using the specified method.

    Parameters:
    dataset_name (str): Name of the dataset.
    method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

    Returns:
    None
    """
    logging.info("Hyperbolic PCA Step.")
    tqdm.write("Hyperbolic PCA Step.")

    # Check if the provided method is valid
    if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
        tqdm.write(f"Method {method} is not supported.")
        logging.error(f"Method {method} is not supported.")
        return
    
    # Define directory paths
    distance_file = 'hyperbolic_points'
    input_directory = f'datasets/{dataset_name}/{distance_file}' 

    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    output_directory = f'datasets/{dataset_name}/{method}/subspaces' 
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    logging.info(f"Dataset: {dataset_name} \t Method: {method}")
    tqdm.write(f"Dataset: {dataset_name} \t Method: {method}")

    # List all .npy files in the input directory that start with 'X'
    npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
    if len(npy_files) == 0:
        tqdm.write(f"No .npy files found in the directory.")
        logging.info(f"No .npy files found in the directory.")
        return

    # Set up a progress bar to show the progress of processing files
    with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:    
        for filename in npy_files:
            try:
                # Get the full path of the current file
                file_path = os.path.join(input_directory, filename)
                # Extract the index from the filename (strip 'X' and '.npy')
                file_index = filename[2:-4]
                # Load the data from the .npy file
                X = np.load(file_path)
                
                # Perform different processing based on the method
                if method in ['sfpca', 'pga']:
                    if method == 'sfpca':
                        # Estimate the hyperbolic subspace using 'sfpca'
                        S = estimate_hyperbolic_subspace(X)
                    else:
                        # Estimate the hyperbolic subspace using 'pga'
                        S = estimate_hyperbolic_subspace_pga(X)
                    # Save the estimated subspace object as a .pkl file
                    with open(os.path.join(output_directory, 'subspace_' + file_index + '.pkl'), 'wb') as file:
                        pickle.dump(S, file)
                elif method in ['horopca', 'bsa']:
                    # Run dimensionality reduction using the specified method
                    Q, mu = run_dimensionality_reduction(method, X)
                    # Save the results as .pt files
                    torch.save(Q, os.path.join(output_directory, 'Q_' + file_index + '.pt'))
                    torch.save(mu, os.path.join(output_directory, 'mu_' + file_index + '.pt'))

                # Update the progress bar with the current file being processed
                pbar.set_postfix({"file": filename})
                pbar.update(1)
                logging.info(f"Processed file: {filename}")

            except Exception as e:
                # Log any errors that occur during processing
                logging.error(f"Error processing file {filename}: {e}")
                continue
###########################################################################
###########################################################################
###########################################################################
def compute_mds_results(dataset_name, method):
    """
    Compute MDS results for a given dataset using the specified method.

    Parameters:
    dataset_name (str): Name of the dataset.
    method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

    Returns:
    None
    """
    logging.info("Compute MDS Costs.")
    tqdm.write("Compute MDS Costs.")

    # Check if the provided method is valid
    if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
        tqdm.write(f"Method {method} is not supported.")
        logging.error(f"Method {method} is not supported.")
        return

    # Define directory paths
    distance_file = 'hyperbolic_points'
    base_directory = f'datasets/{dataset_name}'
    subspace_directory = f'{base_directory}/{method}/subspaces'
    input_directory = f'{base_directory}/{distance_file}'

    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    tqdm.write(f"Dataset:{dataset_name} \t Method:{method}")
    logging.info(f"Dataset:{dataset_name} \t Method:{method}")

    if not os.path.exists(subspace_directory):
        tqdm.write(f"The subspace directory does not exist.")
        logging.info(f"The subspace directory does not exist.")
        return

    output_directory = f'{base_directory}/{method}/mds'
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    results_df = pd.DataFrame(columns=['file_index', 'dimension', 'mds_error'])

    # Iterate over all relevant files in the input directory

    npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
    if len(npy_files) == 0:
        tqdm.write(f"No .npy files found in the directory.")
        logging.info(f"No .npy files found in the directory.")
        return

    with tqdm(total=len(npy_files), unit=f"file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            file_path = os.path.join(input_directory, filename)
            # Extract the file index from the filename
            file_index = filename[2:-4]
            X = np.load(file_path)
            #print(f'MDS distortion: {method} \t file number:{file_index}' )
            distance_matrix = compute_hyperbolic_distance_matrix(X)

            if method in ['sfpca', 'pga']:
                # Load the subspace instance
                S = load_subspace_instance(subspace_directory, file_index)
                num_dimensions = np.shape(S.H)[1]
            elif method in ['horopca', 'bsa']:
                Q, mu_ref = load_subspace_instance_qmu(subspace_directory, file_index)
                num_dimensions = np.shape(Q)[0] + 1

            pbar.set_postfix({"file": filename})
            pbar.update(1)
            logging.info(f"Processed file: {filename}")

            # Iterate through each dimension and compute the distance matrix and MDS error
            for dimension in range(1, num_dimensions):
                inaccurate = False
                if method in ['sfpca','pga']:
                    distance_matrix_d, inaccurate = compute_distance_matrix(X, S, dimension=dimension, method=method) 
                else:
                    distance_matrix_d, inaccurate = compute_distance_matrix(X, Q, mu_ref, dimension, method)

                # Compute the MDS error
                mds_error = np.linalg.norm(distance_matrix - distance_matrix_d, 'fro') / np.linalg.norm(distance_matrix, 'fro')

                if inaccurate:
                    mds_error = np.nan

                results_df = pd.concat([results_df, pd.DataFrame({
                    'file_index': [file_index],
                    'dimension': [dimension],
                    'mds_error': [mds_error]
                })], ignore_index=True)
            
    # Save the results to a CSV file
    output_file_path = os.path.join(output_directory, 'distortions.csv')
    results_df.to_csv(output_file_path, index=False)
###########################################################################
###########################################################################
###########################################################################
def compute_quartet_results(dataset_name, method):
    """
    Compute Quartet Score results for a given dataset using the specified method.

    Parameters:
    dataset_name (str): Name of the dataset.
    method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

    Returns:
    None
    """
    logging.info("Compute Quartet Scores.")
    tqdm.write("Compute Quartet Scores.")

    # Check if the provided method is valid
    if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
        tqdm.write(f"Method {method} is not supported.")
        logging.error(f"Method {method} is not supported.")
        return

    # Define directory paths
    distance_file = 'hyperbolic_points'
    base_directory = f'datasets/{dataset_name}'
    subspace_directory = f'{base_directory}/{method}/subspaces'
    input_directory = f'{base_directory}/{distance_file}'
    
    if not os.path.exists(input_directory):
        tqdm.write(f"The input directory does not exist.")
        logging.info(f"The input directory does not exist.")
        return

    if not os.path.exists(subspace_directory):
        tqdm.write(f"The subspace directory does not exist.")
        logging.info(f"The subspace directory does not exist.")
        return

    output_directory = f'{base_directory}/{method}/quartet'
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    logging.info(f"Dataset: {dataset_name} \t Method: {method}")
    tqdm.write(f"Dataset: {dataset_name} \t Method: {method}")

    # List all .npy files in the input directory that start with 'X'
    npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
    if len(npy_files) == 0:
        tqdm.write(f"No .npy files found in the directory.")
        logging.info(f"No .npy files found in the directory.")
        return

    L = 10**5
    results_df = pd.DataFrame(columns=['file_index', 'dimension', 'quartet_score'])

    n = 0
    # Iterate over all relevant files in the input directory
    with tqdm(total=len(npy_files), unit=f"file", dynamic_ncols=True) as pbar:
        for filename in npy_files:
            n = n + 1
            np.random.seed(n)

            file_path = os.path.join(input_directory, filename)
            # Extract the file index from the filename
            file_index = filename[2:-4]
            X = np.load(file_path)
            N = np.shape(X)[1]

            pbar.set_postfix({"file": filename})
            pbar.update(1)
            logging.info(f"Processed file: {filename}")

            distance_matrix = compute_hyperbolic_distance_matrix(X)

            random_matrix = np.random.uniform(0, 1, (2*L, 4))*N
            integer_matrix = select_unique_combinations(random_matrix,L)
            L_ = np.shape(integer_matrix)[0]

            topology = []
            for i in range(L_):
                index = integer_matrix[i,:]
                distance_matrix_i = distance_matrix[index,:].copy()
                distance_matrix_i = distance_matrix_i[:,index]
                topology.append( determine_best_topology(distance_matrix_i)) 
            topology = np.array(topology)


            if method in ['sfpca', 'pga']:
                # Load the subspace instance
                S = load_subspace_instance(subspace_directory, file_index)
                num_dimensions = np.shape(S.H)[1]
            elif method in ['horopca', 'bsa']:
                Q, mu_ref = load_subspace_instance_qmu(subspace_directory, file_index)                
                num_dimensions = np.shape(Q)[0] + 1

            # Iterate through each dimension and compute the distance matrix and MDS error
            for dimension in range(1, num_dimensions):
                if method in ['sfpca','pga']:
                    distance_matrix_d, inaccurate = compute_distance_matrix(X, S, dimension=dimension, method=method)  
                else:
                    distance_matrix_d,inaccurate = compute_distance_matrix(X, Q, mu_ref, dimension, method)

                topology_d = []
                for i in range(L_):
                    index = integer_matrix[i,:]
                    distance_matrix_di = distance_matrix_d[index,:].copy()
                    distance_matrix_di = distance_matrix_di[:,index]
                    topology_d.append( determine_best_topology(distance_matrix_di)) 
                topology_d = np.array(topology_d)

                accuracy =  np.sum(np.abs(topology_d-topology) == 0)/L_

                if inaccurate:
                    accuracy = np.nan
            
                results_df = pd.concat([results_df, pd.DataFrame({
                    'file_index': [file_index],
                    'dimension': [dimension],
                    'quartet_score': [accuracy]
                })], ignore_index=True)
            # # Save the results to a CSV file
            output_file_path = os.path.join(output_directory, 'accuracies.csv')
            results_df.to_csv(output_file_path, index=False)

    # # Save the results to a CSV file
    # output_file_path = os.path.join(output_directory, 'accuracies.csv')
    # results_df.to_csv(output_file_path, index=False)
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
def determine_best_topology(distance_matrix):
    """
    Determine the topology based on the four-point condition for a given 4x4 distance matrix.

    Parameters:
    distance_matrix (ndarray): A 4x4 distance matrix.

    Returns:
    int: The topology index (1, 2, or 3).
    """

    # Calculate the sums of the opposite pairs
    sum_opposite_1 = distance_matrix[0, 1] + distance_matrix[2, 3]
    sum_opposite_2 = distance_matrix[0, 2] + distance_matrix[1, 3]
    sum_opposite_3 = distance_matrix[0, 3] + distance_matrix[1, 2]

    # Find the minimum sum among the three calculated sums
    minimum_sum = min(sum_opposite_1, sum_opposite_2, sum_opposite_3)

    # Determine the topology index based on which sum is the minimum
    if minimum_sum == sum_opposite_1:
        topology_index = 1
    elif minimum_sum == sum_opposite_2:
        topology_index = 2
    else:
        topology_index = 3

    return topology_index
###########################################################################
###########################################################################
###########################################################################
def select_unique_combinations(random_matrix, L):
    """
    Select unique integer combinations from a random matrix, ensuring no duplicates in each row.

    Parameters:
    random_matrix (np.ndarray): Matrix of random values.
    L (int): Number of unique rows to select.

    Returns:
    np.ndarray: Matrix with L unique rows.
    """
    # Convert the random matrix to integers
    integer_matrix = random_matrix.astype(int)
    
    # Sort each row to easily identify duplicates
    sorted_matrix = np.sort(integer_matrix, axis=1)
    
    # Identify rows with duplicate elements
    has_duplicates = np.any(sorted_matrix[:, :-1] == sorted_matrix[:, 1:], axis=1)
    
    # Filter out rows with duplicates
    unique_matrix = integer_matrix[~has_duplicates]
    
    # Select the first L unique rows
    unique_matrix = unique_matrix[:L, :]
    
    return unique_matrix
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
def lgram_to_points(dimension, gram_matrix):
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
class FourLayerNN(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(FourLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
######################################################################
######################################################################
######################################################################
def train_neural_network(X, y, test_size=0.1, stratify=None, seed=None, num_iterations=20):
    accuracies = []

    np.random.seed(seed)  # For reproducibility of random splits
    
    for iteration in range(num_iterations):
        torch.manual_seed(seed+iteration)
        
        state= np.random.randint(0, 10000)
        #print(state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=state)
    
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        input_dim = X.shape[1]
        hidden1_dim = 512
        hidden2_dim = 128
        hidden3_dim = 32
        output_dim = len(np.unique(y))
        model = FourLayerNN(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predicted.numpy())
            accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    return average_accuracy
######################################################################
######################################################################
######################################################################
def train_svm_and_evaluate(X, y, seed):
    np.random.seed(seed)  # For reproducibility of random splits
    accuracies = []  # To store accuracy of each split
    
    for _ in range(20):  # Perform 20 random splits and evaluations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=np.random.randint(0, 10000))
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    average_accuracy = np.mean(accuracies)
    return average_accuracy
######################################################################
######################################################################
######################################################################
def train_random_forest_and_evaluate(X, y, seed,max_tree_depth=20):
    np.random.seed(seed)  # For reproducibility of random splits
    accuracies = []  # To store accuracy of each split

    max_d = 0
    for _ in range(20):  # Perform 20 random splits and evaluations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=np.random.randint(0, 10000))
        #print(max_tree_depth)
        model = RandomForestClassifier(max_depth=max_tree_depth)
        model.fit(X_train, y_train)
        #tree_depths = [tree.get_depth() for tree in model.estimators_]
        #max_d = max( [max_d , max(tree_depths) ]) 
        #print(max_d)
        y_pred = model.predict(X_test)    
        accuracies.append(accuracy_score(y_test, y_pred))
    average_accuracy = np.mean(accuracies)
    
    return average_accuracy
######################################################################
######################################################################
######################################################################
def run_spherical_mds(dataset_name):
    class parameters:
        def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
            self.D = D
            self.d = d
            self.N = N
            self.sigma = sigma
    ######################################################################
    def load_data(dataset_name):
        if dataset_name == 'GUniFrac':
            address = "datasets/GUniFrac/doc/csv/"
            X = np.load(address+'X.npy')
            X = X.T
        elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
            address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
            X = np.load(address+'X.npy')
            X = X.T
            N, D = np.shape(X)
            for n in range(N):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
        elif dataset_name == 'document':
            address = "datasets/document/"
            X = np.load(address+'X.npy').astype(float)
            address = "datasets/document/"
            label = np.load(address+'label.npy')
            idx = (label==0) + (label == 1)
            X = X[idx,:]
            label = label[idx]
            np.random.seed(42)
            N0 = 400
            N = np.shape(X)[0]
            idx = np.random.choice(N, N0, replace=False)
            X = X[idx,:]
            for n in range(N0):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
        return X
    ######################################################################
    directory = "datasets/" +dataset_name +"/results/" 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    X = load_data(dataset_name)

    # Initial parameters
    D,N = np.shape(X)
    D -= 1
    
    DM = compute_sdm(X)
    AIDM = compute_aidm(X)
    KLDM = compute_kldm(X)
    TVDM = compute_tvdm(X)
    JSDM = compute_jsdm(X)

    # A list of functions to run
    functions = [
        estimate_spherical_subspace,
        estimate_spherical_subspace_liu,
        estimate_spherical_subspace_dai,
        #estimate_spherical_subspace_pga,
        estimate_spherical_subspace_pga_2,
    ]

    # Prepare a list to collect the data
    data_list = []
    for d in range(2,D):
        print(f"Running with d = {d}")
        param = parameters(D=D, d=d, N=N, sigma = 0)
        cnt = 0
        for func in functions:
            cnt = cnt +1
            directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
            if not os.path.exists(directory_):
                continue
            X_ = np.load(directory_+'X_.npy')
            S_ = np.load(directory_+'S_.npy', allow_pickle=True)

            AIDM_ = compute_aidm(X_)
            KLDM_ = compute_kldm(X_)
            DM_ = compute_sdm(X_)
            TVDM_ = compute_tvdm(X_)
            JSDM_ = compute_jsdm(X_)

            sphere_dist = total_distance(X, X_)
            dist_distortion = distance_distortion(DM_, DM)
            ai_distortion = distance_distortion(AIDM_, AIDM)
            kl_distortion = distance_distortion(KLDM_, KLDM)
            tv_distortion = distance_distortion(TVDM_, TVDM)
            js_distortion = distance_distortion(JSDM_, JSDM)

            data_list.append({"d": d, "Method": func.__name__, 
                "sphere_dist":sphere_dist,
                "ai_distortion":ai_distortion, 
                "kl_distortion":kl_distortion,
                "js_distortion":js_distortion,
                "tv_distortion":tv_distortion,
                "dist_distortion": dist_distortion
                })
            print("[",sphere_dist, ai_distortion, kl_distortion, js_distortion, tv_distortion,dist_distortion, "]")
            results_df = pd.DataFrame(data_list)
        print('############################################')
        #results_df.to_csv(results_filename, index=False)
        if d >= min(np.shape(X))-1:
            break

######################################################################
######################################################################
######################################################################
def run_spherical_classifier(dataset_name):
    class parameters:
        def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
            self.D = D
            self.d = d
            self.N = N
            self.sigma = sigma
    ######################################################################
    def load_data(dataset_name):
        if dataset_name == 'GUniFrac':
            address = "datasets/GUniFrac/doc/csv/"
            X = np.load(address+'X.npy')
            X = X.T
            y = np.load('datasets/GUniFrac/doc/csv/SmokingStatus_categories.npy')
        elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
            address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
            X = np.load(address+'X.npy')
            X = X.T
            N, D = np.shape(X)
            for n in range(N):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
            y = np.load('datasets/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
        elif dataset_name == 'document':
            address = "datasets/document/"
            X = np.load(address+'X.npy').astype(float)
            address = "datasets/document/"
            label = np.load(address+'label.npy')
            idx = (label==0) + (label == 1)
            X = X[idx,:]
            label = label[idx]
            np.random.seed(42)
            N0 = 400
            N = np.shape(X)[0]
            idx = np.random.choice(N, N0, replace=False)
            X = X[idx,:]
            y = label[idx]
            for n in range(N0):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
        return X,y
    ######################################################################
    directory = "datasets/" +dataset_name +"/results/" 
    #directory = "../results/" + dataset_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    results_filename = os.path.join(directory, "classification_results_rf.csv")
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
        print("Results already exist. Loaded results:")
    else:
        print("Results file does not exist. Computing the results...")
        X,y = load_data(dataset_name)
        ind = (y != 3)
        y = y[ind]
        #y = y[ind]
        #print(y)

        D,N = np.shape(X)
        D = D-1
        
        functions = [
            sfpca.estimate_spherical_subspace,
            sfpca.estimate_spherical_subspace_liu,
            sfpca.estimate_spherical_subspace_dai,
            #sfpca.estimate_spherical_subspace_pga,
            sfpca.estimate_spherical_subspace_pga_2,
        ]

        # Prepare a list to collect the data
        data_list = []
        seed = 100000
        #average_acc_rf = train_random_forest_and_evaluate(X.T**2, y, seed=seed)

        for d in range(2,D):
            print(f"Running with d = {d}")
            param = parameters(D=D, d=d, N=N, sigma = 0)
            for func in functions:
                directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
                if not os.path.exists(directory_):
                    continue
                X_ = np.load(directory_+'X_.npy')
                S_ = np.load(directory_+'S_.npy', allow_pickle=True)
                H = S_.item().H
                X_low = np.matmul(H.T,X_)

                X_ = X_[:,ind]
                X_low = X_low[:,ind]
                
                #average_acc_nn = train_neural_network(X_.T**2, y, seed=seed)
                #average_acc_svm = train_svm_and_evaluate(X_.T**2, y, seed=seed)
                average_acc_rf = train_random_forest_and_evaluate(X_.T**2, y, seed=seed)

                #average_acc_nn_low = train_neural_network(X_low.T**2, y, seed=seed)
                #average_acc_svm_low = train_svm_and_evaluate(X_low.T**2, y, seed=seed)
                #average_acc_rf_low = train_random_forest_and_evaluate(X_low.T**2, y, seed=seed)
                
                data_list.append({"d": d, "Method": func.__name__, 
                    #"average_acc_nn":average_acc_nn,
                    #"average_acc_svm":average_acc_svm,
                    "average_acc_rf":average_acc_rf
                    #"average_acc_nn_low":average_acc_nn_low,
                    #"average_acc_svm_low":average_acc_svm_low,
                    #"average_acc_rf_low":average_acc_rf_low
                    })
                #print("[",average_acc_nn, average_acc_svm,average_acc_rf,average_acc_nn_low, average_acc_svm_low,average_acc_rf_low, "]")
                print("[",average_acc_rf, "]")
                results_df = pd.DataFrame(data_list)
                data_list.append({"d": d, "Method": func.__name__, 
                    "average_acc_rf":average_acc_rf
                    })
                #print("[",average_acc_rf, "]")
                results_df = pd.DataFrame(data_list)
            print('############################################')
            results_df.to_csv(results_filename, index=False)
            if d >= min(np.shape(X))-1:
               break
######################################################################
def run_spherical_information(dataset_name):
    class parameters:
        def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
            self.D = D
            self.d = d
            self.N = N
            self.sigma = sigma
    ######################################################################
    def load_data(dataset_name):
        if dataset_name == 'GUniFrac':
            address = "datasets/GUniFrac/doc/csv/"
            X = np.load(address+'X.npy')
            X = X.T
            y = np.load('datasets/GUniFrac/doc/csv/SmokingStatus_categories.npy')
        elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
            address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
            X = np.load(address+'X.npy')
            X = X.T
            N, D = np.shape(X)
            for n in range(N):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
            y = np.load('datasets/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
        elif dataset_name == 'document':
            address = "datasets/document/"
            X = np.load(address+'X.npy').astype(float)
            address = "datasets/document/"
            label = np.load(address+'label.npy')
            idx = (label==0) + (label == 1)
            X = X[idx,:]
            label = label[idx]
            np.random.seed(42)
            N0 = 400
            N = np.shape(X)[0]
            idx = np.random.choice(N, N0, replace=False)
            X = X[idx,:]
            y = label[idx]
            for n in range(N0):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
        return X,y
    ######################################################################
    directory = "datasets/" +dataset_name +"/results/" 
    #directory = "../results/" + dataset_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    results_filename = os.path.join(directory, "information_results.csv")
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
        print("Results already exist. Loaded results:")
    else:
        print("Results file does not exist. Computing the results...")
        X,y = load_data(dataset_name)
        ind = (y != 3)
        y = y[ind]

        D,N = np.shape(X)
        D = D-1
        
        functions = [
            sfpca.estimate_spherical_subspace,
            sfpca.estimate_spherical_subspace_liu,
            sfpca.estimate_spherical_subspace_dai,
            #sfpca.estimate_spherical_subspace_pga,
            sfpca.estimate_spherical_subspace_pga_2,
        ]

        # Prepare a list to collect the data
        data_list = []
        for d in range(2,D):
            print(f"Running with d = {d}")
            param = parameters(D=D, d=d, N=N, sigma = 0)
            cnt = 0
            for func in functions:
                cnt = cnt +1
                directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
                if not os.path.exists(directory_):
                    continue
                X_ = np.load(directory_+'X_.npy')
                S_ = np.load(directory_+'S_.npy', allow_pickle=True)
                H = S_.item().H
                X_low = np.matmul(H.T,X_)

                X_ = X_[:,ind]
                X_low = X_low[:,ind]

                average_mi = compute_mutual_information(X_.T, y)
                average_mi_low = compute_mutual_information(X_low.T, y)

                data_list.append({"d": d, "Method": func.__name__, 
                    "average_mi":average_mi,
                    "average_mi_low":average_mi_low
                    })
                print("[",average_mi, average_mi_low, "]")
                results_df = pd.DataFrame(data_list)
            print('############################################')
            results_df.to_csv(results_filename, index=False)
            if d >= min(np.shape(X))-1:
                break


def compute_sparsity(matrix, threshold=0):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    zero_elements = np.sum(np.abs(matrix) <= threshold)
    total_elements = matrix.size
    sparsity_level = zero_elements / total_elements
    
    return sparsity_level
######################################################################
def run_spherical_sparsity(dataset_name):
    class parameters:
        def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
            self.D = D
            self.d = d
            self.N = N
            self.sigma = sigma
    ######################################################################
    def load_data(dataset_name):
        if dataset_name == 'GUniFrac':
            address = "datasets/GUniFrac/doc/csv/"
            X = np.load(address+'X.npy')
            X = X.T
            y = np.load('datasets/GUniFrac/doc/csv/SmokingStatus_categories.npy')
        elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
            address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
            X = np.load(address+'X.npy')
            X = X.T
            N, D = np.shape(X)
            for n in range(N):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
            y = np.load('datasets/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
        elif dataset_name == 'document':
            address = "datasets/document/"
            X = np.load(address+'X.npy').astype(float)
            address = "datasets/document/"
            label = np.load(address+'label.npy')
            idx = (label==0) + (label == 1)
            X = X[idx,:]
            label = label[idx]
            np.random.seed(42)
            N0 = 400
            N = np.shape(X)[0]
            idx = np.random.choice(N, N0, replace=False)
            X = X[idx,:]
            y = label[idx]
            for n in range(N0):
                X[n,:] = X[n,:]/ np.sum(X[n,:])
                X[n,:] = np.sqrt(X[n,:])
            X = X.T
        return X,y
    ######################################################################
    directory = "datasets/" +dataset_name +"/results/" 
    results_filename = os.path.join(directory, "sparsity_results2.csv")
    if os.path.exists(results_filename):
        # Load the DataFrame from the CSV file
        results_df = pd.read_csv(results_filename)
        # Print the loaded results
        print("Results already exist. Loaded results:")
    else:
        print("Results file does not exist. Computing the results...")
        X,y = load_data(dataset_name)
        # Initial parameters
        D,N = np.shape(X)
        D = D-1
        # A list of functions to run
        functions = [
            sfpca.estimate_spherical_subspace,
            sfpca.estimate_spherical_subspace_liu,
            sfpca.estimate_spherical_subspace_dai,
            #sfpca.estimate_spherical_subspace_pga,
            sfpca.estimate_spherical_subspace_pga_2,
        ]

        # Prepare a list to collect the data
        data_list = []
        threshold = 10**(-4)/D
        S = compute_sparsity(X**2, threshold)
        for d in range(2,D):
            #print(f"Running with d = {d}")
            param = parameters(D=D, d=d, N=N, sigma = 0)
            cnt = 0
            for func in functions:
                cnt = cnt +1
                directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
            
                if not os.path.exists(directory_):
                    continue
                X_ = np.load(directory_+'X_.npy')
                #print(X_)
                
                S_ = compute_sparsity(X_**2-X**2, threshold)
                

                data_list.append({"d": d, "Method": func.__name__, 
                    "sparsity_dist":np.abs(S-S_),
                    "sparsity":S_
                    })
                print("[",np.abs(S-S_), S_, "]")
                results_df = pd.DataFrame(data_list)
            print('############################################')
            results_df.to_csv(results_filename, index=False)
            if d >= min(np.shape(X))-1:
                break

# cost = torch.norm(distance_matrix_paramterized - scale*distance_matrix,p='fro')**2/ torch.norm(scale*distance_matrix,p='fro')**2

    
    
    # errors = (torch.div(distance_matrix_paramterized , scale * distance_matrix) -1).abs()[mask]
    # errors_flat = errors.view(-1)
    # sorted_errors, _ = torch.sort(errors_flat)
    

    #print( int(epoch / 2000 * (n*(n-1)//2) ), n*(n-1)//2)
    # if epoch > threshold_epoch:
        # Flatten the errors and sort them
        # errors_flat = errors.view(-1)
        # sorted_errors, _ = torch.sort(errors_flat)

        # distances_flat = distance_matrix[mask].view(-1)
        # sorted_distances, _ = torch.sort(-distances_flat)
        # sorted_distances = -sorted_distances


        
        # Determine the number of elements to discard
        # discard_count = int( (epoch-threshold_epoch) / 2000 * (n*(n-1)//2) )
        
        # Determine the threshold value for the discard
        # if discard_count > 0:
        #     # threshold_value = sorted_errors[discard_count]
        #     threshold_value = sorted_distances[discard_count]

        #     # Create a mask to discard smallest errors
        #     # discard_mask = errors <= threshold_value
        #     keep_mask = distance_matrix <= threshold_value
        # else:
        #     # discard_mask = torch.ones_like(errors, dtype=torch.bool)
        #     keep_mask = mask
        # print(keep_mask)
            
        # cost = torch.norm(distance_matrix_paramterized[keep_mask] - scale*distance_matrix[keep_mask],p='fro')**2/torch.norm(scale*distance_matrix[keep_mask],p='fro')**2

        # a = (1/distance_matrix-1)
        # b = 1
        # W = a * ((epoch-threshold_epoch)/(2000-threshold_epoch))**5 + b
        # W = 1/(scale*distance_matrix)
        # W.fill_diagonal_(1)
        # print(W)
        # cost = torch.norm( torch.mul(distance_matrix_paramterized - scale*distance_matrix,W),p='fro')**2/torch.norm(torch.mul(scale*distance_matrix,W),p='fro')**2
        # W = generate_weighted_mask(distance_matrix, 1-(epoch-threshold_epoch)/(2000-threshold_epoch))
        # # print(W)
        # cost = torch.norm( torch.mul(distance_matrix_paramterized - scale*distance_matrix,W),p='fro')**2

        # Apply the mask to the errors
        #cost = torch.sum(torch.pow(errors[discard_mask],2))
    
    # cost = torch.norm(errors, p='fro') ** 2 / torch.norm(scale * distance_matrix, p='fro') ** 2
    

    #cost = torch.mean( torch.pow(torch.div(distance_matrix_paramterized,scale*distance_matrix )[mask]-1,2) )
    
    # difference_matrix = torch.pow(distance_matrix_paramterized - scale*distance_matrix,2)
    # scale_distance_matrix_2 = torch.pow(scale*distance_matrix,2)
    



    #cost = torch.sum( torch.pow(D_X - grammian_matrix,2) )/ torch.sum(torch.pow(grammian_matrix,2) )
    #print(cost)

    #cost = torch.sum(difference_matrix[mask])/torch.sum(scale_distance_matrix_2[mask])
    

    # elif feed_scale is None:
        # scale = torch.div( torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix_paramterized)),
            # torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix)))
        # pdist = torch.div(distance_matrix_paramterized, distance_matrix)[W]
        # scale = torch.div(torch.sum(torch.pow(pdist,2)) , torch.sum(pdist) )
        

        # if weights is not None:
        #     weight = weights[i]
        # else:
        #     weight = 1
        # if i < cnt:
        #     weight *= 1
        # else:
        #     if np.abs(i-cnt) < window:
        #         weight *= p**np.abs(cnt-i)
        #     else:
        #         p = 0
        #         weight *= p**np.abs(cnt-i)
        # if i <= cnt:
        #     weight = 1
        # else:
        #     weight = 1/np.sqrt(np.abs(i-cnt+1))
        # weight = 1
        # if cnt < window:
        #     if np.abs(cnt-i) < window:
        #         weight = p**(np.abs(cnt-i))
        #     else:
        #         weight = 10**(-20) # drastically small
        # else:
        #     small_window = (0.01)*len(masks) # value = 1/200 at 2 (out of 200) 
        #     ratio = np.sqrt( (cnt-window)/(len(masks)-1-window) ) # current progress ... make the window smaller
            
        #     front_window =  ratio * small_window + (1-ratio) * window
        #     pf = np.exp(np.log(threshold)/ (front_window) )

        #     if i < cnt:
        #         if np.abs(cnt-i) < window:
        #             weight = p**(np.abs(cnt-i))
        #         else:
        #             weight = threshold
        #     else:
        #         weight = pf**(np.abs(cnt-i))
        # weight = 1
        #cost += weight*torch.log10(cost_i)
        # new_weights.append(weight)
        # weights.append(weight)
    
    # cost /= np.sum(new_weights)
    # else:

        #     # _, _,_,costs = cost_function(tangents, distance_matrix, masks, scale_update_mode, cnt )
            
        #     training_epochs = int(0.05*epochs)
        #     initial_target = 0
        #     final_target = num_bins-1
            
        #     initial_cnt = epochs_to_estimate_curvature + training_epochs
        #     final_cnt = epochs
        #     if epoch < initial_cnt:
        #         cnt = 0
        #     elif epoch < final_cnt:
        #         a = (final_target-initial_target)/(final_cnt-initial_cnt)
        #         b = initial_target - a* initial_cnt
        #         cnt = int(a * epoch + b)
        #     print(f'cnt is {cnt}')
        #     for param_group in optimizer_tangents.param_groups:
        #         # r = (epoch/epochs)
        #         param_group['lr'] = lrs[cnt]*0.03 #r*decrease_factor2+(1-r)*decrease_factor1



        #     # Optimize tangents
        #     optimizer_tangents.zero_grad()
        #     # loss, scale , distance_matrix_paramterized,costs = cost_function(tangents, distance_matrix, masks, scale_update_mode,cnt,feed_scale = torch.tensor(scale.item()))
        #     lr = np.log(lrs[cnt])
        #     min_lr = np.log(np.min(lrs))
        #     max_lr = np.log(np.max(lrs))
        #     rate = 1-(lr - min_lr)/(max_lr - min_lr)
        #     print(f'rate is {rate}')
        #     # loss, scale , distance_matrix_paramterized, costs = cost_function(tangents, distance_matrix, masks, scale_update_mode,cnt, r = rate)


        #     loss.backward()

        #     with torch.no_grad():
        #         tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)

        #     optimizer_tangents.step()

            
        #     # bin_means = compute_binned_costs(distance_matrix_paramterized, scale, distance_matrix, num_bins)
        #     # Clear previous plot and update with new data
        #     ax1.clear()
        #     ax2.clear()

        #     # Plot costs on the first subplot
        #     colors = ['#6495ED'] * num_bins  # Use medium blue for default color
        #     colors[cnt] = '#FF0000'  # Use vibrant red for the bin at index `cnt`
        #     ax1.bar(range(num_bins), costs, color=colors)
        #     ax1.set_ylabel('Costs', color='blue')
        #     ax1.set_ylim(-5, 6)  # Set consistent y-axis limits
        #     ax1.tick_params(axis='y', labelcolor='blue')

        #     # Plot means on the second subplot
        #     ax2.bar(range(num_bins), np.log10(means), color='green', alpha=0.5)
        #     ax2.set_xlabel('Bin')
        #     ax2.set_ylabel('Means', color='green')
        #     ax2.set_ylim(-6, 1)  # Set consistent y-axis limits
        #     ax2.tick_params(axis='y', labelcolor='green')

        #     # Title and adjust layout
        #     fig.suptitle(f'Epoch {epoch + 1}')
        #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        #     # Save figure
        #     save_dir = "training_frames"
        #     fig_path = os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png")
        #     fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)  # Use tight layout and minimal padding

        #     # Update the figure display
        #     clear_output(wait=True)
        #     display(fig)
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        #     # Optionally, pause for a short time to allow display to update
        #     plt.pause(0.01)



        #     # custom_scheduler.step(loss)
        #     current_lr = optimizer_tangents.param_groups[0]['lr']
        #     print(f"Epoch {epoch}, Loss: {loss.item()}, Scale: {scale.item()}, Learning Rate: {current_lr}")




            # else:
            #     loss, scale , distance_matrix_paramterized, costs = cost_function(tangents, distance_matrix, masks, scale_update_mode,cnt, weights = weights)
            # weights = []
            # for i in range(len(costs)):
            #     weights.append(10**(-costs[i]))
            
            # bin_means = compute_binned_costs(distance_matrix_paramterized, scale, distance_matrix, num_bins)
            # clear_output(wait=True)
            # ax.clear()
            # ax.bar(range(num_bins), bin_means)
            # ax.set_title(f'Epoch {epoch + 1}')
            # ax.set_xlabel('Bin')
            # ax.set_ylabel('Mean Cost')
            # display(fig)
            # plt.pause(0.01)

            # custom_scheduler.step(loss)
            # current_lr = optimizer_tangents.param_groups[0]['lr']
            # print(f"Epoch {epoch}, Loss: {loss.item()}, Scale: {scale.item()}, Learning Rate: {current_lr}")


            # size = distance_matrix.size()
    # W = torch.ones(size, dtype=torch.bool)
    # # Set the diagonal elements to False
    # W.fill_diagonal_(False)

    # if r is not None:
    #     matrix = torch.pow(scale*distance_matrix , -r)
    # else:
    #     matrix = torch.ones_like(distance_matrix)

    # print(epoch,torch.max(matrix[W]).item(), torch.min(matrix[W]).item())
    # cost = torch.norm( torch.mul(distance_matrix_paramterized[W] - scale*distance_matrix[W], matrix[W]),p='fro')**2 /torch.norm( matrix[W],p='fro')**2


    # window = (0.35)*len(masks) # value = 1/200 at 50 (out of 200) exponential model
    # threshold = 1/len(masks)    
    # p = np.exp(np.log(threshold)/window)
    # new_weights = []

    # means = []
    # lrs = []
    # for i in range(len(masks)):
        # W = masks[i]
        # means.append(torch.mean(distance_matrix[W]).item())
        # if i == 0:
        #     lrs.append(1)
        # else:
        #     lrs.append(means[i]/means[0])
    