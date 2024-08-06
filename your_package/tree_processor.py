import os
import copy
import torch
import pickle
import logging
import numpy as np
import treeswift as ts
from collections.abc import Collection
from datetime import datetime

# Assume spaceform_pca_lib and embedding are part of your_package
from your_package import spaceform_pca_lib, embedding
from typing import Union, Set, Optional, List, Callable, Tuple, Any
from .config import TOTAL_EPOCHS,LEARNING_RATE_INIT, MAX_DIAM, DEFAULT_MODEL, EMBEDDING_MODE,LOG_DIR, RESULTS_DIR,IMAGE_DIR,MOVIE_DIR, EPS,MAX_HEATPLOT, MIN_HEATPLOT,MOVIE_LENGTH,SUBSPACE_DIR
import matplotlib.patches as patches

 

from memory_profiler import profile

import imageio

import matplotlib.pyplot as plt
import seaborn as sns
# sns.reset_defaults()

import gc
import sys
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
class Tree:
    def __init__(self, *source, enable_logging: bool = False):
        """
        Initialize a Tree object. Load from a source, which can be:
         -- a single string, giving the file path; name is set to base name by default
         -- two inputs: a string (name) and a treeswift tree object (contents)
         -- enable_logging: if True, logging is enabled
        """
        self.logger = None
        if enable_logging:
            self.setup_logging()
        if len(source) == 1 and isinstance(source[0], str):
            file_path = source[0]
            self.name = os.path.basename(file_path)
            self.contents = self._load_tree(file_path)
            self.log_info(f"Tree initialized from file: {file_path}")
        elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], ts.Tree):
            self.name = source[0]
            self.contents = source[1]
            self.log_info(f"Tree initialized with name: {self.name}")
        else:
            raise ValueError("Provide either a single file path as a string or two inputs: a string (name) and a treeswift tree object (contents).")
        self.metadata = {}

    def setup_logging(self, log_dir: str = "log", log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s'):
        """
        Set up logging configuration.
        """
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"Tree_{current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
        self.log_info("Logging setup complete.")

    def log_info(self, message: str):
        """
        Log an informational message.
        """
        if self.logger:
            self.logger.info(message)

    @classmethod
    def from_contents(cls, name: str, contents: ts.Tree, enable_logging: bool = False) -> 'Tree':
        """
        Create a Tree object from given contents.
        """
        cls_instance = cls(name, contents, enable_logging=enable_logging)
        cls_instance.log_info(f"Tree created from contents with name: {name}")
        return cls_instance

    def copy(self) -> 'Tree':
        """
        Create a deep copy of the Tree object.
        """
        
        tree_copy = copy.deepcopy(self)
        self.log_info(f"Tree '{self.name}' copied successfully.")
        return tree_copy


    def _load_tree(self, file_path: str) -> ts.Tree:
        """
        Load a tree from a Newick file.
        """
        self.log_info(f"Loading tree from file: {file_path}")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist")
            return ts.read_tree_newick(file_path)
        except Exception as e:
            self.log_info(f"Failed to load tree from {file_path}: {e}")
            raise

    def save_tree(self, file_path: str, format: str = 'newick') -> None:
        """
        Save the tree to a file in the specified format.
        """
        try:
            if format == 'newick':
                self.contents.write_tree_newick(file_path)
                self.log_info(f"Tree '{self.name}' saved successfully.")
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self.log_info(f"Failed to save tree '{self.name}': {e}")
            raise

    def __repr__(self) -> str:
        """
        Return a string representation of the Tree object.
        """
        repr_str = f"Tree({self.name})"
        self.log_info(f"Representation: {repr_str}")
        return repr_str

    def terminal_names(self) -> List[str]:
        """
        Get the list of terminal (leaf) names in the tree.
        """
        
        terminal_names = list(self.contents.labels(leaves=True, internal=False))
        self.log_info(f"Terminal names for tree '{self.name}' retrieved.")
        return terminal_names
        
    def distance_matrix(self) -> np.ndarray:
        """
        Compute and return the distance matrix for the tree.
        """
        
        labels = self.terminal_names()
        distance_dict = self.contents.distance_matrix(leaf_labels=True)
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        n = len(labels)
        distance_matrix = np.zeros((n, n))

        for label1, row in distance_dict.items():
            i = label_to_index[label1]
            for label2, distance in row.items():
                j = label_to_index[label2]
                distance_matrix[i, j] = distance
        self.log_info(f"Distance matrix computed for tree '{self.name}'.")
        return distance_matrix
        

    def diameter(self) -> float:
        """
        Compute and return the diameter of the tree.
        """
        diameter = self.contents.diameter()
        self.log_info(f"Diameter of tree '{self.name}': {diameter}")
        return diameter
        

    def normalize(self) -> None:
        """
        Normalize the branch lengths of the tree so that the diameter is 1.
        """
        self.log_info(f"Normalizing tree '{self.name}' with current diameter.")
        diameter = self.diameter()
        if not np.isclose(diameter, 0.0):
            scale_factor = 1.0 / diameter
            for node in self.contents.traverse_postorder():
                if node.get_edge_length() is not None:
                    node.set_edge_length(node.get_edge_length() * scale_factor)
            self.log_info(f"Tree '{self.name}' normalized with scale factor {scale_factor}.")
        else:
            self.log_info(f"Tree '{self.name}' has a diameter of zero and cannot be normalized.")

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
class MultiTree:
    def __init__(self, *source: Union[str, Set['Tree']], enable_logging: bool = False):
        """
        Initialize a MultiTree object. Load trees from a source, which can be:
         -- a single string, giving the file path; name is set to base name by default
         -- a set of ts.Tree objects and a string (name)

        Parameters:
        enable_logging (bool): Flag to enable or disable logging.
        """
        self.logger = None
        if enable_logging:
            self.setup_logging()

        if len(source) == 1 and isinstance(source[0], str):
            file_path = source[0]
            self.name = os.path.basename(file_path)
            self.trees = self._load_trees(file_path)
            self.log_info(f"Initialized MultiTree with trees from file: {file_path}")
        elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], Set):
            self.name = source[0]
            self.trees = [Tree(tree.name, tree.contents) for tree in source[1]]
            self.log_info(f"Initialized MultiTree with trees from provided set. Name: {self.name}")
        else:
            raise ValueError("Provide either a single file path as a string or two inputs: a string (name) and a set of ts.Tree objects (contents).")
        self.metadata = {}

    def setup_logging(self, log_dir: str = "log", log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s'):
        """
        Set up logging configuration.
        """
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"MultiTree_{current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)

    def log_info(self, message: str):
        """
        Log an informational message.
        """
        if self.logger:
            self.logger.info(message)

    def copy(self) -> 'MultiTree':
        """
        Create a deep copy of the MultiTree object.
        """
        multitree_copy = copy.deepcopy(self)
        self.log_info(f"MultiTree '{self.name}' copied.")
        return multitree_copy

    def _load_trees(self, file_path: str) -> Collection['Tree']:
        """
        Load trees from a Newick file and return a list of Tree objects.
        """
        if not os.path.exists(file_path):
            self.log_info(f"The file {file_path} does not exist")
            raise FileNotFoundError(f"The file {file_path} does not exist")
        
        self.log_info(f"Loading multitree from file: {file_path}")
        tree_list = []
              
        try:
            for idx, tree in enumerate(ts.read_tree_newick(file_path)):
                tree_list.append(Tree(f'tree_{idx+1}', tree))
            self.log_info(f"Loaded {len(tree_list)} trees from {file_path}.")
        except Exception as e:
            self.log_info(f"Failed to load trees from {file_path}: {e}")
            raise ValueError(f"Failed to load trees from {file_path}: {e}")
        return tree_list

    def save_trees(self, file_path: str, format: str = 'newick') -> None:
        """
        Save all trees to a file in the specified format.
        """
        if format == 'newick':
            try:
                with open(file_path, 'w') as f:
                    for tree in self.trees:
                        f.write(tree.contents.newick() + "\n")
                self.log_info(f"Trees saved to {file_path} in {format} format.")
            except Exception as e:
                self.log_info(f"Failed to save trees to {file_path}: {e}")
                raise
        else:
            self.log_info(f"Unsupported format: {format}")
            raise ValueError(f"Unsupported format: {format}")

    def distance_matrix(self, 
                           func: Callable[[np.ndarray], float] = np.nanmean, 
                           confidence: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute the distance matrix of each individual tree and compute the element-wise aggregate
        (mean, median, max, min) of them. The aggregate function can be specified as an input.
        
        Additionally, return a confidence matrix indicating the ratio of non-NaN values at each element.

        Parameters:
        func (Callable[[np.ndarray], float]): Function to compute the aggregate. Default is np.nanmean.
        confidence (bool): If True, also return the confidence matrix. Default is False.

        Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - The aggregated distance matrix.
            - (Optional) Confidence matrix indicating the ratio of non-NaN values.
        """
        if not self.trees:
            self.log_info("No trees available to compute distance matrices.")
            raise ValueError("No trees available to compute distance matrices")

        distance_dicts = []
        all_labels = set()
        for tree in self.trees:
            labels = tree.contents.labels(leaves=True, internal=False)
            distance_dict = tree.contents.distance_matrix(leaf_labels=True)
            distance_dicts.append(distance_dict)
            all_labels.update(labels)
        
        all_labels = sorted(all_labels)
        label_to_index = {label: idx for idx, label in enumerate(all_labels)}
        n = len(all_labels)
        
        distance_matrices = []
        count_matrices = []
        for distance_dict in distance_dicts:
            distance_matrix = np.full((n, n), np.nan)
            count_matrix = np.zeros((n, n))  # Matrix to count non-NaN entries
            for label1, row in distance_dict.items():
                i = label_to_index[label1]
                for label2, distance in row.items():
                    j = label_to_index[label2]
                    distance_matrix[i, j] = distance
                    count_matrix[i, j] += 1
            np.fill_diagonal(distance_matrix, 0)  # Fill diagonal with 0
            distance_matrices.append(distance_matrix)
            count_matrices.append(count_matrix)
        
        stacked_matrices = np.stack(distance_matrices)
        stacked_counts = np.stack(count_matrices)

        # Apply the aggregation function element-wise, excluding NaNs
        aggregated_matrix = func(stacked_matrices, axis=0)

        # Compute confidence matrix
        confidence_matrix = np.sum(stacked_counts > 0, axis=0) / len(self.trees)
        
        self.log_info("Distance matrix computation complete.")
        if confidence:
            return aggregated_matrix, confidence_matrix
        else:
            return aggregated_matrix

    def search_trees(self, query: str) -> List['Tree']:
        """
        Search for trees by name.
        """
        result = [tree for tree in self.trees if query in tree.name]
        self.log_info(f"Search for trees with query '{query}' returned {len(result)} results.")
        return result

    def add_metadata(self, key: str, value: str) -> None:
        """
        Add metadata to the MultiTree object.
        """
        self.metadata[key] = value
        self.log_info(f"Metadata added: {key} = {value}")

    def __iter__(self) -> Collection['Tree']:
        """
        Return an iterator over the trees.
        """
        self.log_info("Returning an iterator over trees.")
        return iter(self.trees)

    def __len__(self) -> int:
        """
        Return the number of trees.
        """
        length = len(self.trees)
        self.log_info(f"Number of trees: {length}")
        return length

    def __contains__(self, item) -> bool:
        """
        Check if an item is in the collection.
        """
        contains = item in self.trees
        self.log_info(f"Item {'found' if contains else 'not found'} in collection.")
        return contains

    def __repr__(self) -> str:
        """
        Return a string representation of the MultiTree object.
        """
        repr_str = f"MultiTree({self.name}, {len(self.trees)} trees)"
        self.log_info(f"Representation: {repr_str}")
        return repr_str

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
class TreeProcessor:
    def __init__(self, tree):
        """
        Initialize a TreeProcessor object with a tree.
        """
        self.tree = tree
        self.current_time = None
        self.setup_logging()
        self.log_info("TreeProcessor is initialized.")
        self.log_info(f"Tree: {self.tree.name}")

    def __repr__(self):
        """
        Return a string representation of the TreeProcessor object.
        """
        repr_str = f"TreeProcessor({self.tree.name})"
        self.log_info(f"Representation: {repr_str}")
        return repr_str

    def setup_logging(self, log_dir: str = LOG_DIR, log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s'):
        """
        Set up logging configuration.
        """
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_time = current_time
        log_file = os.path.join(log_dir, f"TreeProcessor_{current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
    
    def log_info(self, message: str):
        """
        Log an informational message.
        """
        logging.info(message)


    def hyperbolic_embedding(self, dimension, **kwargs) -> 'HyperbolicSpace':
        """
        Embed the tree in a hyperbolic space using either a naive or precise method.

        Parameters:
        - dimension (int): Dimensionality of the hyperbolic space (required).
        - mode (str): Embedding mode: 'naive' for basic, 'precise' for advanced (default is 'naive').
        - epochs (int): Number of epochs for optimization in precise mode (default is 2000).
        - max_diameter (float): Maximum diameter for scaling the distance matrix (default is 10).
        - learning_rate (Optional[float]): Learning rate for the optimization process (default is None).
        - scale_learning (Optional[callable]): Function for scaling during optimization (default is None).
        - weight_exponent (Optional[float]): Exponent for weights in optimization (default is None).
        - initial_lr (Optional[float]): Initial learning rate for optimization (default is 0.1).

        Returns:
        - HyperbolicSpace: The resulting hyperbolic space embedding of the tree.
        
        Raises:
        - ValueError: If 'dimension' is not provided.
        - RuntimeError: For any errors during the embedding process.
        """
        # Retrieve and validate keyword arguments
        mode = kwargs.get('mode', EMBEDDING_MODE)
        total_epochs = kwargs.get('total_epochs', TOTAL_EPOCHS)
        initial_lr = kwargs.get('initial_lr', LEARNING_RATE_INIT)
        max_diameter = kwargs.get('max_diameter', MAX_DIAM)
        learning_rate = kwargs.get('learning_rate', None)
        scale_learning = kwargs.get('scale_learning', None)
        weight_exponent = kwargs.get('weight_exponent', None)
        enable_movie = kwargs.get('enable_movie', True)

        if dimension is None:
            raise ValueError("The 'dimension' parameter is required.")

        # Constants
        scale_factor = max_diameter / self.tree.diameter()
        initial_curvature = -(scale_factor ** 2)

        try:
            # Scale distance matrix
            distance_matrix = self.tree.distance_matrix() * scale_factor
            N = distance_matrix.shape[0]

            # Naive embedding
            self.log_info("Initiating naive embedding of the tree in hyperbolic space.")
            gramian = -np.cosh(distance_matrix)
            points = spaceform_pca_lib.lgram_to_points(gramian, dimension)
            points = np.array([spaceform_pca_lib.project_vector_to_hyperbolic_space(x) for x in points.T]).T

            # Initialize hyperbolic space
            hyperbolic_space = embedding.HyperbolicSpace(model=DEFAULT_MODEL, curvature=initial_curvature)
            hyperbolic_space.points = points
            self.log_info("Naive hyperbolic embedding completed.")

            # Precise embedding
            if mode != 'naive':
                self.log_info("Initiating precise hyperbolic embedding.")
                initial_tangents = spaceform_pca_lib.hyperbolic_log_torch(torch.tensor(points))
                tangents, scale = spaceform_pca_lib.optimize_embedding(
                                                        torch.tensor(distance_matrix),
                                                        dimension,
                                                        initial_tangents=initial_tangents,
                                                        epochs=total_epochs,
                                                        log_function=self.log_info,
                                                        learning_rate=learning_rate,
                                                        scale_learning=scale_learning,
                                                        weight_exponent=weight_exponent,
                                                        initial_lr=initial_lr,
                                                        enable_save = True,
                                                        time = self.current_time
                                                        )
                # Ensure scale is a numpy array
                scale = scale.detach().numpy() if scale.requires_grad else scale.numpy()
                hyperbolic_space.curvature *= np.abs(scale) ** 2
                hyperbolic_space.points = spaceform_pca_lib.hyperbolic_exponential_torch(tangents)
                self.log_info("Precise hyperbolic embedding completed.")

        except ValueError as ve:
            self.log_info(f"Value error during hyperbolic embedding: {ve}")
            raise
        except RuntimeError as re:
            self.log_info(f"Runtime error during hyperbolic embedding: {re}")
            raise
        except Exception as e:
            self.log_info(f"Unexpected error during hyperbolic embedding: {e}")
            raise
        if enable_movie:
            fps = total_epochs // MOVIE_LENGTH
            self.create_figures()
            self.create_movie(fps = fps)
        
        directory = f'{SUBSPACE_DIR}/{self.current_time}'
        os.makedirs(directory, exist_ok=True)

        filename = 'hyperbolic_space.pkl'
        filepath = f'{directory}/{filename}'

        try:
            with open(filepath, 'wb') as file:
                pickle.dump(hyperbolic_space, file)
            self.log_info(f"Object successfully saved to {filepath}")
        except IOError as ioe:
            self.log_info(f"IO error while saving the object: {ioe}")
            raise
        except pickle.PicklingError as pe:
            self.log_info(f"Pickling error while saving the object: {pe}")
            raise
        except Exception as e:
            self.log_info(f"Unexpected error while saving the object: {e}")
            raise
        return hyperbolic_space

    

    def create_figures(self, output_dir: str = None):
        """
        Create and save figures based on RE matrices and distance matrices.

        Parameters:
        - output_dir (str): Directory to save the output figures. If None, uses a timestamped directory.
        """
        if output_dir is None:
            timestamp = self.current_time
        else:
            timestamp = os.path.basename(output_dir)
            
        output_dir = f'{IMAGE_DIR}/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        path = f'{RESULTS_DIR}/{timestamp}'

        weight_history = -np.load(os.path.join(path, "weight_history.npy"))
        lr_history = np.log10(np.abs(np.load(os.path.join(path, "lr_history.npy"))))
        scale_history = np.load(os.path.join(path, "scale_history.npy"))        

        npy_files = sorted([f for f in os.listdir(path) if f.startswith('RE') and f.endswith('.npy')],
                           key=lambda f: int(f.split('_')[1].split('.')[0]))
        total_epochs = len(npy_files)
        log10_distance_matrix = np.log10(self.tree.distance_matrix() + EPS)
        mask = np.eye(log10_distance_matrix.shape[0], dtype=bool)

        
        rms_values = []
        upper_diag_indices = None
        min_heatplot = float('inf')
        max_heatplot = float('-inf')

        for file_name in npy_files:
            re_matrix = np.load(os.path.join(path, file_name))
            log10_re_matrix = np.log10(re_matrix + EPS)
            np.fill_diagonal(log10_re_matrix, np.nan)
            current_min = np.nanmin(log10_re_matrix)
            current_max = np.nanmax(log10_re_matrix)
            min_heatplot = min(min_heatplot, current_min)
            max_heatplot = max(max_heatplot, current_max)
            if upper_diag_indices is None:
                upper_diag_indices = np.triu_indices_from(re_matrix, k=1)
            upper_diag_elements = re_matrix[upper_diag_indices]
            rms = np.sqrt(np.mean(upper_diag_elements))
            rms_values.append(rms)
        del log10_re_matrix, re_matrix, upper_diag_indices
        gc.collect()

        if rms_values:
            max_rms = max(rms_values) * 1.1
            min_rms = min(rms_values) * 0.9
        else:
            max_rms = 1
            min_rms = 0
        max_lr = max(lr_history)+0.1
        min_lr = min(lr_history)-0.1

        for i, file_name in enumerate(npy_files):
            epoch = int(file_name.split('_')[1].split('.')[0])
            save_path = os.path.join(output_dir, f'heatmap_{epoch}.png')

            log10_re_matrix = np.log10(np.load(os.path.join(path, file_name)) + EPS)

            fig = plt.figure(figsize=(12, 12), tight_layout=True)
            gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 2, 2], width_ratios=[1, 1])

            ax_rms = fig.add_subplot(gs[0, :])
            ax_rms.plot(range(1, len(rms_values[:epoch]) + 1), rms_values[:epoch], marker='o')
            ax_rms.set_xlim(1, total_epochs)
            ax_rms.set_ylim(min_rms, max_rms)
            ax_rms.set_xlabel('Epoch')
            ax_rms.set_ylabel('RMS of RE')
            ax_rms.set_title('Evolution of Relative Errors')

            # Additional plots for weight and learning rate
            ax_weight = fig.add_subplot(gs[1, 0])
            epochs_range = range(1, len(weight_history[:epoch]) + 1)
            weight_color = 'blue'
            highlight_color = 'red'
            highlight_mask = scale_history[:epoch]

            ax_weight.plot(epochs_range, weight_history[:epoch], marker='o', linestyle='', color=weight_color, label='Scale Learning Disabled')
            if highlight_mask is not None:
                highlighted_weights = [weight_history[i] if highlight_mask[i] else np.nan for i in range(len(weight_history[:epoch]))]
                ax_weight.plot(epochs_range, highlighted_weights, marker='o', linestyle='', color=highlight_color, label='Scale Learning Enabled')

            ax_weight.set_xlim(1, total_epochs)
            ax_weight.set_ylim(0, 1)
            ax_weight.set_xlabel('Epoch')
            ax_weight.set_ylabel('-Weight Exponent')
            ax_weight.set_title('Evolution of Weights')
            ax_weight.legend()

            ax_lr = fig.add_subplot(gs[1, 1])
            ax_lr.plot(range(1, len(lr_history[:epoch]) + 1), lr_history[:epoch], marker='o')
            ax_lr.set_xlim(1, total_epochs)
            ax_lr.set_ylim(min_lr, max_lr)
            ax_lr.set_xlabel('Epoch')
            ax_lr.set_ylabel('log10(Learning Rate)')
            ax_lr.set_title('Evolution of Learning Rates')

            ax1 = fig.add_subplot(gs[2:, 0])
            ax2 = fig.add_subplot(gs[2:, 1])

            sns.heatmap(log10_re_matrix, mask=mask, ax=ax1, cmap='viridis', cbar_kws={'label': 'log10(RE)'}, vmin=min_heatplot, vmax=max_heatplot, square=True, xticklabels=False, yticklabels=False)
            ax1.set_title(f'Relative Error (RE) Matrix (Epoch {epoch})')
            cbar = ax1.collections[0].colorbar
            cbar.set_ticks(cbar.get_ticks())
            cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])

            sns.heatmap(log10_distance_matrix, mask=mask, ax=ax2, cmap='viridis', cbar_kws={'label': 'log10(Distance)'}, square=True, xticklabels=False, yticklabels=False)
            ax2.set_title('Distance Matrix')
            cbar = ax2.collections[0].colorbar
            cbar.set_ticks(cbar.get_ticks())
            cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])

            # Add thin black frame around heatmaps
            for ax in [ax1, ax2]:
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='black', fill=False, linewidth=2)
                ax.add_patch(rect)

            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close('all')

            # Free memory explicitly
            del log10_re_matrix, fig, gs, ax_rms, ax_weight, ax_lr, ax1, ax2
            gc.collect()

            self.log_info(f"Figure saved: {save_path}")

        # Explicitly delete other large arrays
        del weight_history, lr_history, scale_history, log10_distance_matrix, mask, rms_values
        gc.collect()


    def create_movie(self, output_dir: str = None, fps: int = 10):
        """
        Create a movie from the saved heatmaps.

        Parameters:
        - output_dir (str): Directory to save the output movie. If None, uses a timestamped directory.
        - fps (int): Frames per second for the movie.
        """
        if output_dir is None:
            timestamp = self.current_time    
        else:
            timestamp = os.path.basename(output_dir)
        output_dir = f'{MOVIE_DIR}/{timestamp}'
        os.makedirs(output_dir, exist_ok=True)

        image_dir = f'{IMAGE_DIR}/{timestamp}'
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        # Custom sorting function to sort filenames numerically based on the heatmap number
        def extract_number(filename):
            base_name = filename.split('_')[-1].replace('.png', '')
            return int(base_name)

        image_files.sort(key=extract_number)  # Ensure files are in the correct numerical order
        
        total_epoch = len(image_files)
        # Construct full file paths for images
        image_files = [os.path.join(image_dir, f) for f in image_files]

        movie_path = os.path.join(output_dir, 're_vs_distance_evolution.mp4')
        try:
            with imageio.get_writer(movie_path, fps=fps) as writer:
                for image_file in image_files:
                    image = imageio.imread(image_file)
                    writer.append_data(image)

            self.log_info(f"Movie created: {movie_path}")
        except Exception as e:
            self.log_info(f"Error creating movie: {e}")
            raise