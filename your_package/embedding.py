import numpy as np
import torch
import pickle
import logging
import os
import copy
from typing import Optional, Union, List
from datetime import datetime
from your_package import config
from torch.optim import Adam

class Embedding:
    """
    A class representing an abstract embedding.

    Attributes:
        geometry (str): The geometry of the space (e.g., 'euclidean', 'hyperbolic').
        points (torch.Tensor): A PyTorch tensor representing the points in the space.
        labels (list): A list of labels corresponding to the points in the space.
        logger (logging.Logger): A logger for the class if logging is enabled.
    """
    
    def __init__(self, 
                 geometry: Optional[str] = 'hyperbolic', 
                 points: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 labels: Optional[List[Union[str, int]]] = None,
                 enable_logging: bool = False):
        """
        Initializes the Embedding.

        Args:
            geometry (str): The geometry of the space. Default is 'hyperbolic'.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.
        """
        self.logger = None
        if enable_logging:
            self.setup_logging()

        if geometry not in {'euclidean', 'hyperbolic'}:
            if self.logger:
                self.logger.error("Invalid geometry type: %s", geometry)
            raise ValueError("Invalid geometry type. Choose either 'euclidean' or 'hyperbolic'.")
        
        self._geometry = geometry
        self._points = self._convert_points(points) if points is not None else torch.empty((0, 0))
        self._labels = labels if labels is not None else list(range(self._points.shape[0]))
        self.log_info(f"Initialized Embedding with geometry={self._geometry}")

    def setup_logging(self, log_dir: str = "log", log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Set up logging configuration.

        Args:
            log_dir (str): Directory where log files will be saved. Default is 'log'.
            log_level (int): Logging level. Default is logging.INFO.
            log_format (str): Format for logging messages. Default is '%(asctime)s - %(levelname)s - %(message)s'.
        """
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"Embedding_{current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
        self.log_info("Logging setup complete.")

    def log_info(self, message: str) -> None:
        """
        Log an informational message.

        Args:
            message (str): The message to log.
        """
        if self.logger:
            self.logger.info(message)

    @property
    def geometry(self) -> str:
        """Gets the geometry of the space."""
        return self._geometry   
    
    @property
    def points(self) -> torch.Tensor:
        """
        Gets the points in the space.

        Returns:
            torch.Tensor: The points in the Loid space.
        """
        return self._points

    @points.setter
    def points(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Sets the points in the space and checks norm constraints.

        Args:
            value (Union[np.ndarray, torch.Tensor]): The new points to set.

        Raises:
            ValueError: If the norm constraints are violated by the new points.
        """
        self._points = self._convert_points(value)
        self._update_dimensions()
        self._validate_norms() 
        self.log_info(f"Updated points with shape={self._points.shape}")

    @property
    def labels(self) -> List[Union[str, int]]:
        """Gets the labels corresponding to the points."""
        return self._labels
    
    @labels.setter
    def labels(self, value: List[Union[str, int]]) -> None:
        """Sets the labels corresponding to the points."""
        if len(value) != self._points.shape[0]:
            if self.logger:
                self.logger.error("The number of labels must match the number of points, got: %d labels for %d points", len(value), self._points.shape[0])
            raise ValueError("The number of labels must match the number of points")
        self._labels = value
        self.log_info("Updated labels with length=%d", len(self._labels))

    def _update_dimensions(self) -> None:
        """Updates the dimension based on the points. Must be implemented by a subclass."""
        raise NotImplementedError("update_dimensions must be implemented by a subclass")

    def _validate_norms(self) -> None:
        """Validates that all points are within the norm constraints for the Loid model."""
        NotImplementedError("_validate_norms must be implemented by a subclass")
    
    def _convert_points(self, value: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Converts the points to a PyTorch tensor.

        Args:
            value (Union[np.ndarray, torch.Tensor]): The points to convert.

        Returns:
            torch.Tensor: The converted points.
        """
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float32, non_blocking=True)
        else:
            if self.logger:
                self.logger.error("Points must be a NumPy array or PyTorch tensor, got: %s", type(value))
            raise TypeError("Points must be a NumPy array or PyTorch tensor")

    def save(self, filename: str) -> None:
        """
        Saves the Embedding instance to a file using pickle.

        Args:
            filename (str): The file to save the instance to.
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
                self.log_info(f"Saved Embedding to {filename}")
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to save Embedding: %s", e)
            raise
    
    def copy(self) -> 'Embedding':
        """
        Create a deep copy of the Embedding object.
        """
        
        Embedding_copy = copy.deepcopy(self)
        self.log_info(f"Embedding copied successfully.")
        return Embedding_copy

    @staticmethod
    def load(filename: str) -> 'Embedding':
        """
        Loads an Embedding instance from a file using pickle.

        Args:
            filename (str): The file to load the instance from.

        Returns:
            Embedding: The loaded Embedding instance.
        """
        try:
            with open(filename, 'rb') as file:
                instance = pickle.load(file)
            if instance.logger:
                instance.log_info(f"Loaded Embedding from {filename}")
            return instance
        except Exception as e:
            if instance.logger:
                instance.logger.error("Failed to load Embedding: %s", e)
            raise

    def __repr__(self) -> str:
        """Returns a string representation of the Embedding."""
        return (f"Embedding(geometry={self._geometry}, points_shape={self._points.shape}, labels_length={len(self._labels)})")
####################################################################################################
####################################################################################################
####################################################################################################
class HyperbolicEmbedding(Embedding):
    """
    A class representing a hyperbolic embedding.

    Attributes:
        curvature (torch.Tensor): The curvature of the hyperbolic space.
        model (str): The model used ('poincare' or 'loid').
        norm_constraint (tuple): Constraints for point norms in the hyperbolic space.
    """

    def __init__(
        self,
        curvature: Optional[float] = -1,
        model: Optional[str] = 'poincare',
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[Union[str, int]]] = None,
        enable_logging: bool = False
    ):
        """
        Initializes the HyperbolicEmbedding.

        Args:
            curvature (Optional[float]): The curvature of the space. Must be negative. Default is -1.
            model (Optional[str]): The model of the space ('poincare' or 'loid'). Default is 'poincare'.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.

        Raises:
            ValueError: If the curvature is not negative.
        """
        if curvature >= 0:
            raise ValueError("Curvature must be negative for hyperbolic space.")

        super().__init__(geometry='hyperbolic', points=points, labels=labels, enable_logging=enable_logging)
        self._curvature = torch.tensor(curvature, dtype=torch.float32) if not isinstance(curvature, torch.Tensor) else curvature.to(dtype=torch.float32, non_blocking=True)
        self.model = model
        self.log_info(f"Initialized HyperbolicEmbedding with curvature={self._curvature} and model={self.model}")

    @property
    def curvature(self) -> torch.Tensor:
        """Gets the curvature of the space."""
        return self._curvature

    @curvature.setter
    def curvature(self, value: float) -> None:
        """Sets the curvature of the space.

        Args:
            value (float): The new curvature value.
        """
        self._curvature = torch.tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value.to(dtype=torch.float32, non_blocking=True)
        self.log_info(f"Updated curvature to {self._curvature}")

    def switch_model(self) -> 'HyperbolicEmbedding':
        """
        Switches between Poincare and Loid models.

        Returns:
            HyperbolicEmbedding: A new instance of HyperbolicEmbedding with the switched model.

        Raises:
            ValueError: If there are no points to switch model.
        """
        if self._points.numel() == 0:
            raise ValueError("No points to switch model.")

        self.log_info(f"Switching model from {self.model}")

        if self.model == 'poincare':
            norm_points = torch.norm(self._points, dim=0)
            new_points = torch.zeros((self._points.shape[0] + 1, self._points.shape[1]))
            new_points[0] = (1 + norm_points**2) / (1 - norm_points**2)
            new_points[1:] = (2 * self._points) / (1 - norm_points**2)
            new_space = LoidEmbedding(curvature=self._curvature, points=new_points, enable_logging=self.logger is not None)
            self.log_info("Switched to LoidEmbedding model.")
            return new_space
        elif self.model == 'loid':
            x1 = self._points[0]
            bar_x = self._points[1:]
            new_points = bar_x / (x1 + 1)
            new_space = PoincareEmbedding(curvature=self._curvature, points=new_points, enable_logging=self.logger is not None)
            self.log_info("Switched to PoincareEmbedding model.")
            return new_space
        else:
            if self.logger:
                self.logger.error("Unknown model type: %s", self.model)
            raise ValueError("Unknown model type.")

    def poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Poincare distance between points x and y.

        Args:
            x (torch.Tensor): Point(s) in the Poincare ball.
            y (torch.Tensor): Point(s) in the Poincare ball.

        Returns:
            torch.Tensor: Poincare distance between x and y.
        """
        norm_x = torch.sum(x**2, dim=0, keepdim=True)
        norm_y = torch.sum(y**2, dim=0, keepdim=True)
        diff_norm = torch.sum((x - y)**2, dim=0, keepdim=True)
        denominator = (1 - norm_x) * (1 - norm_y)
        distance = torch.acosh(1 + 2 * diff_norm / denominator)
        return distance

    
    def to_poincare(self, vectors: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transforms vectors from Loid to Poincare model.

        Args:
            vectors (Union[np.ndarray, torch.Tensor]): Input vectors (columns are vectors, or a single vector).

        Returns:
            torch.Tensor: Transformed vectors in Poincare model.

        Raises:
            TypeError: If input vectors are not a NumPy array or a PyTorch tensor.
        """
        if isinstance(vectors, np.ndarray):
            vectors = torch.tensor(vectors, dtype=torch.float32)
        elif isinstance(vectors, torch.Tensor):
            vectors = vectors.to(dtype=torch.float32, non_blocking=True)
        else:
            raise TypeError("Input vectors must be a NumPy array or a PyTorch tensor.")

        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(1)
        new_points = vectors[1:, :] / (1 + vectors[0, :])
        return new_points

    def to_loid(self, vectors: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transforms vectors from Poincare to Loid model.

        Args:
            vectors (Union[np.ndarray, torch.Tensor]): Input vectors (columns are vectors, or a single vector).

        Returns:
            torch.Tensor: Transformed vectors in Loid model.

        Raises:
            TypeError: If input vectors are not a NumPy array or a PyTorch tensor.
        """
        if isinstance(vectors, np.ndarray):
            vectors = torch.tensor(vectors, dtype=torch.float32)
        elif isinstance(vectors, torch.Tensor):
            vectors = vectors.to(dtype=torch.float32, non_blocking=True)
        else:
            raise TypeError("Input vectors must be a NumPy array or a PyTorch tensor.")

        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(1)

        norm_points = torch.norm(vectors, dim=0)
        new_points = torch.zeros((vectors.shape[0] + 1, vectors.shape[1]))
        new_points[0] = (1 + norm_points**2) / (1 - norm_points**2)
        new_points[1:] = (2 * vectors) / (1 - norm_points**2)        
        return new_points

    def matrix_sqrtm(self,A):
        """
        Computes the matrix square root of a positive definite matrix, with eigenvalue thresholding.

        Args:
            A (torch.Tensor): The positive definite matrix.

        Returns:
            torch.Tensor: The matrix square root of A.
        """
        eigvals, eigvecs = torch.linalg.eigh(A)
        # Threshold negative eigenvalues to zero
        eigvals = torch.clamp(eigvals, min=0)
        sqrt_eigvals = torch.diag(torch.sqrt(eigvals))
        return eigvecs @ sqrt_eigvals @ eigvecs.T

    def __repr__(self) -> str:
        """Returns a string representation of the HyperbolicEmbedding."""
        return (f"HyperbolicEmbedding(curvature={self._curvature.item()}, model={self.model}, points_shape={self._points.shape})")
####################################################################################################
####################################################################################################
####################################################################################################
class PoincareEmbedding(HyperbolicEmbedding):
    """
    A class representing a Poincare hyperbolic embedding space.

    Inherits from:
        HyperbolicEmbedding

    Attributes:
        norm_constraint (tuple): Constraints for point norms in the Poincare model.
    """

    def __init__(self, 
                 curvature: Optional[float] = -1, 
                 points: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 labels: Optional[List[Union[str, int]]] = None,
                 enable_logging: bool = False) -> None:
        """
        Initializes the PoincareEmbedding.

        Args:
            curvature (Optional[float]): The curvature of the space. Must be negative.
            points (Optional[Union[np.ndarray, torch.Tensor]]): A NumPy array or PyTorch tensor of points. Default is None.
            labels (Optional[List[Union[str, int]]]): A list of labels corresponding to the points. Default is None.
            enable_logging (bool): If True, logging is enabled. Default is False.
        """
        super().__init__(curvature=curvature, points=points, labels=labels, enable_logging=enable_logging)
        self.norm_constraint = config.poincare_domain
        self._update_dimensions()
        self._validate_norms()
        self.log_info(f"Initialized PoincareEmbedding with curvature={self.curvature} and checked point norms")
        
    
    def _validate_norms(self) -> None:
        """Validates that all points are within the norm constraints for the Poincare model."""
        norm2 = self.norm2()
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if self.logger:
                self.logger.error(f"Points norm constraint violated: norms={norm2}, constraint=({min_norm}, {max_norm})")
            raise ValueError(f"Points norm constraint violated: norms must be in range ({min_norm}, {max_norm})")

    def _update_dimensions(self) -> None:
        """Updates the dimension and number of points based on the Poincare model."""
        self.dimension = self._points.size(0) if self._points.numel() > 0 else 0
        self.n_points = self._points.size(1) if self._points.numel() > 0 else 0
        self.log_info(f"Updated dimensions to {self.dimension}")
        self.log_info(f"Updated n_points to {self.n_points}")

    def norm2(self) -> torch.Tensor:
        """Computes the squared L2 norm of the points."""
        norm2 = torch.norm(self._points, dim=0)**2
        self.log_info(f"Computed squared L2 norms: {norm2}")
        return norm2

    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the Poincare model.

        Returns:
            torch.Tensor: The distance matrix.
        """
        G = torch.matmul(self._points.T, self._points)
        diag_vec = torch.diag(G)
        
        EDM = -2 * G + diag_vec.view(1, -1) + diag_vec.view(-1, 1)
        EDM = torch.relu(EDM)
        EDM = EDM / (1 - diag_vec.view(-1, 1))
        EDM = EDM / (1 - diag_vec.view(1, -1))
        distance_matrix = (1 / torch.sqrt(torch.abs(self.curvature))) * torch.arccosh(1 + 2 * EDM)        
        self.log_info(f"Computed distance matrix with shape distance_matrix.shape")

        return distance_matrix

    
    def centroid(self, 
                 mode: str = 'default', 
                 lr: float = config.frechet_lr, 
                 max_iter: int = config.frechet_max_iter, 
                 tol: float = config.frechet_tol) -> torch.Tensor:
        """
        Compute the centroid of the points in the Poincare space.
        
        Args:
            mode (str): The mode to compute the centroid. 
            lr (float): Learning rate for the optimizer. Default is config.frechet_lr.
            max_iter (int): Maximum number of iterations. Default is config.frechet_max_iter.
            tol (float): Tolerance for stopping criterion. Default is config.frechet_tol.
        
        Returns:
            torch.Tensor: The centroid of the points.
        """
        if mode == 'default':
            X = self.to_loid(self.points)
            centroid = X.mean(dim=1, keepdim=True)
            norm2 = -centroid[0]**2 + torch.sum(centroid[1:]**2, dim=0)
            
            centroid = 1 / torch.sqrt(-norm2) * centroid
            centroid[0] = torch.sqrt(1 + torch.sum(centroid[1:]**2))
            centroid = centroid[1:] / (centroid[0] + 1)
            return centroid

        elif mode == 'Frechet':
            N = self.points.shape[1]
            centroid = self.points.mean(dim=1, keepdim=True).clone().detach().requires_grad_(True)
            optimizer = Adam([centroid], lr=lr)

            for _ in range(max_iter):
                optimizer.zero_grad()
                distances = self.poincare_distance(self.points, centroid)
                loss = torch.sum(distances**2)
                loss.backward(retain_graph=True)  # Retain graph for multiple backward passes if needed
                optimizer.step()

                if torch.norm(centroid).item() >= 1:
                    centroid.data = centroid.data / torch.norm(centroid).item() * config.norm_projection

                if torch.norm(centroid.grad) < tol:
                    break
            return centroid.detach()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def translate(self, vector: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """
        Translates the points by a given vector in the Poincare model.

        Args:
            vector (Optional[Union[np.ndarray, torch.Tensor]]): The translation vector.
        """
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector, dtype=torch.float32)
        
        vector = vector.view(-1, 1)
        norm2 = torch.norm(vector, dim=0)**2
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if self.logger:
                self.logger.error("In Poincare model, the L2 norm of the points must be strictly less than 1. Invalid norms: %s", norm2)
            raise ValueError("In Poincare model, the L2 norm of the points must be strictly less than 1.")

        self.log_info(f"Translating points by vector with shape {vector.shape}")
        self._points = self._add(vector, self._points)
        self.log_info(f"Points translated. New points shape: {self._points.shape}")

    def _add(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Adds a vector to the points in the Poincare model.

        Args:
            b (torch.Tensor): The translation vector.
            x (torch.Tensor): The points.

        Returns:
            torch.Tensor: The translated points.
        """
        b = b.view(-1, 1)

        if x.shape[0] != b.shape[0]:
            if self.logger:
                self.logger.error("Dimension mismatch between points (%s) and vector (%s)", x.shape, b.shape)
            raise ValueError("Dimension mismatch between points and vector")  

        norm_x_sq = torch.sum(x ** 2, dim=0, keepdim=True)
        norm_b_sq = torch.sum(b**2, dim=0, keepdim=True)
        dot_product = 2 * torch.matmul(x.T, b).view(1, -1)

        denominator = (1 + dot_product + norm_x_sq * norm_b_sq).view(1, -1)
        numerator_x = x * (1 - norm_b_sq).view(1, -1)
        numerator_b = b * (1 + dot_product + norm_x_sq).view(1, -1)
        numerator = numerator_x + numerator_b

        result = numerator / denominator
        return result

    def rotate(self, R: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Rotates the points by a given matrix in the Poincare model.

        Args:
            R (Union[np.ndarray, torch.Tensor]): The rotation matrix.
        """
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float32)
        
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        if not torch.allclose(R.T @ R, I, atol=config.atol):
            self.log_info("The provided matrix is not a valid rotation matrix. Attempting to orthogonalize.")
            R = R @ torch.linalg.inv(self.matrix_sqrtm(R.T @ R))
        
        self.log_info(f"Rotating points with matrix shape {R.shape}")
        self._points = R @ self._points
        self.log_info(f"Points rotated. New points shape: {self._points.shape}")

    def center(self,mode = 'default') -> None:
        """Centers the points by translating them to the centroid."""
        centroid = self.centroid(mode = mode)
        self.log_info(f"Centroid computed: {centroid}")
        self.translate(-centroid)
        self.log_info(f"Points centered. New points shape: {self._points.shape}")
####################################################################################################
####################################################################################################
####################################################################################################    
class LoidEmbedding(HyperbolicEmbedding):
    """
    A class representing the Loid model in hyperbolic space.

    Inherits from:
        HyperbolicEmbedding

    Attributes:
        curvature (float): The curvature of the hyperbolic space.
        points (torch.Tensor): The points in the Loid space.
        labels (List[Union[str, int]]): Optional labels for the points.
        enable_logging (bool): Whether to enable logging.
    """

    def __init__(
        self,
        curvature: Optional[float] = -1,
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[Union[str, int]]] = None,
        enable_logging: Optional[bool] = False
    ) -> None:
        """
        Initializes the LoidEmbedding with the given parameters.

        Args:
            curvature (float, optional): The curvature of the space. Defaults to -1.
            points (Union[np.ndarray, torch.Tensor], optional): Initial points in the space. Defaults to None.
            labels (List[Union[str, int]], optional): Labels for the points. Defaults to None.
            enable_logging (bool, optional): Whether to enable logging. Defaults to False.
        """
        super().__init__(curvature=curvature, points=points, labels=labels, enable_logging=enable_logging)
        self.model = 'loid'
        self._update_dimensions()
        self.norm_constraint = config.loid_domain
        self._validate_norms()
        self.log_info(f"Initialized LoidEmbedding with curvature={self.curvature} and checked point norms")

    def _validate_norms(self) -> None:
        """
        Validates that all points are within the norm constraints for the Loid model.

        Raises:
            ValueError: If any point's norm is outside the specified constraint range.
        """
        norm2 = self.norm2()
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 <= min_norm) or torch.any(norm2 > max_norm):
            if self.logger:
                self.logger.error(f"Points norm constraint violated: norms={norm2}, constraint=({min_norm}, {max_norm})")
            raise ValueError(f"Points norm constraint violated: norms must be in range ({min_norm}, {max_norm})")

    def _update_dimensions(self) -> None:
        """
        Updates the dimensions of the space based on the current points.
        """
        self.dimension = self._points.size(0) - 1 if self._points.numel() > 0 else 0
        self.n_points = self._points.size(1) if self._points.numel() > 0 else 0
        self.log_info(f"Updated dimensions to {self.dimension}")
        self.log_info(f"Updated n_points to {self.n_points}")

    def norm2(self) -> torch.Tensor:
        """
        Computes the Lorentzian norm squared of the points.

        Returns:
            torch.Tensor: The squared norms of the points.
        """
        if len(self._points) != 0:
            x_0 = self._points[0]
            norm2 = -x_0**2 + torch.sum(self._points[1:]**2, dim=0)
            self.log_info(f"Computed Lorentzian norms: {norm2}")
            return norm2
        else: 
            return torch.tensor([])
    
    def distance_matrix(self) -> torch.Tensor:
        """
        Computes the distance matrix for points in the Loid model.

        Returns:
            torch.Tensor: The distance matrix of shape (n_points, n_points).
        """
        dimension = self.dimension
        J = torch.eye(dimension+1)
        J[0,0] = -1

        G = torch.matmul(torch.matmul((self._points).T,J), self._points)
        G = torch.where(G > -1, torch.tensor(-1.0, dtype=G.dtype, device=G.device), G)
        distance_matrix = (1/torch.sqrt(torch.abs(self.curvature))) * torch.arccosh(-G)        
        self.log_info(f"Computed distance matrix with shape {distance_matrix.shape}")

        return distance_matrix

    def rotate(self, R: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Rotates the points by a given matrix in the Loid model.

        Args:
            R (Union[np.ndarray, torch.Tensor]): The rotation matrix.

        Raises:
            ValueError: If R is not a valid rotation matrix.
        """
        if isinstance(R, np.ndarray):
            R = torch.tensor(R, dtype=torch.float32)
        # Check if R is a rotation matrix (orthogonal for real matrices)
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        cond1 = not torch.allclose(R.T @ R, I, atol=config.atol)
        cond2 = not torch.isclose(R[0, 0], torch.tensor(1.0, dtype=R.dtype), atol=config.atol)
        cond3 = not torch.allclose(R[0, 1:], torch.zeros_like(R[0, 1:]), atol=config.atol)
        cond4 = not torch.allclose(R[1:, 0], torch.zeros_like(R[1:, 0]), atol=config.atol)
        if cond1 or cond2 or cond3 or cond4:
            self.log_info("The provided matrix is not a valid rotation matrix.")
            raise ValueError("The provided matrix is not a valid rotation matrix.")

        self.log_info(f"Rotating points with matrix of shape {R.shape}")
        self.points = R @ self.points
        self.log_info(f"Points rotated. New points shape: {self._points.shape}")

    def translate(self, vector: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        """
        Translates the points by a given vector in the Loid model.

        Args:
            vector (Optional[Union[np.ndarray, torch.Tensor]]): The translation vector.

        Raises:
            ValueError: If the J-norm of the vector is not exactly -1.
        """
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector, dtype=torch.float32)
        
        vector = vector.view(-1,1)
        norm2 = -vector[0]**2 + torch.sum(vector[1:]**2)
        min_norm, max_norm = self.norm_constraint
        if torch.any(norm2 < min_norm) or torch.any(norm2 >= max_norm):
            if self.logger:
                self.logger.error("In Loid model, the J-norm of the points must be exactly equal to -1. Invalid norms: %s", norm2)
            raise ValueError("In Loid model, the J-norm of the points must be exactly equal to -1.")

        
        self.log_info(f"Translating points by vector with shape {vector.shape}")
        self._points = self._add(vector,self._points)
        self.log_info(f"Points translated. New points shape: {self._points.shape}")


    def _add(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Adds a vector to the points in the Loid model.

        Args:
            b (torch.Tensor): The vector to add.
            x (torch.Tensor): The current points.

        Returns:
            torch.Tensor: The updated points after addition.

        Raises:
            ValueError: If the hyperbolic norm of the vector is not -1 or if dimensions mismatch.
        """
        # Ensure b is a column vector
        b = b.view(-1, 1)
        
        # Calculate the hyperbolic norm
        hyperbolic_norm = -b[0]**2 + torch.sum(b[1:]**2)
        if not torch.isclose(hyperbolic_norm, torch.tensor(-1.0, dtype=b.dtype, device=b.device)):
            if self.logger:
                self.logger.error("The hyperbolic norm of the vector must be -1")
            raise ValueError("The hyperbolic norm of the vector must be -1")

        if x.shape[0] != b.shape[0]:
            if self.logger:
                self.logger.error("Dimension mismatch between points (%s) and vector (%s)", x.shape, b.shape)
            raise ValueError("Dimension mismatch between points and vector")

        D = b.shape[0] - 1
        b_ = b[1:]
        norm_b = torch.norm(b_)
        I = torch.eye(D, device=b.device, dtype=b.dtype)
        
        Rb = torch.zeros((D + 1, D + 1), device=b.device, dtype=b.dtype)
        Rb[0, 0] = torch.sqrt(1 + norm_b**2)
        Rb[0, 1:] = b_.view(1, -1)
        Rb[1:, 0] = b_.view(-1, 1).squeeze()
        Rb[1:, 1:] = self.matrix_sqrtm(I + torch.outer(b_.view(-1), b_.view(-1)))

        return Rb @ x

    def centroid(self, 
                 mode: str = 'default', 
                 lr: float = config.frechet_lr, 
                 max_iter: int = config.frechet_max_iter, 
                 tol: float = config.frechet_tol) -> torch.Tensor:
        """
        Compute the centroid of the points in the Loid space.
        
        Args:
            mode (str): The mode to compute the centroid. 
            lr (float): Learning rate for the optimizer. Default is config.frechet_lr.
            max_iter (int): Maximum number of iterations. Default is config.frechet_max_iter.
            tol (float): Tolerance for stopping criterion. Default is config.frechet_tol.
        
        Returns:
            torch.Tensor: The centroid of the points.
        """
        if mode == 'default':
            X = self.points
            centroid = X.mean(dim=1, keepdim=True)
            norm2 = -centroid[0]**2 + torch.sum(centroid[1:]**2, dim=0)
            
            centroid = 1/torch.sqrt(-norm2)*centroid
            centroid[0]= torch.sqrt(1+torch.sum(centroid[1:]**2))
            return centroid

        elif mode == 'Frechet':
            X = self.to_poincare(self.points)
            N = X.shape[1]

            centroid = X.mean(dim=1, keepdim=True).clone().detach().requires_grad_(True)

            optimizer = Adam([centroid], lr=lr)

            for _ in range(max_iter):
                optimizer.zero_grad()
                distances = self.poincare_distance(X, centroid)
                loss = torch.sum(distances**2)
                loss.backward(retain_graph=True)  # Retain graph for multiple backward passes if needed
                optimizer.step()

                if torch.norm(centroid).item() >= 1:
                    centroid.data = centroid.data / torch.norm(centroid).item() * config.norm_projection

                if torch.norm(centroid.grad) < tol:
                    break
            return self.to_loid(centroid)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def center(self,mode = 'default') -> None:
        """Centers the points by translating them to the centroid."""
        centroid = self.centroid(mode = mode)
        self.log_info(f"Centroid computed: {centroid}")
        
        _centroid = -centroid
        _centroid[0] = centroid[0]
        self.translate(_centroid)
        self.log_info(f"Points centered. New points shape: {self._points.shape}")
#############################################################################################
class EuclideanSpace(Embedding):
    def __init__(self, points=None):
        super().__init__('euclidean', points)
        self.curvature = 0
    ##########################################
    def __repr__(self):
        return f"PointSet({self.points}, model='{self.model}', geometry=hyperbolic('{self.model}'), curvature={self.curvature})"
    ##########################################
    def translate(self, vector):
        if isinstance(self.points, np.ndarray):
            if len(vector) == self.dimension:
                self.points += np.reshape(vector, (self.dimension, 1))
            else:
                raise ValueError("Dimension of the translation vector is wrong.")
        elif isinstance(self.points, torch.Tensor):
            if len(vector) == self.dimension:
                self.points += vector.view(self.dimension, 1)
            else:
                raise ValueError("Dimension of the translation vector is wrong.")
        else:
            raise TypeError("Points should be either a NumPy array or a PyTorch tensor.")
    ##########################################
    def rotate(self, R):
        if isinstance(R, np.ndarray) or isinstance(R, torch.Tensor):
            if R.shape[1] == self.points.shape[0]:
                self.points = R @ self.points
            else:
                raise ValueError("Matrix dimensions are not compatible for multiplication.")
        else:
            raise TypeError("Input R must be a NumPy array or a PyTorch tensor.")


    def center(self):
        centroid = self.centroid()
        if isinstance(self.points, np.ndarray):
            self.points -= centroid.reshape(-1, 1)
        elif isinstance(self.points, torch.Tensor):
            self.points -= centroid.view(-1, 1)
        else:
            raise TypeError("Points should be either a NumPy array or a PyTorch tensor.")

    def centroid(self):
        if isinstance(self.points, np.ndarray):
            return np.mean(self.points, axis=1)
        elif isinstance(self.points, torch.Tensor):
            return torch.mean(self.points, axis=1)
        else:
            raise TypeError("Points should be either a NumPy array or a PyTorch tensor.")


#############################################################################################
#############################################################################################
#############################################################################################

class MultiEmbedding:
    def __init__(self):
        self.embeddings = {}

    def add_embedding(self, label, embedding):
        self.embeddings[label] = embedding

    def discard_embedding(self, labels):
        for label in labels:
            self.embeddings.pop(label, None)

    def align_embeddings(self):
        if not self.embeddings:
            return

        avg_distance_matrix = self.average_distance_matrix()

        gramian = -np.cosh(distance_matrix)
        X = spaceform_pca_lib.lgram_to_points(dimension, gramian)
        for n in range(N):
            x = X[:, n]
            X[:, n] = spaceform_pca_lib.project_vector_to_hyperbolic_space(x)

        reference_embedding = embedding.Embedding(geometry = 'hyperbolic', model = 'loid', curvature = -1)
        reference_embedding.points = X
        reference_embedding.center()
        
        for label, embedding in self.embeddings.items():
            embedding = embedding.center()
            # procrustes


    def average_distance_matrix(self):
        distance_matrices = []
        for embedding in self.embeddings.values():
            distance_matrix = embedding.distance_matrix()
            distance_matrices.append(distance_matrix)
        avg_distance_matrix = np.mean(distance_matrices, axis=0)
        return avg_distance_matrix

    def find_outliers(self):
        # Implement outlier detection logic
        pass


