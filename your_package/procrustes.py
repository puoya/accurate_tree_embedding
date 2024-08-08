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


class HyperbolicProcrustes:
    """
    A class for performing Hyperbolic orthogonal Procrustes analysis, mapping one embedding to another.

    Attributes:
        cost (float): The cost associated with the mapping, inversely related to its quality.
        _mapping_matrix (torch.Tensor): Transformation matrix that maps the source embedding to the target embedding.
        source_embedding (Embedding): Source embedding instance.
        target_embedding (Embedding): Target embedding instance.
        logger (logging.Logger): Logger for recording class activities if enabled.
    """
    
    def __init__(self, 
                 source_embedding: 'Embedding', 
                 target_embedding: 'Embedding', 
                 mode: str = 'default', 
                 enable_logging: bool = False):
        """
        Initializes the HyperbolicProcrustes instance.

        Args:
            source_embedding (Embedding): The source embedding to map from.
            target_embedding (Embedding): The target embedding to map to.
            mode (str): Mode of computation, either 'default' or 'accurate'.
            enable_logging (bool): If True, enables logging. Default is False.
        """
        self.logger = None
        if enable_logging:
            self.setup_logging()

        self.source_embedding = source_embedding
        self.source_model = source_embedding.model
        self.target_embedding = target_embedding
        self.target_model = target_embedding.model
        self._mapping_matrix = None

        self.log_info("Initializing HyperbolicProcrustes")
        self._validate_embeddings()
        self._compute_mapping(mode=mode)

    def setup_logging(self, log_dir: str = "log", log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s') -> None:
        """
        Configures logging for the class.

        Args:
            log_dir (str): Directory for saving log files. Defaults to 'log'.
            log_level (int): Logging level. Defaults to logging.INFO.
            log_format (str): Format for log messages. Defaults to '%(asctime)s - %(levelname)s - %(message)s'.
        """
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"HyperbolicProcrustes_{current_time}.log")
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
        self.log_info("Logging setup complete.")

    def log_info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to be logged.
        """
        if self.logger:
            self.logger.info(message)

    def _validate_embeddings(self) -> None:
        """
        Validates that the source and target embeddings have compatible dimensions and curvatures.
        """
        self.log_info("Validating embeddings")

        # Ensure embeddings are in compatible models
        if self.source_embedding.model == 'loid':
            self.source_embedding = self.source_embedding.switch_model()
        if self.target_embedding.model == 'loid':
            self.target_embedding = self.target_embedding.switch_model()

        # Check for matching shapes
        if self.source_embedding.points.shape != self.target_embedding.points.shape:
            self.log_info("Source and target embeddings must have the same shape")
            raise ValueError("Source and target embeddings must have the same shape")
        
        # Check for matching curvatures
        if not torch.isclose(self.source_embedding.curvature, self.target_embedding.curvature, config.atol):
            self.log_info("Source and target curvatures must be equal")
            raise ValueError("Source and target curvatures must be equal")
        
        self.log_info("Validation successful")

    def project_to_orthogonal(self, R: torch.Tensor) -> torch.Tensor:
        """
        Projects the given matrix R to the nearest orthogonal matrix using Singular Value Decomposition (SVD).

        Args:
            R (torch.Tensor): The matrix to be projected.

        Returns:
            torch.Tensor: The orthogonal matrix closest to R.
        """
        U, _, V = torch.svd(R)
        return U @ V.T

    def matrix_sqrtm(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix square root of a positive definite matrix with eigenvalue thresholding.

        Args:
            A (torch.Tensor): A positive definite matrix.

        Returns:
            torch.Tensor: The matrix square root of A.
        """
        eigvals, eigvecs = torch.linalg.eigh(A)
        eigvals = torch.clamp(eigvals, min=0)
        sqrt_eigvals = torch.diag(torch.sqrt(eigvals))
        return eigvecs @ sqrt_eigvals @ eigvecs.T

    def map_to_translation(self, b: torch.Tensor) -> torch.Tensor:
        """
        Constructs a translation matrix for the given vector b.

        Args:
            b (torch.Tensor): The translation vector.

        Returns:
            torch.Tensor: The translation matrix.
        """
        D = len(b)
        norm_b = torch.norm(b)
        I = torch.eye(D, device=b.device, dtype=b.dtype)
        Rb = torch.zeros((D + 1, D + 1), device=b.device, dtype=b.dtype)
        Rb[0, 0] = torch.sqrt(1 + norm_b**2)
        Rb[0, 1:] = b.view(1, -1)
        Rb[1:, 0] = b.view(-1, 1).squeeze()
        Rb[1:, 1:] = self.matrix_sqrtm(I + torch.outer(b.view(-1), b.view(-1)))
        return Rb

    def map_to_rotation(self, U: torch.Tensor) -> torch.Tensor:
        """
        Constructs a rotation matrix from the given matrix U.

        Args:
            U (torch.Tensor): The rotation matrix.

        Returns:
            torch.Tensor: The rotation matrix in augmented space.
        """
        D = U.size(0)
        Ru = torch.eye(D + 1, device=U.device, dtype=U.dtype)
        Ru[1:, 1:] = U
        return Ru

    def _compute_mapping(self, mode: str = 'default') -> None:
        """
        Computes the Hyperbolic orthogonal Procrustes mapping and associated cost.

        Args:
            mode (str): Computation mode, 'default' for a basic approach or 'accurate' for refined optimization.
        """
        self.log_info("Computing mapping")

        D = self.source_embedding.dimension

        # Center the source and target embeddings
        src_embedding = self.source_embedding.copy()
        src_center = src_embedding.centroid()
        src_embedding.center()

        target_embedding = self.target_embedding.copy()
        target_center = target_embedding.centroid()
        target_embedding.center()

        # Compute optimal rotation matrix using SVD
        target_points_loid = src_embedding.to_loid(target_embedding.points)
        src_points_loid = src_embedding.to_loid(src_embedding.points)

        U, _, Vt = torch.svd(torch.mm(target_points_loid[1:], src_points_loid[1:].T))
        R = torch.mm(U, Vt.T)
        src_embedding.rotate(R)

        # Compute the transformation matrix in Lorentzian space
        transformation = (
            self.map_to_translation(target_center) @ 
            self.map_to_rotation(R) @ 
            self.map_to_translation(-src_center)
        )
        self._mapping_matrix = transformation  # Default mode

        if mode == 'accurate':
            src_points = src_embedding._points
            b = torch.zeros(D, requires_grad=True)
            R = torch.eye(D, requires_grad=True)
            optimizer = Adam([b, R], lr=0.001)

            for epoch in range(1000):
                optimizer.zero_grad()

                # Apply the current rotation and translation to the source points
                src_embedding.points = R @ src_points.clone()
                src_embedding.translate(b)

                # Compute the cost function
                cost = sum(
                    (src_embedding.poincare_distance(src_embedding.points[:, n], target_embedding.points[:, n]))**2 
                    for n in range(src_embedding.n_points)
                )

                # Normalize translation vector if necessary
                if torch.norm(b) >= 1:
                    b.data.div_(torch.norm(b)).mul_(self.config.norm_projection)

                cost.backward(retain_graph=True)
                
                # Orthogonal projection of rotation matrix
                with torch.no_grad():
                    R.data.copy_(self.project_to_orthogonal(R))

                # Handle NaN gradients
                with torch.no_grad():
                    for param in optimizer.param_groups[0]['params']:
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad, nan=0.0)
                optimizer.step()

                self.log_info(f"Epoch {epoch}, Cost: {cost.item()}, b: {b.detach().numpy()}, R: {R.detach().numpy()}")
            
            # Final orthogonal projection of R
            R = self.project_to_orthogonal(R.detach())
            self._mapping_matrix = (
                self.map_to_translation(b.detach()) @ 
                self.map_to_rotation(R) @ 
                transformation
            )

    def map(self, source_embedding: 'Embedding') -> 'Embedding':
        """
        Applies the computed mapping matrix to transform the given embedding.

        Args:
            source_embedding (Embedding): The embedding to be transformed.

        Returns:
            Embedding: The transformed embedding.
        """
        self.log_info("Mapping Embedding")
        if source_embedding.model == 'poincare':
            source_embedding = source_embedding.switch_model()
        target_embedding = source_embedding.copy()
        target_embedding.points = (self._mapping_matrix) @ source_embedding.points
        if target_embedding.model != self.target_model:
            target_embedding = target_embedding.switch_model()
        
        self.log_info(f"Mapped points with shape: {target_embedding.points.shape}")
        return target_embedding