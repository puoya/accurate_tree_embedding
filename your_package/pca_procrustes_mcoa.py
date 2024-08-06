# your_package/pca_procrustes_mcoa.py

class PCAResults:
    def __init__(self, eigen_values, eigen_vectors):
        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors

    @staticmethod
    def pca(embedding, dimensions):
        # Implement PCA logic
        pass

    @staticmethod
    def procrustes(embedding1, embedding2):
        # Implement Procrustes analysis logic
        pass
