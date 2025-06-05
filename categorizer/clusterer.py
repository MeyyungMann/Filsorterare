import logging
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

class ContentClusterer:
    def __init__(self, min_clusters: int = 2, max_clusters: int = 10):
        """
        Initialize the clusterer with parameters for clustering.
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
    def find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Find the optimal number of clusters using silhouette score.
        Args:
            embeddings: Array of embeddings to cluster
        Returns:
            Optimal number of clusters
        """
        if len(embeddings) < self.min_clusters:
            return len(embeddings)
            
        best_score = -1
        best_n_clusters = self.min_clusters
        
        for n_clusters in range(self.min_clusters, min(self.max_clusters + 1, len(embeddings))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(embeddings, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    
        return best_n_clusters
    
    def cluster_files(self, file_embeddings: Dict[Path, np.ndarray]) -> Dict[int, List[Path]]:
        """
        Cluster files based on their embeddings.
        Args:
            file_embeddings: Dictionary mapping file paths to their embeddings
        Returns:
            Dictionary mapping cluster IDs to lists of file paths
        """
        if not file_embeddings:
            return {}
            
        # Convert embeddings to array
        embeddings = np.array(list(file_embeddings.values()))
        file_paths = list(file_embeddings.keys())
        
        # Find optimal number of clusters
        n_clusters = self.find_optimal_clusters(embeddings)
        logging.info(f"Optimal number of clusters: {n_clusters}")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group files by cluster
        clusters = {}
        for file_path, label in zip(file_paths, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(file_path)
            
        return clusters
    
    def get_cluster_centers(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Get the center points of each cluster.
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters
        Returns:
            Array of cluster centers
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_ 