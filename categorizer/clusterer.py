import logging
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from sklearn.metrics.pairwise import cosine_similarity

class ContentClusterer:
    def __init__(self, min_clusters: int = 2, max_clusters: int = 10, similarity_threshold: float = 0.3):
        """
        Initialize the clusterer with parameters for clustering.
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            similarity_threshold: Minimum similarity score to consider files similar
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold
    
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
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # If most files are very similar, use fewer clusters
        avg_similarity = np.mean(similarities)
        if avg_similarity > 0.7:  # High average similarity
            return max(2, min(3, len(embeddings)))
        elif avg_similarity > 0.5:  # Medium average similarity
            return max(2, min(5, len(embeddings)))
        
        # Otherwise, use silhouette score
        for n_clusters in range(self.min_clusters, min(self.max_clusters + 1, len(embeddings))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(embeddings, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    
        return best_n_clusters
    
    def cluster_files(self, file_embeddings: Dict[Path, np.ndarray], file_contents: Dict[Path, str]) -> Dict[str, List[Path]]:
        """
        Cluster files using semantic similarity.
        Args:
            file_embeddings: Dictionary mapping file paths to their embeddings
            file_contents: Dictionary mapping file paths to their contents
        Returns:
            Dictionary mapping cluster IDs to lists of file paths
        """
        if not file_embeddings:
            return {}
        
        # Convert embeddings to numpy array
        paths = list(file_embeddings.keys())
        embeddings = np.array([file_embeddings[path] for path in paths])
        
        # If only one file, return it as a single cluster
        if len(paths) == 1:
            return {"0": paths}
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find optimal number of clusters
        n_clusters = self.find_optimal_clusters(embeddings)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group files by cluster
        clusters = {}
        for label in range(n_clusters):
            cluster_files = [path for path, l in zip(paths, cluster_labels) if l == label]
            if cluster_files:  # Only add non-empty clusters
                # Use string IDs for clusters
                cluster_id = str(len(clusters))
                clusters[cluster_id] = cluster_files
        
        # Post-process clusters to merge similar ones
        merged_clusters = {}
        used_clusters = set()
        
        for cluster_id, files in clusters.items():
            if cluster_id in used_clusters:
                continue
                
            current_cluster = files.copy()
            used_clusters.add(cluster_id)
            
            # Check similarity with other clusters
            for other_id, other_files in clusters.items():
                if other_id in used_clusters:
                    continue
                    
                # Calculate average similarity between clusters
                cluster_similarities = []
                for file1 in files:
                    for file2 in other_files:
                        idx1 = paths.index(file1)
                        idx2 = paths.index(file2)
                        cluster_similarities.append(similarities[idx1][idx2])
                
                avg_similarity = np.mean(cluster_similarities)
                
                # Merge clusters if they're similar enough
                if avg_similarity > self.similarity_threshold:
                    current_cluster.extend(other_files)
                    used_clusters.add(other_id)
            
            # Add merged cluster
            merged_clusters[str(len(merged_clusters))] = current_cluster
        
        return merged_clusters
    
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