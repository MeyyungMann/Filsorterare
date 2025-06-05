import logging
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import re
from collections import defaultdict

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
        # Initialize empty content type patterns
        self.content_patterns = defaultdict(list)
        self.content_weights = defaultdict(float)
        
    def learn_content_patterns(self, file_contents: Dict[Path, str], initial_categories: Dict[str, List[str]] = None):
        """
        Learn content patterns from the files and optionally initialize with known categories.
        Args:
            file_contents: Dictionary mapping file paths to their contents
            initial_categories: Optional dictionary of known categories and their patterns
        """
        # Initialize with known categories if provided
        if initial_categories:
            for category, patterns in initial_categories.items():
                self.content_patterns[category].extend(patterns)
                # Start with a base weight of 1.0
                self.content_weights[category] = 1.0
        
        # Extract patterns from content
        for content in file_contents.values():
            # Find common phrases and patterns
            words = re.findall(r'\b\w{4,}\b', content.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            # Add frequent words as potential patterns
            for word, freq in word_freq.items():
                if freq > 2:  # Word appears more than twice
                    # Add to a temporary category
                    self.content_patterns['temp'].append(word)
        
        # Calculate initial weights based on pattern distinctiveness
        self._calculate_pattern_weights()
    
    def _calculate_pattern_weights(self):
        """
        Calculate weights for each content type based on pattern distinctiveness.
        """
        # Count how many patterns are unique to each category
        pattern_counts = defaultdict(int)
        all_patterns = set()
        
        for category, patterns in self.content_patterns.items():
            for pattern in patterns:
                pattern_counts[pattern] += 1
                all_patterns.add(pattern)
        
        # Calculate weights based on pattern uniqueness
        for category, patterns in self.content_patterns.items():
            unique_patterns = sum(1 for p in patterns if pattern_counts[p] == 1)
            total_patterns = len(patterns)
            if total_patterns > 0:
                # Weight is based on the ratio of unique patterns
                self.content_weights[category] = 1.0 + (unique_patterns / total_patterns)
    
    def _analyze_content_type(self, content: str) -> str:
        """
        Analyze content type using learned patterns.
        Args:
            content: File content
        Returns:
            Most likely content type
        """
        content = content.lower()
        scores = defaultdict(float)
        
        # Score each category based on pattern matches
        for category, patterns in self.content_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    scores[category] += 1
        
        # Return the category with the highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "unknown"
    
    def _adjust_embeddings(self, embeddings: np.ndarray, file_paths: List[Path], file_contents: Dict[Path, str]) -> np.ndarray:
        """
        Adjust embeddings using learned weights.
        """
        adjusted_embeddings = embeddings.copy()
        
        for i, path in enumerate(file_paths):
            content = file_contents.get(path, "")
            content_type = self._analyze_content_type(content)
            weight = self.content_weights.get(content_type, 1.0)
            adjusted_embeddings[i] *= weight
            
        return adjusted_embeddings
    
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
    
    def cluster_files(self, file_embeddings: Dict[Path, np.ndarray], file_contents: Dict[Path, str]) -> Dict[int, List[Path]]:
        """
        Cluster files using learned patterns and weights.
        """
        if not file_embeddings:
            return {}
        
        # First, do a strict content type pre-clustering
        content_type_groups = defaultdict(list)
        for path, content in file_contents.items():
            # Check for code first - most distinct
            if any(keyword in content for keyword in ['def ', 'class ', 'import ', 'return ', 'print(']):
                content_type_groups['code'].append(path)
                continue
            
            # Check for work documents
            if any(keyword in content for keyword in ['meeting', 'agenda', 'report', 'project', 'team', 'business']):
                content_type_groups['work'].append(path)
                continue
            
            # Check for shopping lists
            if any(keyword in content for keyword in ['shopping', 'grocery', 'buy', 'milk', 'eggs', 'bread']):
                content_type_groups['shopping'].append(path)
                continue
            
            # Check for todos
            if any(keyword in content for keyword in ['todo', 'task', 'to-do', 'to do', 'checklist']):
                content_type_groups['todo'].append(path)
                continue
            
            content_type_groups['other'].append(path)
        
        # Now cluster within each content type group
        final_clusters = {}
        cluster_id = 0
        
        for content_type, paths in content_type_groups.items():
            if not paths:
                continue
            
            # Get embeddings for this content type
            type_embeddings = np.array([file_embeddings[path] for path in paths])
            
            # If only one file in this type, add it directly to clusters
            if len(paths) == 1:
                final_clusters[cluster_id] = paths
                cluster_id += 1
                continue
            
            # Find optimal number of clusters for this content type
            n_clusters = min(len(paths), self.find_optimal_clusters(type_embeddings))
            
            # Perform clustering for this content type
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            cluster_labels = kmeans.fit_predict(type_embeddings)
            
            # Group files by cluster
            for label in range(n_clusters):
                cluster_files = [path for path, l in zip(paths, cluster_labels) if l == label]
                if cluster_files:  # Only add non-empty clusters
                    final_clusters[cluster_id] = cluster_files
                    cluster_id += 1
        
        return final_clusters
    
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