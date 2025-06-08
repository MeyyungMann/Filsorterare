import logging
from typing import List, Tuple, Dict
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class ContentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a specific sentence transformer model.
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logging.info(f"Loaded model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def create_embeddings(self, files_content: List[Tuple[Path, str]]) -> Dict[Path, np.ndarray]:
        """
        Create embeddings for a list of file contents.
        Args:
            files_content: List of tuples containing (file_path, content)
        Returns:
            Dictionary mapping file paths to their embeddings
        """
        if not files_content:
            logging.warning("No files to process")
            return {}
        
        # Extract just the content for batch processing
        contents = [content for _, content in files_content]
        file_paths = [path for path, _ in files_content]
        
        try:
            # Create embeddings in batches
            embeddings = self.model.encode(
                contents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create mapping of file paths to embeddings
            return dict(zip(file_paths, embeddings))
            
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def encode(self, text: str) -> np.ndarray:
        """
        Create an embedding for a single text string.
        Args:
            text: The text to encode
        Returns:
            numpy array containing the embedding
        """
        try:
            if not isinstance(text, str):
                raise ValueError(f"Expected string input, got {type(text)}")
            
            if not text.strip():
                raise ValueError("Empty text input")
            
            # Create embedding for single text
            embeddings = self.model.encode(
                [text],  # Wrap in list since model expects a list of texts
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Get the first embedding since we only encoded one text
            embedding = embeddings[0]
            
            # Ensure we get a 1D array
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            # Verify the output is a numpy array
            if not isinstance(embedding, np.ndarray):
                raise ValueError(f"Expected numpy array output, got {type(embedding)}")
            
            return embedding
            
        except Exception as e:
            logging.error(f"Error creating embedding: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Return the name of the model being used."""
        return self.model.get_config_dict()['model_name'] 