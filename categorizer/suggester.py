import logging
from typing import Dict, List
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CategorySuggester:
    def __init__(self, model_path: str = r"C:\Users\meyyu\Desktop\mistral_7"):
        """
        Initialize the suggester with the local Mistral model.
        Args:
            model_path: Path to the local Mistral model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            logging.info(f"Loaded local model from: {model_path}")
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {str(e)}")
            raise

    def _generate_prompt(self, file_paths: List[Path]) -> str:
        """
        Generate a prompt for the model based on file paths.
        Args:
            file_paths: List of file paths in a cluster
        Returns:
            Formatted prompt string
        """
        file_names = [path.name for path in file_paths]
        prompt = f"""Given these files:
{', '.join(file_names)}

What would be a good category name for these files? 
The category name should be:
1. Short and descriptive
2. In English
3. A single word or short phrase
4. Reflect the common theme or purpose of these files

Category name:"""
        return prompt

    def _get_category_suggestion(self, prompt: str) -> str:
        """
        Get a category suggestion from the model.
        Args:
            prompt: The prompt to send to the model
        Returns:
            Suggested category name
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the category name from the response
        category = response.split("Category name:")[-1].strip()
        return category

    def suggest_categories(self, clusters: Dict[int, List[Path]]) -> Dict[int, str]:
        """
        Suggest category names for each cluster of files.
        Args:
            clusters: Dictionary mapping cluster IDs to lists of file paths
        Returns:
            Dictionary mapping cluster IDs to category names
        """
        categories = {}
        
        for cluster_id, file_paths in clusters.items():
            prompt = self._generate_prompt(file_paths)
            try:
                category = self._get_category_suggestion(prompt)
                categories[cluster_id] = category
                logging.info(f"Cluster {cluster_id} suggested category: {category}")
            except Exception as e:
                logging.error(f"Error generating category for cluster {cluster_id}: {str(e)}")
                categories[cluster_id] = f"Category_{cluster_id}"
                
        return categories 