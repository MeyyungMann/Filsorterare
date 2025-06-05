import logging
from typing import Dict, List, Tuple
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

CONTENT_TYPE_PATTERNS = {
    "code": ['def ', 'class ', 'import ', 'return ', 'print('],
    "personal_shopping": ['shopping', 'grocery', 'buy', 'milk', 'eggs', 'bread', 'food', 'vegetables', 'fruits'],
    "personal_todo": ['todo', 'task', 'to-do', 'to do', 'checklist'],
    "work_document": ['meeting', 'agenda', 'minutes', 'attendees', 'project', 'team', 'business', 'report', 'quarterly', 'sales']
}

CATEGORY_GUIDELINES = {
    "code": "Use 'Source Code' for programming scripts or code snippets.",
    "personal_shopping": "Use 'Shopping Lists' for grocery or purchase lists.",
    "personal_todo": "Use 'Personal Tasks' for task lists or personal to-do items.",
    "work_document": "Use 'Work Documents' for meetings, business reports, or agendas.",
    "document": "Use 'General Notes' for uncategorized or miscellaneous content."
}

class CategorySuggester:
    def __init__(self, model_path: str = r"C:\Users\meyyu\Desktop\mistral_7"):
        """
        Initialize the suggester with the local Mistral model and automatic learning components.
        Args:
            model_path: Path to the local Mistral model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        # Initialize the LLM
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

        # Initialize the embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize pattern recognition
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict:
        """Initialize base patterns for automatic recognition."""
        return {
            "content_patterns": defaultdict(list),
            "structural_patterns": defaultdict(list),
            "semantic_patterns": defaultdict(list)
        }

    def _analyze_content(self, content: str) -> Dict:
        """
        Analyze content to identify patterns and characteristics.
        Args:
            content: File content
        Returns:
            Dictionary of content analysis results
        """
        content = content.lower()
        analysis = {
            "format": {
                "is_code": False,
                "is_list": False,
                "is_table": False,
                "is_narrative": False
            },
            "structure": [],
            "semantic_features": set(),
            "content_patterns": set()
        }
        
        # Analyze format
        if re.search(r'(def|class|import|return|print|function|var|const|let)\s', content):
            analysis["format"]["is_code"] = True
            analysis["structure"].append("code")
            
        if re.search(r'^[\s-]*[-*•]\s', content, re.MULTILINE):
            analysis["format"]["is_list"] = True
            analysis["structure"].append("list")
            
        if re.search(r'\|.*\|', content) or re.search(r'\t.*\t', content):
            analysis["format"]["is_table"] = True
            analysis["structure"].append("table")
            
        if len(re.findall(r'[.!?]', content)) > 3:
            analysis["format"]["is_narrative"] = True
            analysis["structure"].append("narrative")

        # Extract semantic features
        words = re.findall(r'\b\w{4,}\b', content)
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
            
        # Add frequent words as semantic features
        for word, freq in word_freq.items():
            if freq > 2:
                analysis["semantic_features"].add(word)

        # Identify content patterns
        self._identify_content_patterns(content, analysis)
        
        return analysis

    def _identify_content_patterns(self, content: str, analysis: Dict):
        """
        Identify patterns in the content.
        Args:
            content: File content
            analysis: Analysis dictionary to update
        """
        # Code patterns
        if analysis["format"]["is_code"]:
            if re.search(r'def\s+\w+\s*\(', content):
                analysis["content_patterns"].add("python_function")
            if re.search(r'class\s+\w+', content):
                analysis["content_patterns"].add("python_class")
            if re.search(r'import\s+\w+', content):
                analysis["content_patterns"].add("python_import")

        # List patterns
        if analysis["format"]["is_list"]:
            if re.search(r'^[\s-]*[-*•]\s*[A-Z]', content, re.MULTILINE):
                analysis["content_patterns"].add("capitalized_list")
            if re.search(r'^[\s-]*[-*•]\s*\d+\.', content, re.MULTILINE):
                analysis["content_patterns"].add("numbered_list")

        # Table patterns
        if analysis["format"]["is_table"]:
            if re.search(r'\|.*\|.*\|', content):
                analysis["content_patterns"].add("markdown_table")
            if re.search(r'\t.*\t', content):
                analysis["content_patterns"].add("tab_separated")

        # Narrative patterns
        if analysis["format"]["is_narrative"]:
            if re.search(r'^#+\s', content, re.MULTILINE):
                analysis["content_patterns"].add("markdown_headers")
            if re.search(r'^[A-Z][^.!?]*[.!?]', content, re.MULTILINE):
                analysis["content_patterns"].add("proper_sentences")

    def _generate_prompt(self, file_paths: List[Path], file_contents: Dict[Path, str]) -> str:
        """
        Generate a prompt for the model based on file paths and their contents.
        Args:
            file_paths: List of file paths in a cluster
            file_contents: Dictionary mapping file paths to their contents
        Returns:
            Formatted prompt string
        """
        # Analyze each file's content
        file_info = []
        all_patterns = set()
        all_features = set()
        
        for path in file_paths:
            content = file_contents.get(path, "")
            analysis = self._analyze_content(content)
            all_patterns.update(analysis["content_patterns"])
            all_features.update(analysis["semantic_features"])
            
            # Get first 3 lines of content or first 200 characters
            preview = content.split('\n')[:3]
            preview = '\n'.join(preview)[:200]
            
            file_info.append(
                f"File: {path.name}\n"
                f"Format: {', '.join(analysis['structure']) if analysis['structure'] else 'General text'}\n"
                f"Patterns: {', '.join(analysis['content_patterns']) if analysis['content_patterns'] else 'None'}\n"
                f"Preview: {preview}\n"
            )

        prompt = f"""Given these files and their content analysis:

{''.join(file_info)}

Content Analysis:
- Detected Patterns: {', '.join(all_patterns) if all_patterns else 'None'}
- Semantic Features: {', '.join(all_features) if all_features else 'None'}

Please suggest a category name for these files based on their content and purpose.
The category name should be:
1. Short and descriptive (1-3 words)
2. In English
3. Reflect the common theme or purpose of these files
4. Be specific enough to distinguish from other categories
5. Use common file organization conventions

Consider:
- The detected content patterns
- The semantic features
- The file structure and format
- Common file organization practices

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

    def suggest_categories(self, clusters: Dict[int, List[Path]], file_contents: Dict[Path, str]) -> Dict[int, str]:
        """
        Suggest category names for each cluster of files.
        Args:
            clusters: Dictionary mapping cluster IDs to lists of file paths
            file_contents: Dictionary mapping file paths to their contents
        Returns:
            Dictionary mapping cluster IDs to category names
        """
        categories = {}
        
        for cluster_id, file_paths in clusters.items():
            prompt = self._generate_prompt(file_paths, file_contents)
            try:
                category = self._get_category_suggestion(prompt)
                categories[cluster_id] = category
                logging.info(f"Cluster {cluster_id} suggested category: {category}")
            except Exception as e:
                logging.error(f"Error generating category for cluster {cluster_id}: {str(e)}")
                categories[cluster_id] = f"Category_{cluster_id}"
                
        return categories
