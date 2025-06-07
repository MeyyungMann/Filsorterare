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

# Define short category mappings
CATEGORY_MAPPINGS = {
    # Code related
    "python_function": "code",
    "python_class": "code",
    "python_import": "code",
    "code": "code",
    
    # Task related
    "todo": "task",
    "task": "task",
    "checklist": "task",
    "personal_todo": "task",
    "numbered_list": "task",
    "capitalized_list": "task",
    
    # Document related
    "markdown_headers": "doc",
    "proper_sentences": "doc",
    "narrative": "doc",
    "work_document": "doc",
    
    # List related
    "markdown_table": "list",
    "tab_separated": "list",
    "list": "list",
    
    # Shopping related
    "shopping": "personal",
    "grocery": "personal",
    "personal_shopping": "personal",
    
    # Meeting related
    "meeting": "meeting",
    "agenda": "meeting",
    "minutes": "meeting"
}

# Fallback categories for different content types
FALLBACK_CATEGORIES = {
    "code": "code",
    "list": "list",
    "table": "data",
    "narrative": "doc",
    "default": "misc"
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

    def _analyze_content(self, content: str, file_path: Path = None) -> Dict:
        """
        Analyze content to identify patterns and characteristics.
        Args:
            content: File content
            file_path: Path of the file being analyzed
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
        
        # Check filename for patterns
        if file_path:
            filename = file_path.stem.lower()
            if any(word in filename for word in ['todo', 'task', 'checklist']):
                analysis["content_patterns"].add("todo")
            if any(word in filename for word in ['meeting', 'agenda', 'minutes']):
                analysis["content_patterns"].add("meeting")
            if any(word in filename for word in ['shopping', 'grocery']):
                analysis["content_patterns"].add("shopping")
            if any(word in filename for word in ['code', 'script', 'py']):
                analysis["content_patterns"].add("code")
        
        # Analyze format
        if re.search(r'(def|class|import|return|print|function|var|const|let)\s', content):
            analysis["format"]["is_code"] = True
            analysis["structure"].append("code")
            analysis["content_patterns"].add("code")
            
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
            if freq > 1:  # Lowered threshold to catch more patterns
                analysis["semantic_features"].add(word)
                # Direct mapping of common words to patterns
                if word in ['todo', 'task', 'checklist']:
                    analysis["content_patterns"].add("todo")
                elif word in ['meeting', 'agenda', 'minutes']:
                    analysis["content_patterns"].add("meeting")
                elif word in ['shopping', 'grocery', 'milk', 'eggs', 'bread']:
                    analysis["content_patterns"].add("shopping")
                elif word in ['report', 'document', 'notes']:
                    analysis["content_patterns"].add("doc")

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

    def _get_short_category(self, analysis: Dict) -> str:
        """
        Get a short category name based on content analysis.
        Args:
            analysis: Content analysis dictionary
        Returns:
            Short category name
        """
        # First try to match content patterns
        for pattern in analysis["content_patterns"]:
            if pattern in CATEGORY_MAPPINGS:
                return CATEGORY_MAPPINGS[pattern]
        
        # Then try to match format
        for format_type, is_present in analysis["format"].items():
            if is_present and format_type in FALLBACK_CATEGORIES:
                return FALLBACK_CATEGORIES[format_type]
        
        # Finally, check semantic features
        for feature in analysis["semantic_features"]:
            if feature in CATEGORY_MAPPINGS:
                return CATEGORY_MAPPINGS[feature]
        
        return FALLBACK_CATEGORIES["default"]

    def suggest_categories(self, clusters: Dict[int, List[Path]], file_contents: Dict[Path, str]) -> Dict[int, str]:
        """
        Suggest categories for clusters of files.
        Args:
            clusters: Dictionary mapping cluster IDs to lists of file paths
            file_contents: Dictionary mapping file paths to their contents
        Returns:
            Dictionary mapping cluster IDs to category names
        """
        categories = {}
        
        for cluster_id, file_paths in clusters.items():
            # Analyze all files in the cluster
            cluster_patterns = set()
            cluster_features = set()
            format_counts = defaultdict(int)
            
            for path in file_paths:
                content = file_contents.get(path, "")
                analysis = self._analyze_content(content, path)  # Pass file path for filename analysis
                cluster_patterns.update(analysis["content_patterns"])
                cluster_features.update(analysis["semantic_features"])
                
                # Count format types
                for format_type, is_present in analysis["format"].items():
                    if is_present:
                        format_counts[format_type] += 1
            
            # Get the most common format
            dominant_format = max(format_counts.items(), key=lambda x: x[1])[0] if format_counts else "default"
            
            # Create a combined analysis
            combined_analysis = {
                "content_patterns": cluster_patterns,
                "semantic_features": cluster_features,
                "format": {k: v > 0 for k, v in format_counts.items()},
                "dominant_format": dominant_format
            }
            
            # Get short category name
            category = self._get_short_category(combined_analysis)
            categories[cluster_id] = category
            
        return categories
