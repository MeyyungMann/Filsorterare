import logging
from typing import Dict, List
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import re
from sklearn.metrics.pairwise import cosine_similarity

class CategorySuggester:
    def __init__(self, model_path: str = r"C:\\Users\\meyyu\\Desktop\\mistral_7"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            ).to(self.device)
            logging.info(f"Loaded local model from: {model_path}")
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {str(e)}")
            raise

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def merge_similar_clusters(self, categories: Dict[str, str]) -> Dict[str, str]:
        names = list(set(categories.values()))
        name_embeddings = [self.embedder.encode(name) for name in names]

        merged = {}
        for i, name in enumerate(names):
            merged[name] = name
            for j in range(i + 1, len(names)):
                sim = cosine_similarity([name_embeddings[i]], [name_embeddings[j]])[0][0]
                if sim > 0.85:
                    merged[names[j]] = name

        new_categories = {}
        for cluster_id, cat in categories.items():
            new_categories[cluster_id] = merged.get(cat, cat)

        return new_categories

    def _analyze_content_similarity(self, file_paths: List[Path], file_contents: Dict[Path, str]) -> bool:
        if len(file_paths) <= 1:
            return True

        content_embeddings = []
        for path in file_paths:
            content = file_contents.get(path, "")
            embedding = self.embedder.encode(content)
            embedding = embedding.flatten()
            content_embeddings.append(embedding)

        semantic_similarities = []
        for i in range(len(content_embeddings)):
            for j in range(i + 1, len(content_embeddings)):
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(content_embeddings[i]),
                    torch.tensor(content_embeddings[j]),
                    dim=0
                )
                semantic_similarities.append(similarity.item())

        avg_similarity = sum(semantic_similarities) / len(semantic_similarities)

        # Extract meaningful content chunks with improved handling of technical content
        content_chunks = []
        for path in file_paths:
            content = file_contents.get(path, "")
            
            # Handle code blocks and technical sections
            if path.suffix in {'.py', '.js', '.java', '.cpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala'}:
                # For code files, split by functions/classes
                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            else:
                # For other files, split by paragraphs and preserve technical sections
                chunks = []
                current_chunk = []
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = []
                    else:
                        current_chunk.append(line)
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
            
            content_chunks.extend(chunks)

        # Create embeddings for content chunks
        chunk_embeddings = []
        for chunk in content_chunks:
            embedding = self.embedder.encode(chunk)
            embedding = embedding.flatten()
            chunk_embeddings.append(embedding)

        # Calculate chunk-level similarities
        chunk_similarities = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(chunk_embeddings[i]),
                    torch.tensor(chunk_embeddings[j]),
                    dim=0
                )
                chunk_similarities.append(similarity.item())

        avg_chunk_similarity = sum(chunk_similarities) / len(chunk_similarities) if chunk_similarities else 0

        # Enhanced keyword extraction for technical content
        all_keywords = {}
        for path in file_paths:
            content = file_contents.get(path, "").lower()
            
            # Preserve technical terms and domain-specific words
            words = []
            for word in content.split():
                # Keep technical terms intact (camelCase, snake_case, etc.)
                if any(c.isupper() for c in word) or '_' in word:
                    words.append(word)
                else:
                    # Clean regular words
                    cleaned = word.strip('.,!?()[]{}":;')
                    if len(cleaned) > 3 and cleaned.lower() not in {
                        'this', 'that', 'with', 'from', 'have', 'they', 'what',
                        'when', 'where', 'which', 'there', 'their', 'about',
                        'would', 'think', 'could', 'should', 'because', 'through'
                    }:
                        words.append(cleaned)
            
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            for word, freq in word_freq.items():
                if word in all_keywords:
                    all_keywords[word]['count'] += freq
                    all_keywords[word]['files'].add(path)
                else:
                    all_keywords[word] = {'count': freq, 'files': {path}}

        top_keywords = []
        for word, data in all_keywords.items():
            if data['count'] >= 2 or len(data['files']) > 1:
                top_keywords.append({
                    'word': word,
                    'count': data['count'],
                    'files': len(data['files'])
                })

        top_keywords.sort(key=lambda x: (x['files'], x['count']), reverse=True)
        top_keywords = top_keywords[:10]

        if not top_keywords:
            return False

        keyword_summary = "\n".join([
            f"- {kw['word']} (appears {kw['count']} times in {kw['files']} files)"
            for kw in top_keywords
        ])

        prompt = f"""Given these keywords from several documents, what is the common theme?

Keywords:
{keyword_summary}

Please analyze these keywords and:
1. Identify the main theme or subject
2. Determine if these documents are related
3. Provide a brief explanation of why they belong together

Make sure to:
- Consider technical and domain-specific terminology
- Look for patterns in code structure or medical terminology
- Identify common programming concepts or medical conditions
- Group similar technical documentation together
- Consider both high-level concepts and specific implementations

Analysis:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = decoded.split("Analysis:")[-1].strip()

            if not analysis or not isinstance(analysis, str):
                logging.error(f"Invalid analysis text: {analysis}")
                return False

            analysis_embedding = self.embedder.encode(analysis)
            if isinstance(analysis_embedding, str):
                logging.error("Analysis embedding is still a string!")
                return False
            analysis_embedding = np.array(analysis_embedding).flatten()

            analysis_similarities = []
            for i, embedding in enumerate(content_embeddings):
                embedding = np.array(embedding).flatten()
                analysis_tensor = torch.from_numpy(analysis_embedding)
                content_tensor = torch.from_numpy(embedding)
                if torch.cuda.is_available():
                    analysis_tensor = analysis_tensor.cuda()
                    content_tensor = content_tensor.cuda()
                similarity = torch.nn.functional.cosine_similarity(
                    analysis_tensor,
                    content_tensor,
                    dim=0
                )
                analysis_similarities.append(similarity.item())

            avg_analysis_similarity = sum(analysis_similarities) / len(analysis_similarities)

            # Adjusted thresholds for technical content
            return (avg_similarity > 0.2 or avg_chunk_similarity > 0.3) and avg_analysis_similarity > 0.2

        except Exception as e:
            logging.error(f"Error in content similarity analysis: {str(e)}")
            return False

    def _generate_prompt(self, file_paths: List[Path], file_contents: Dict[Path, str]) -> str:
        """
        Generate a prompt for the model to suggest a category.
        Args:
            file_paths: List of file paths in the cluster
            file_contents: Dictionary mapping file paths to their contents
        Returns:
            Formatted prompt string
        """
        # First check if files are similar enough to be grouped
        if not self._analyze_content_similarity(file_paths, file_contents):
            return None  # Signal that these files should not be grouped together
        
        file_info = []
        
        for path in file_paths:
            content = file_contents.get(path, "")
            # Get first 3 lines or first 200 characters
            preview = content.strip().split("\n")[:3]
            preview = "\n".join(preview)[:200]
            file_info.append(f"File: {path.name}\nPreview:\n{preview}\n")
        
        prompt = f"""Below is a list of files and a short content preview from each file.

Your task is to suggest a short, descriptive category name that reflects the main theme or purpose shared by these files.

Files:
{''.join(file_info)}

Instructions:
- Analyze the actual content of each file to identify common themes and purposes
- Create a category name that encompasses all files in the list
- If the category name contains multiple words, separate them with underscores (e.g., 'project_docs', 'team_meetings')
- Use common, natural language naming conventions
- Do not use generic names like 'Miscellaneous' or 'Documents'
- Base your suggestion purely on the actual content
- Avoid referencing filenames directly
- Be specific and meaningful
- Keep the category name concise (1-3 words)
- IMPORTANT: Only group files if they share similar content or purpose
- IMPORTANT: Look for common themes in the actual content, not just in filenames
- IMPORTANT: Consider the purpose and context of the content
- IMPORTANT: Group similar content types together (e.g., all recipes should be in one category)
- IMPORTANT: Do not combine unrelated content types unless they share a clear common theme
- IMPORTANT: Consider file types and their typical purposes (e.g., recipes, documentation, notes, etc.)

Category name:"""
        return prompt

    def _get_category_suggestion(self, prompt: str) -> str:
        """
        Get category suggestion from the model.
        Args:
            prompt: Formatted prompt string
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
                do_sample=True,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        category = response.split("Category name:")[-1].strip()
        
        # Clean up the category name
        category = category.split("\n")[0].strip()  # Take only the first line
        category = re.sub(r'[^\w\s-]', '', category)  # Remove special characters
        category = category.strip()
        
        # Replace spaces with underscores
        category = category.replace(' ', '_')
        
        # Normalize common category types
        category = category.lower()
        
        # Define category mappings
        category_mappings = {
            # Recipes and food
            r'recipe|food|cooking|dish|meal': 'recipes',
            r'grocery|shopping|ingredients': 'grocery_lists',
            
            # Documentation
            r'doc|documentation|api|reference|guide|manual': 'documentation',
            r'readme|setup|install|config': 'setup_docs',
            
            # Notes and learning
            r'note|study|learning|lecture|course': 'notes',
            r'math|calculation|formula|equation': 'math_notes',
            
            # Work and projects
            r'task|todo|project|work|assignment': 'tasks',
            r'meeting|minutes|discussion|agenda': 'meetings',
            r'report|summary|analysis|review': 'reports',
            
            # Planning
            r'plan|schedule|itinerary|agenda': 'plans',
            r'travel|trip|vacation|journey': 'travel_plans',
            r'workout|exercise|fitness|training': 'workout_plans',
            
            # Code
            r'code|script|program|software': 'code',
            r'python|java|cpp|csharp|javascript': 'code',
            
            # Generic
            r'misc|other|various|general': 'miscellaneous'
        }
        
        # Apply category mappings
        for pattern, mapped_category in category_mappings.items():
            if re.search(pattern, category, re.IGNORECASE):
                category = mapped_category
                break
        
        # Handle generic categories
        if category.lower() in {"documents", "misc", "files", "other"}:
            category = "miscellaneous"
            
        return category

    def suggest_categories(self, clusters: Dict[str, List[Path]], file_contents: Dict[Path, str]) -> Dict[str, str]:
        """
        Suggest categories for clusters of files.
        Args:
            clusters: Dictionary mapping cluster IDs to lists of file paths
            file_contents: Dictionary mapping file paths to their contents
        Returns:
            Dictionary mapping cluster IDs to category names
        """
        categories = {}
        new_clusters = {}  # Store new clusters here
        
        # First, process all clusters
        for cluster_id, file_paths in clusters.items():
            prompt = self._generate_prompt(file_paths, file_contents)
            if prompt is None:
                # If files are too different, create separate categories
                for i, path in enumerate(file_paths):
                    new_id = f"{cluster_id}_{i}"
                    categories[new_id] = f"Category_{path.stem}"
                    new_clusters[new_id] = [path]
            else:
                try:
                    category = self._get_category_suggestion(prompt)
                    categories[cluster_id] = category
                    new_clusters[cluster_id] = file_paths
                    logging.info(f"Cluster {cluster_id} suggested category: {category}")
                except Exception as e:
                    logging.error(f"Error generating category for cluster {cluster_id}: {str(e)}")
                    categories[cluster_id] = f"Category_{cluster_id}"
                    new_clusters[cluster_id] = file_paths
        
        # Update the clusters dictionary with the new structure
        clusters.clear()
        clusters.update(new_clusters)
        
        return categories

    def get_category_guidelines(self) -> Dict[str, str]:
        """
        Get guidelines for categories.
        Returns:
            Dictionary mapping category names to their guidelines
        """
        # Since we're using dynamic categories, we'll return a generic guideline
        return {
            "guidelines": """
Categories are generated based on the content of files in each cluster.
Each category name reflects the main theme or purpose shared by its files.
Categories are designed to be:
- Short and descriptive (1-3 words)
- Based on actual content
- Specific and meaningful
- Using natural language
"""
        }
