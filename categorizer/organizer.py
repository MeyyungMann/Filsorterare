import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set
import os

class FileOrganizer:
    def __init__(self, base_output_dir: Path):
        """
        Initialize the organizer with the base output directory.
        Args:
            base_output_dir: Base directory where categories will be created
        """
        self.base_output_dir = Path(base_output_dir)
        
    def _create_category_dir(self, category: str) -> Path:
        """
        Create a directory for a category, handling name conflicts.
        Args:
            category: Category name
        Returns:
            Path to the created directory
        """
        # Clean category name for filesystem
        clean_name = "".join(c for c in category if c.isalnum() or c in (' ', '-', '_')).strip()
        category_dir = self.base_output_dir / clean_name
        
        # Handle name conflicts
        counter = 1
        original_dir = category_dir
        while category_dir.exists():
            category_dir = original_dir.parent / f"{original_dir.name}_{counter}"
            counter += 1
            
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir
    
    def _handle_file_conflict(self, source: Path, target: Path) -> Path:
        """
        Handle file naming conflicts in the target directory.
        Args:
            source: Source file path
            target: Target file path
        Returns:
            New target path
        """
        if not target.exists():
            return target
            
        counter = 1
        original_target = target
        while target.exists():
            target = original_target.parent / f"{original_target.stem}_{counter}{original_target.suffix}"
            counter += 1
            
        return target
    
    def organize_files(self, 
                      clusters: Dict[int, List[Path]], 
                      categories: Dict[int, str],
                      dry_run: bool = False) -> Dict[str, List[Path]]:
        """
        Organize files into their respective category directories.
        Args:
            clusters: Dictionary mapping cluster IDs to lists of file paths
            categories: Dictionary mapping cluster IDs to category names
            dry_run: If True, only simulate the organization without moving files
        Returns:
            Dictionary mapping category names to lists of moved file paths
        """
        organized_files = {}
        moved_files: Set[Path] = set()
        
        for cluster_id, file_paths in clusters.items():
            if cluster_id not in categories:
                logging.warning(f"No category found for cluster {cluster_id}")
                continue
                
            category = categories[cluster_id]
            category_dir = self._create_category_dir(category)
            
            if category not in organized_files:
                organized_files[category] = []
                
            for file_path in file_paths:
                if file_path in moved_files:
                    continue
                    
                target_path = category_dir / file_path.name
                target_path = self._handle_file_conflict(file_path, target_path)
                
                try:
                    if not dry_run:
                        shutil.move(str(file_path), str(target_path))
                        logging.info(f"Moved {file_path} to {target_path}")
                    else:
                        logging.info(f"Would move {file_path} to {target_path}")
                        
                    organized_files[category].append(target_path)
                    moved_files.add(file_path)
                    
                except Exception as e:
                    logging.error(f"Error moving file {file_path}: {str(e)}")
                    
        return organized_files
    
    def create_summary(self, organized_files: Dict[str, List[Path]]) -> str:
        """
        Create a summary of the organization results.
        Args:
            organized_files: Dictionary mapping category names to lists of file paths
        Returns:
            Summary string
        """
        summary = ["File Organization Summary:", ""]
        
        for category, files in organized_files.items():
            summary.append(f"Category: {category}")
            summary.append(f"Number of files: {len(files)}")
            summary.append("Files:")
            for file_path in files:
                summary.append(f"  - {file_path.name}")
            summary.append("")
            
        return "\n".join(summary) 