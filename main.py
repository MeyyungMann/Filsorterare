# python main.py --input-dir ./test_files
# python main.py --input-dir ./test_files --dry-run
# add commit comment

#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List

from categorizer.loader import FileLoader
from categorizer.embedder import ContentEmbedder
from categorizer.clusterer import ContentClusterer
from categorizer.suggester import CategorySuggester
from categorizer.organizer import FileOrganizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description='Content-based file sorter using local AI'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for organized files (default: input_dir + "_organized")'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode without user interaction'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the organization without moving files'
    )
    return parser.parse_args()

def get_user_confirmation(categories: Dict[str, str], clusters: Dict[str, List[Path]]) -> bool:
    """Get user confirmation for the suggested categories."""
    print("\nSuggested Categories:")
    for cluster_id, category in categories.items():
        print(f"\nCategory: {category}")
        print("Files:")
        for file_path in clusters[cluster_id]:
            print(f"  - {file_path.name}")
    
    while True:
        response = input("\nDo you want to proceed with this organization? (yes/no): ").lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        print("Please answer 'yes' or 'no'")

def main():
    setup_logging()
    args = parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logging.error(f"Input directory {input_path} does not exist")
        return
    
    output_path = Path(args.output_dir) if args.output_dir else input_path.parent / f"{input_path.name}_organized"
    
    logging.info(f"Processing directory: {input_path}")
    logging.info(f"Output directory: {output_path}")
    logging.info(f"Mode: {'Headless' if args.headless else 'Interactive'}")
    
    try:
        # 1. Load files
        loader = FileLoader()
        files_content = loader.load_directory(input_path)
        if not files_content:
            logging.error("No supported files found in the input directory")
            return
        
        # Convert files_content to dictionary for easier access
        file_contents_dict = {path: content for path, content in files_content}
        
        # 2. Create embeddings
        embedder = ContentEmbedder()
        file_embeddings = embedder.create_embeddings(files_content)
        
        # 3. Cluster files
        clusterer = ContentClusterer()
        clusters = clusterer.cluster_files(file_embeddings, file_contents_dict)
        
        # 4. Suggest categories - Now passing file_contents_dict
        suggester = CategorySuggester()
        categories = suggester.suggest_categories(clusters, file_contents_dict)
        
        # 5. Get user confirmation if not in headless mode
        if not args.headless and not get_user_confirmation(categories, clusters):
            logging.info("Organization cancelled by user")
            return
        
        # 6. Organize files
        organizer = FileOrganizer(output_path)
        organized_files = organizer.organize_files(clusters, categories, args.dry_run)
        
        # 7. Print summary
        if not args.dry_run:
            print("\n" + organizer.create_summary(organized_files))
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 