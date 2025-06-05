import os
import shutil
from pathlib import Path
from typing import List, Set
import mimetypes

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def get_file_extension(file_path: Path) -> str:
    """Get file extension in lowercase."""
    return file_path.suffix.lower()

def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type is None or not mime_type.startswith('text/')

def get_unique_filename(target_path: Path) -> Path:
    """Generate a unique filename if the target path already exists."""
    if not target_path.exists():
        return target_path
        
    counter = 1
    while True:
        new_path = target_path.parent / f"{target_path.stem}_{counter}{target_path.suffix}"
        if not new_path.exists():
            return new_path
        counter += 1

def safe_move_file(source: Path, target: Path) -> bool:
    """
    Safely move a file, handling name conflicts.
    Returns True if successful, False otherwise.
    """
    try:
        target = get_unique_filename(target)
        shutil.move(str(source), str(target))
        return True
    except Exception as e:
        return False

def get_directory_size(directory: Path) -> int:
    """Get total size of a directory in bytes."""
    total_size = 0
    for path in directory.rglob('*'):
        if path.is_file():
            total_size += get_file_size(path)
    return total_size

def get_file_count(directory: Path) -> int:
    """Get total number of files in a directory."""
    return sum(1 for _ in directory.rglob('*') if _.is_file()) 