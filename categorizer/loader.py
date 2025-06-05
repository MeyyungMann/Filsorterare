import logging
from pathlib import Path
from typing import Dict, List, Tuple
import mimetypes
import docx
import PyPDF2
import chardet

class FileLoader:
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._read_text_file,
            '.pdf': self._read_pdf_file,
            '.docx': self._read_docx_file,
            # Add more file types as needed
        }
        
    def load_directory(self, directory_path: Path) -> List[Tuple[Path, str]]:
        """
        Load all supported files from a directory and its subdirectories.
        Returns a list of tuples containing (file_path, content).
        """
        files_content = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                content = self.load_file(file_path)
                if content:
                    files_content.append((file_path, content))
                    
        return files_content
    
    def load_file(self, file_path: Path) -> str:
        """
        Load content from a single file.
        Returns the file content as a string, or None if file type is not supported.
        """
        extension = file_path.suffix.lower()
        
        if extension in self.supported_extensions:
            try:
                return self.supported_extensions[extension](file_path)
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {str(e)}")
                return None
        else:
            logging.warning(f"Unsupported file type: {extension} for file {file_path}")
            return None
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read content from a text file with automatic encoding detection."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def _read_pdf_file(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        text = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    
    def _read_docx_file(self, file_path: Path) -> str:
        """Extract text from a DOCX file."""
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs]) 