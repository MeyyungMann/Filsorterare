import logging
from pathlib import Path
from typing import Dict, List, Tuple
import mimetypes
import docx
import PyPDF2
import chardet
import json
import yaml
import csv
import xml.etree.ElementTree as ET

class FileLoader:
    def __init__(self):
        self.supported_extensions = {
            # Text-based files
            '.txt': self._read_text_file,
            '.py': self._read_text_file,
            '.md': self._read_text_file,
            '.rst': self._read_text_file,
            '.log': self._read_text_file,
            '.ini': self._read_text_file,
            '.conf': self._read_text_file,
            '.cfg': self._read_text_file,
            '.json': self._read_json_file,
            '.yaml': self._read_yaml_file,
            '.yml': self._read_yaml_file,
            '.xml': self._read_xml_file,
            '.csv': self._read_csv_file,
            '.tsv': self._read_csv_file,
            
            # Document files
            '.pdf': self._read_pdf_file,
            '.docx': self._read_docx_file,
            '.doc': self._read_docx_file,
            '.odt': self._read_text_file,  # OpenDocument Text
            
            # Code files
            '.js': self._read_text_file,
            '.ts': self._read_text_file,
            '.html': self._read_text_file,
            '.css': self._read_text_file,
            '.java': self._read_text_file,
            '.cpp': self._read_text_file,
            '.c': self._read_text_file,
            '.h': self._read_text_file,
            '.hpp': self._read_text_file,
            '.cs': self._read_text_file,
            '.php': self._read_text_file,
            '.rb': self._read_text_file,
            '.go': self._read_text_file,
            '.rs': self._read_text_file,
            '.swift': self._read_text_file,
            '.kt': self._read_text_file,
            '.scala': self._read_text_file,
            
            # Data files
            '.sql': self._read_text_file,
            '.sh': self._read_text_file,
            '.bat': self._read_text_file,
            '.ps1': self._read_text_file,
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

    def _read_json_file(self, file_path: Path) -> str:
        """Read and format JSON file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)

    def _read_yaml_file(self, file_path: Path) -> str:
        """Read and format YAML file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return yaml.dump(data, default_flow_style=False)

    def _read_xml_file(self, file_path: Path) -> str:
        """Read and format XML file content."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode', method='xml')

    def _read_csv_file(self, file_path: Path) -> str:
        """Read and format CSV/TSV file content."""
        delimiter = ',' if file_path.suffix == '.csv' else '\t'
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            return '\n'.join([delimiter.join(row) for row in reader]) 