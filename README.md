# Filsorterare

An AI-powered file organization tool that automatically categorizes files based on their content using local AI models.

## Features

- Content-based file categorization using local AI
- Supports multiple file types (txt, pdf, docx)
- Automatic category suggestion using Mistral 7B
- Interactive and headless modes
- CUDA acceleration support
- Dry-run mode for testing

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/filsorterare.git
cd filsorterare
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python main.py --input-dir /path/to/your/files
```

Options:
- `--input-dir`: Directory containing files to organize (required)
- `--output-dir`: Directory where organized files will be placed (default: input_dir + "_organized")
- `--headless`: Run without user interaction
- `--dry-run`: Simulate organization without moving files

Examples:

1. Interactive mode (default):
```bash
python main.py --input-dir ./my_files
```

2. Headless mode:
```bash
python main.py --input-dir ./my_files --headless
```

3. Dry run (test without moving files):
```bash
python main.py --input-dir ./my_files --dry-run
```

4. Specify output directory:
```bash
python main.py --input-dir ./my_files --output-dir ./organized_files
```

## How It Works

1. **File Loading**: The tool reads supported files (txt, pdf, docx) from the input directory
2. **Content Analysis**: File contents are converted into embeddings using sentence transformers
3. **Clustering**: Similar files are grouped together using K-means clustering
4. **Category Suggestion**: The Mistral 7B model suggests meaningful category names
5. **Organization**: Files are moved to their respective category directories

## Project Structure