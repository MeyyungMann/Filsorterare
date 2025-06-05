from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CATEGORIZER_DIR = PROJECT_ROOT / "categorizer"
UTILS_DIR = PROJECT_ROOT / "utils"

# Model settings
MODEL_NAME = "mistral-7b"  # Local model name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32

# Clustering settings
MIN_CLUSTER_SIZE = 3
CLUSTER_THRESHOLD = 0.7

# File processing settings
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.py', '.js', '.html', '.css', '.json',
    '.xml', '.yaml', '.yml', '.csv', '.tsv'
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Local Mistral 7B model path
LOCAL_MISTRAL_PATH = os.getenv("LOCAL_MISTRAL_PATH", r"C:\Users\meyyu\Desktop\mistral_7") 