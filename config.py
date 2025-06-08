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

# Local Mistral 7B model path
LOCAL_MISTRAL_PATH = os.getenv("LOCAL_MISTRAL_PATH")
if not LOCAL_MISTRAL_PATH:
    raise ValueError("LOCAL_MISTRAL_PATH environment variable is not set. Please check your .env file.") 