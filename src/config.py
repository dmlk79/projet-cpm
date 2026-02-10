"""Configuration globale du projet ASR"""
import torch
from pathlib import Path
import os
from loguru import logger

# Setup logging
LOG_DIR = Path.home() / "projet-cpm" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "asr_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

def get_device():
    """Détecte et configure le device optimal"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"GPU détecté: {gpu_name}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        
        return device
    else:
        logger.warning("GPU non disponible, utilisation CPU")
        return torch.device("cpu")

# Configuration
DEVICE = get_device()
PROJECT_ROOT = Path.home() / "projet-cpm"
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
MODELS_DIR = DATA_DIR / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Model config
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
BATCH_SIZE = 12
USE_FP16 = True

# Cache HuggingFace
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
os.environ['HF_HOME'] = str(CACHE_DIR)

logger.info(f"Configuration chargée - Device: {DEVICE}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Corpus: {CORPUS_DIR}")
