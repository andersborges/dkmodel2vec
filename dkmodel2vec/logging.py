import logging
import psutil
import os
from logging import getLogger

logger = getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            logging.FileHandler("training.log"),  # File
        ],
    )


def log_memory_usage(step: str):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024 / 1024
    logger.info(f"Memory usage at {step}: {memory_mb:.1f} GB")
