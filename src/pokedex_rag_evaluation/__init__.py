import logging
from pathlib import Path

from dotenv import load_dotenv

from pokedex_rag_evaluation.utils import init_logging

logger = logging.getLogger(__name__)

if Path("config/logging.yaml").exists():
    init_logging("config/logging.yaml")
    logger.info("Logging initialized")
else:
    logger.error("Logging config not found")

if Path("config/.env").exists():
    load_dotenv("config/.env")
    logger.info("Environment variables loaded")
else:
    logger.error("Environment variables not found")
