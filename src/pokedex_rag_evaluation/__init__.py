import logging
import os

from dotenv import load_dotenv

from pokedex_rag_evaluation.utils import init_logging

logger = logging.getLogger(__name__)

if os.path.exists("config/logging.yaml"):
    init_logging("config/logging.yaml")
    logger.info("Logging initialized")
else:
    logger.error("Logging config not found")

if os.path.exists("config/.env"):
    load_dotenv("config/.env")
    logger.info("Environment variables loaded")
else:
    logger.error("Environment variables not found")
