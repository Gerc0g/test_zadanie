from loguru import logger

from config import settings

logger.add(
    "logs/debug.log",
    format="{time} {level} {message}",
    level=settings.LOG_LEVEL,
    rotation="250 KB",
    compression="zip"
)