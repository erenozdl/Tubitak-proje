"""
Logging configuration.

Sets up a logger that writes to both the console and a rotating log file
under the logs/ directory. Each log entry includes the sender, timestamp,
and the classification category for easy auditing.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "email_service.log")

# 5 MB per file, keep last 3 rotated copies
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3


def get_logger(name: str = "email_service") -> logging.Logger:
    """
    Return a configured logger instance.

    The logger writes INFO+ to a rotating file and to stdout so you can
    monitor the service in the terminal while also keeping a persistent log.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── File handler (rotating) ──────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
