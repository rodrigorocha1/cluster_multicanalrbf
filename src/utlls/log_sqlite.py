import logging
import os

from src.utlls.sqlite_handler import SQLiteHandler

DB_PATH = os.path.join(os.getcwd(), 'logs', 'logs.db')

logger = logging.getLogger("meu_logger")
logger.setLevel(logging.DEBUG)

handler = SQLiteHandler(DB_PATH)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)

