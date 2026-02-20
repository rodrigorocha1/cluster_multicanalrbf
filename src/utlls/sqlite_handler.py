import logging
import sqlite3
from datetime import datetime
from typing import Optional


class SQLiteHandler(logging.Handler):

    def __init__(self, db_path: str):
        super().__init__()
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            db_path,
            check_same_thread=False
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._conn is None:
                return

            message = self.format(record)

            exception_text = None
            if record.exc_info:
                if self.formatter:
                    exception_text = self.formatter.formatException(record.exc_info)
                else:
                    exception_text = logging.Formatter().formatException(record.exc_info)

            self._conn.execute(
                """
                INSERT INTO logs (
                    created,
                    data_registro,
                    level,
                    logger,
                    module,
                    funcName,
                    lineNo,
                    message,
                    exception,
                    process,
                    thread,
                    url,
                    requisicao,
                    codigo
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.created,
                    datetime.now(),
                    record.levelname.upper(),
                    record.name,
                    record.module,
                    record.funcName,
                    record.lineno,
                    message,
                    exception_text,
                    record.process,
                    record.thread,
                    getattr(record, "url", None),
                    getattr(record, "requisicao", None),
                    getattr(record, "codigo", None),
                ),
            )

            self._conn.commit()

        except Exception:
            self.handleError(record)


if __name__ == "__main__":
    import logging

    from src.utlls.sqlite_handler import SQLiteHandler
    import os

    DB_PATH = os.path.join(os.getcwd(), 'logs', 'logs.db')

    logger = logging.getLogger("meu_logger")
    logger.setLevel(logging.DEBUG)

    handler = SQLiteHandler(DB_PATH)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Log simples
    logger.info(
        "Requisição iniciada",
        extra={
            "url": "/api/v1/clientes",
            "requisicao": "GET",
            "codigo": 200
        }
    )

    # Log com exception
    try:
        10 / 0
    except ZeroDivisionError:
        logger.exception(
            "Erro interno na API",
            extra={
                "url": "/api/v1/clientes",
                "requisicao": "GET",
                "codigo": 500
            }
        )

    print("Logs gravados com sucesso.")
