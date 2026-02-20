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

