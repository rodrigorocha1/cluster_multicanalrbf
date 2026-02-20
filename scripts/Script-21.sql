CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    created REAL NOT NULL,

    data_registro DATETIME NOT NULL ,

    level TEXT NOT NULL CHECK (
        level IN ('DEBUG','INFO','WARNING','ERROR','CRITICAL','NOTSET')
    ),

    logger TEXT NOT NULL,
    module TEXT NOT NULL,
    funcName TEXT NOT NULL,
    lineNo INTEGER NOT NULL,

    message TEXT,

    exception TEXT,

    process INTEGER,
    thread INTEGER,

    url TEXT,
    requisicao TEXT,
    codigo INTEGER
);


CREATE INDEX IF NOT EXISTS idx_logs_data_registro
ON logs (data_registro);

DROP TABLE logs;

select *
FROM logs

