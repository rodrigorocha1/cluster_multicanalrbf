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
order by 3 DESC ;


SELECT DATETIME('now', 'localtime', '-1 day');


CREATE table videos(
	id_video text primary key,
	titulo_video text 
)


create TABLE  param_ultima_data_busca(
	paramento text,
	data_inicio DATETIME
)

SELECT *
FROM videos;

truncate table videos

select *
FROM param_ultima_data_busca;


INSERT INTO
param_ultima_data_busca
VALUES('ultima_data_publicacao_video','2026-02-19 21:39:01')

