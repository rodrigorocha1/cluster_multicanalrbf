import os
import sqlite3
from typing import List, Tuple


class SQLiteDB:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.path.join(os.getcwd(), "logs", "logs.db")

        self.db_path = db_path
        self.conn: sqlite3.Connection = self._conectar()

    def _conectar(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        return conn

    def inserir(self, nome_tabela: str, colunas: str, valores: Tuple) -> None:
        placeholders = ",".join("?" * len(valores))
        sql = f"INSERT INTO {nome_tabela} ({colunas}) VALUES ({placeholders})"
        self.conn.execute(sql, valores)
        self.conn.commit()

    def buscar(
        self,
        nome_tabela: str,
        colunas: str = "*",
        where: str = "",
    ) -> List[Tuple]:

        sql = f"SELECT {colunas} FROM {nome_tabela}"
        if where:
            sql += f" WHERE {where}"

        cursor = self.conn.execute(sql)
        return cursor.fetchall()

    def executar(self, sql: str, parametros: Tuple = ()) -> None:
        self.conn.execute(sql, parametros)
        self.conn.commit()

    def fechar_conexao(self) -> None:
        self.conn.close()