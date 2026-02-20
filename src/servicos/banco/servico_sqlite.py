import os.path
import sqlite3
from typing import List, Tuple, Optional


class SQLiteDB:
    def __init__(self, db_path: str = os.path.join(os.getcwd(), 'logs', 'logs.db')):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._conectar()

    def _conectar(self):
        print(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.commit()

    def inserir(self, nome_tabela: str, colunas: str, valores: Tuple):
        """
        Insere um registro na tabela.
        :param nome_tabela: Nome da tabela
        :param colunas: Colunas separadas por vírgula, ex: 'id_canal
        :param valores: Tupla com os valores a inserir, ex: ('ID_CANAL')
        """
        placeholders = ",".join("?" * len(valores))
        sql = f"INSERT INTO {nome_tabela} ({colunas}) VALUES ({placeholders})"
        self.conn.execute(sql, valores)
        self.conn.commit()

    def buscar(self, nome_tabela: str, colunas: str = "*", where: str = "") -> List[Tuple]:
        """
        Busca registros na tabela.
        :param nome_tabela: Nome da tabela
        :param colunas: Colunas a selecionar
        :param where: Condição WHERE opcional, ex: "idade > 25"
        :return: Lista de tuplas com os registros
        """
        sql = f"SELECT {colunas} FROM {nome_tabela}"
        if where:
            sql += f" WHERE {where}"
        cursor = self.conn.execute(sql)
        return cursor.fetchall()

    def fechar(self):
        """Fecha a conexão com o banco."""
        if self.conn:
            self.conn.close()
            self.conn = None
