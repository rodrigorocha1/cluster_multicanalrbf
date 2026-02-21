import sqlite3

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente


class FimCadeia(Corrente):

    def __init__(self, caminho_banco: str) -> None:
        super().__init__()
        self.__caminho_banco = caminho_banco

    def executar_processo(self, contexto: Contexto) -> bool:
        try:
            with sqlite3.connect(self.__caminho_banco) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                   UPDATE param_ultima_data_busca
                    SET data_inicio = DATETIME('now', 'localtime', '-1 day')
                    WHERE paramento = 'ultima_data_publicacao_video'
                    """
                )
                conn.commit()

                return True

        except sqlite3.Error as e:
            print(f"Erro ao buscar última data: {e}")
            return False
