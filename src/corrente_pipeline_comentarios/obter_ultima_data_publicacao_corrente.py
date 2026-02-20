from datetime import datetime, timezone
import sqlite3
from typing import Optional

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente


class ObterUltimaDataPublicacaoCorrente(Corrente):

    def __init__(self, caminho_banco: str) -> None:
        super().__init__()
        self.__caminho_banco = caminho_banco

    def executar_processo(self, contexto: Contexto) -> bool:
        try:
            with sqlite3.connect(self.__caminho_banco) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_inicio
                    FROM param_ultima_data_busca
                    where paramento = 'ultima_data_publicacao_video'
                    LIMIT 1
                    """
                )

                resultado: Optional[tuple[str]] = cursor.fetchone()
                data_publicacao = datetime.strptime(resultado[0], "%Y-%m-%d %H:%M:%S") \
                    .replace(tzinfo=timezone.utc)




                if resultado is None:
                    return False

                contexto.data_publicacao = data_publicacao
                return True

        except sqlite3.Error as e:
            print(f"Erro ao buscar última data: {e}")
            return False