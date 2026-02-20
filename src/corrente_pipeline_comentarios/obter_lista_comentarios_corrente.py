from typing import List, Tuple

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.banco.servico_sqlite import SQLiteDB


class ObterListacomentariosCorrente(Corrente):

    def __init__(self, servico_youtube: IApiYoutube):
        super().__init__()
        self.__servico_banco = SQLiteDB()
        self.__servico_youtube = servico_youtube

    def __buscar_videos(self) -> List[Tuple[str, ...]]:
        lista_id_video = self.__servico_banco.buscar(
            nome_tabela='videos',
            where='1 = 1',
            colunas='id_video'

        )
        print(lista_id_video)
        print(type(lista_id_video))
        return lista_id_video

    def executar_processo(self, contexto: Contexto) -> bool:
        for video in self.__buscar_videos():
            print(video[0])
        return True
