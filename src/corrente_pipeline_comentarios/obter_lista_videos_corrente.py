from itertools import chain
from typing import List, Generator

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.banco.servico_sqlite import SQLiteDB
from src.utlls.log_sqlite import logger


class ObterListaVideosCorrente(Corrente):

    def __init__(self, lista_canais: List[str], servico_youtube: IApiYoutube):
        super().__init__()
        self.__lista_canais = lista_canais
        self.__servico_youtube = servico_youtube
        self.__servico_banco = SQLiteDB()

    def __verificar_video_inserido(self, id_video: str) -> bool:
        lista_videos = self.__servico_banco.buscar(
            nome_tabela="videos",
            where=f"id_video = '{id_video}'"

        )
        if lista_videos:
            logger.info(f'Video {id_video} já foi inserido')
            return True
        logger.info(f'Video {id_video}  inserido')
        return False

    def __inserir_lista_videos(self, lista_id_videos: List[Generator[tuple[str, str], None, None]]):
        for id_video, titulo_video in chain.from_iterable(lista_id_videos):
            logger.info(f'Inserindo video {id_video} - {titulo_video}')
            flag = self.__verificar_video_inserido(id_video)
            if not flag:
                self.__servico_banco.inserir(
                    nome_tabela="videos",
                    colunas="id_video, titulo_video",
                    valores=(id_video, titulo_video)
                )

    def executar_processo(self, contexto: Contexto) -> bool:
        lista_id_videos = [
            self.__servico_youtube.obter_video_por_data(id_canal=id_canal, data_inicio=contexto.data_publicacao) for
            id_canal in
            contexto.lista_canais
        ]
        self.__inserir_lista_videos(lista_id_videos)

        return True
