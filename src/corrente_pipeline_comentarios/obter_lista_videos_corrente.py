from itertools import chain
from typing import List

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
        logger.info(f'inserindo {id_video}  no banco')

        return False

    def __inserir_lista_videos(self, id_video: str, titulo_video: str):

        logger.info(f'Inserindo video {id_video} - {titulo_video}')
        flag = self.__verificar_video_inserido(id_video)
        if not flag:
            self.__servico_banco.inserir(
                nome_tabela="videos",
                colunas="id_video, titulo_video",
                valores=(id_video, titulo_video)
            )


    def executar_processo(self, contexto: Contexto) -> bool:
        if contexto.data_publicacao is None:
            raise ValueError("data_publicacao não pode ser None neste estágio do pipeline")
        lista_id_videos = [
            self.__servico_youtube.obter_video_por_data(id_canal=id_canal, data_inicio=contexto.data_publicacao) for
            id_canal in
            contexto.lista_canais
        ]
        print(lista_id_videos)

        for dados_videos in chain.from_iterable(lista_id_videos):
            try:

                id_video = dados_videos['id']['videoId']
                titulo_video = dados_videos['snippet']['title']
                self.__inserir_lista_videos(id_video, titulo_video)
            except Exception as e:
                logger.error(f'Erro ao buscar vídeo', extra=
                {
                    'requisicao': dados_videos,
                    'exception': str(e)
                })

                continue

        self.__servico_banco.fechar_conexao()

        return True
