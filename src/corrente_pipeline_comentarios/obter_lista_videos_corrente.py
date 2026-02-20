from typing import List
from datetime import datetime
from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterListaVideosCorrente(Corrente):

    def __init__(self, lista_canais: List[str], servico_youtube: IApiYoutube):
        super().__init__()
        self.__lista_canais = lista_canais
        self.__servico_youtube = servico_youtube




    def executar_processo(self, contexto: Contexto) -> bool:
        lista_id_videos = [
            self.__servico_youtube.obter_video_por_data(id_canal=id_canal, data_inicio=contexto.data_publicacao) for id_canal in
            contexto.lista_canais
        ]


        contexto.lista_video = lista_id_videos
        return True
