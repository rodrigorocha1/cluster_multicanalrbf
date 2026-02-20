from typing import List
from datetime import datetime
from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterListaComentariosCorrente(Corrente):

    def __init__(self, lista_canais: List[str], servico_youtube: IApiYoutube):
        super().__init__()
        self.__lista_canais = lista_canais
        self.__servico_youtube = servico_youtube

    def __buscar_id_canais(self) -> List[str]:
        lista_canais_resultado = [
            self.__servico_youtube.obter_id_canal(canal)[0] for canal in self.__lista_canais
        ]
        return lista_canais_resultado

    def __buscar_video(self, lista_id_canal: List[str], data_publicacao: datetime):
        lista_id_videos = [
            self.__servico_youtube.obter_video_por_data(id_canal=id_canal, data_inicio=data_publicacao) for id_canal in lista_id_canal
        ]
        return lista_id_videos

    def executar_processo(self, contexto: Contexto) -> bool:
        lista_canais = self.__buscar_id_canais()
        for video in self.__buscar_video(lista_id_canal=lista_canais, data_publicacao=contexto.data_publicacao):
            for dado in video:
                print(dado)
        return True
