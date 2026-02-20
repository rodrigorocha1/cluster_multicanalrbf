from typing import List

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterListaVideo(Corrente):

    def __init__(self, lista_canal: List[str], servico_youtube: IApiYoutube):
        super().__init__()
        self.__lista_canal = lista_canal
        self.__servico_youtube = servico_youtube

    def __buscar_id_canais(self) -> List[str]:
        lista_canais_resultado = [
            self.__servico_youtube.obter_id_canal(canal)[0] for canal in self.__lista_canal
        ]
        return lista_canais_resultado

    def __buscar_video(self):
        pass

    def executar_processo(self, contexto: Contexto) -> bool:
        return True
