from typing import List

from src.api_youtube.iapi_youtube import IApiYoutube
from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente


class ObterListaVideo(Corrente):

    def __init__(self, lista_canal: List[str], servico_youtube: IApiYoutube):
        super().__init__()
        self.__lista_canal = lista_canal
        self.__servico_youtube = servico_youtube



    def buscar_id_canais(self) -> List[str]:
        self.__servico_youtube.

    def executar_processo(self, contexto: Contexto) -> bool:
        return True
