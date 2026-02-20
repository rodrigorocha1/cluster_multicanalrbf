from typing import List

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterListCanaisCorrente(Corrente):

    def __init__(self, lista_canais: List[str], servico_youtube: IApiYoutube):
        super().__init__()
        self.__lista_canais = lista_canais
        self.__servico_youtube = servico_youtube

    def __buscar_id_canais(self) -> List[str]:
        lista_canais_resultado = [
            self.__servico_youtube.obter_id_canal(canal)[0] for canal in self.__lista_canais
        ]
        return lista_canais_resultado

    def executar_processo(self, contexto: Contexto) -> bool:
        lista_canais = self.__buscar_id_canais()

        contexto.lista_canais = lista_canais

        return True
