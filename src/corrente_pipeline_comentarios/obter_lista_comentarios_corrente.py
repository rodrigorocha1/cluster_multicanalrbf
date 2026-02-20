from itertools import chain

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterListacomentariosCorrente(Corrente):

    def __init__(self, servico_youtube: IApiYoutube):
        super().__init__()

        self.__servico_youtube = servico_youtube

    def __buscar_comentarios_videos_antigos(self):
        return ['a', 'b']

    def __buscar_videos_recentes(self):
        pass

    def executar_processo(self, contexto: Contexto) -> bool:

        return True
