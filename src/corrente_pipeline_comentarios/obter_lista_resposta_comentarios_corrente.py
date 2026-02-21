from itertools import chain

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.banco.servico_sqlite import SQLiteDB


class ObterListaRespostaComentariosCorrente(Corrente):

    def __init__(self, servico_youtube: IApiYoutube):
        super().__init__()
        self.__servico_banco = SQLiteDB()
        self.__servico_youtube = servico_youtube

    def executar_processo(self, contexto: Contexto) -> bool:

        for dados_resposta_comentarios in contexto.lista_id_comentarios:

            id_canal = dados_resposta_comentarios[0]
            id_video = dados_resposta_comentarios[1]
            id_comentario = dados_resposta_comentarios[2]
            req_resp_comentarios = self.__servico_youtube.obter_resposta_comentarios(id_comentario)
            contexto.lista_req_resp_comentarios.append((id_canal, id_video, id_comentario, req_resp_comentarios))




        return True
