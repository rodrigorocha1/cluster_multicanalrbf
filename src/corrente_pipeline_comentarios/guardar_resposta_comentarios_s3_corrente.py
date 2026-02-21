from datetime import datetime

import duckdb
import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.config.configuracao_log import logger
from src.servicos.servico_s3.iservicos3 import Iservicos3


class GuardarDadosYoutubeRespostaComentariosS3Corrente(Corrente):

    def __init__(self, servico_banco_analitico: IoperacoesBanco, servico_s3: Iservicos3):
        super().__init__()
        self.__servico_s3 = servico_s3
        self.__caminho_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.__caminho_arquivo = f'youtube/bronze/resposta_comentarios_youtube'
        self.__servico_banco = servico_banco_analitico

    def executar_processo(self, contexto: Contexto) -> bool:

        for dados in contexto.lista_req_resp_comentarios:
            for req_resposta_comentarios in dados[3]:
                print(dados)

                logger.info(f'Guardando json do canal {dados[0]}')
                id_resposta_comentarios = req_resposta_comentarios['id']
                id_comentario = dados[2]
                data_atualizacao_api = req_resposta_comentarios['snippet']['updatedAt']
                caminho_base = f"{self.__caminho_arquivo}/*/*/*/resposta_comentarios.json"

                condicao = f"id = '{id_resposta_comentarios}' AND snippet.updatedAt = '{data_atualizacao_api}'"
                caminho_consulta = f"s3://extracao/{caminho_base}"
                try:
                    dataframe = self.__servico_banco.consultar_dados(caminho_consulta=caminho_consulta,
                                                                     id_consulta=condicao)
                except duckdb.IOException as e:
                    logger.error(f'{e}')
                    logger.info(f'{id_comentario}  error atualizacao')
                    dataframe = pd.DataFrame()

                if dataframe.empty:
                    logger.info(f'{id_comentario}  teve atualizacao')
                    caminho_arquivo = f"{self.__caminho_arquivo}/id_canal_{dados[0]}/id_video_{dados[1]}/id_comentario_{dados[2]}/resposta_comentarios.json"

                    self.__servico_s3.guardar_dados(req_resposta_comentarios, caminho_arquivo)
                else:
                    logger.info(f'{id_comentario} não teve atualizacao')


        return True
