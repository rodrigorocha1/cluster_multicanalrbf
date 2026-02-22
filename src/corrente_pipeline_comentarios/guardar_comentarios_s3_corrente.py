from typing import List, Tuple

import duckdb
import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.banco.servico_sqlite import SQLiteDB
from src.servicos.servico_s3.iservicos3 import Iservicos3
from src.utlls.log_sqlite import logger


class GuardarComentariosS3Corrente(Corrente):

    def __init__(self, servico_youtube: IApiYoutube, servico_banco_analitico: IoperacoesBanco, servico_s3: Iservicos3):
        super().__init__()
        self.__servico_banco = SQLiteDB()
        self.__servico_youtube = servico_youtube
        self.__servico_banco_analitico = servico_banco_analitico
        self.__servico_s3 = servico_s3
        self.__caminho_arquivo = f'youtube/bronze/comentarios_youtube'

    def __buscar_videos(self) -> List[Tuple[str, ...]]:
        lista_id_video = self.__servico_banco.buscar(
            nome_tabela='videos',
            where='1 = 1',
            colunas='id_video'

        )
        return lista_id_video

    def executar_processo(self, contexto: Contexto) -> bool:
        for comentario in contexto.lista_req_comentarios:
            print('=' * 20, 'ínicio', '=' * 20)

            id_canal = comentario['snippet']['channelId']
            id_video = comentario['snippet']['videoId']
            id_comentario = comentario['id']
            data_atualizacao_api = comentario['snippet']['topLevelComment']['snippet']['updatedAt']
            condicao = f"id = '{id_comentario}' AND snippet.topLevelComment.snippet.updatedAt = '{data_atualizacao_api}'"
            caminho_base = f"{self.__caminho_arquivo}/*/*/comentarios.json"
            caminho_consulta = f"s3://extracao/{caminho_base}"
            try:
                dataframe = self.__servico_banco_analitico.consultar_dados(caminho_consulta=caminho_consulta,
                                                                           id_consulta=condicao)
            except duckdb.IOException as e:
                dataframe = pd.DataFrame()
            if dataframe.empty:
                logger.info(f'Guardando   comentários: {id_comentario} ')

                caminho_completo = f"{self.__caminho_arquivo}/id_canal_{id_canal}/id_video_{id_video}/comentarios.json"
                self.__servico_s3.guardar_dados(comentario, caminho_completo)
            else:
                logger.info(f' comentário: {id_comentario} não teve atualizacao')
            contexto.lista_id_comentarios.append(
                (id_canal, id_video, id_comentario)
            )
            print('=' * 20, 'fim', '=' * 20)
        return True
