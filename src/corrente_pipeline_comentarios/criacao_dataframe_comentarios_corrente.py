from typing import List

import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco


class CriacaoDataframeComentariosCompletoCorrente(Corrente):

    def __init__(self, operacoes_banco: IoperacoesBanco):
        super().__init__()
        self.__banco_analitico = operacoes_banco
        self.__camino_consulta_comentarios = f's3://extracao/youtube/bronze/comentarios_youtube/*/*/comentarios.json'
        self.__caminho_consulta_resposta_comentarios = f's3://extracao/youtube/bronze/resposta_comentarios_youtube/*/*/*/resposta_comentarios.json'

    def __obter_dataset_comentarios(self) -> pd.DataFrame:
        dataframe_comentarios = self.__banco_analitico.consultar_dados('1=1', self.__camino_consulta_comentarios)
        df_snippet_comentarios = pd.json_normalize(
            dataframe_comentarios['snippet'].tolist(),
            sep='_'
        )

        df_comentarios_final = df_snippet_comentarios[
            ['channelId', 'videoId', 'topLevelComment_id', 'topLevelComment_snippet_textDisplay']]

        df_comentarios_final = df_comentarios_final.rename(
            columns={
                "channelId": 'id_canal',
                "videoId": "id_video",
                "topLevelComment_id": "id_comentario",
                "topLevelComment_snippet_textDisplay": "texto_comentario"

            },

        )
        return df_comentarios_final

    def __obter_dataset_resposta_comentarios(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataframe_reposta_comentarios = self.__banco_analitico.consultar_dados(
            '1=1',
            self.__caminho_consulta_resposta_comentarios
        )
        records: List = dataframe_reposta_comentarios[['snippet', 'id']].fillna(
            {'snippet': {}}).to_dict(orient='records')
        df_snippet_resposta_comentarios = pd.json_normalize(records, sep='_')

        df_snippet_resposta_comentarios['id_comentario'] = df_snippet_resposta_comentarios['id'].str.split('.').str[1]

        df_resposta_comentarios_final = pd.merge(
            dataset[['id_video', 'id_comentario']],
            df_snippet_resposta_comentarios,
            left_on='id_comentario',
            right_on='snippet_parentId',
            how='inner'
        )
        df_resposta_comentarios_final = df_resposta_comentarios_final[
            ['snippet_channelId', 'id_video', 'id_comentario_x', 'snippet_textDisplay']]

        df_resposta_comentarios_final = df_resposta_comentarios_final.rename(columns={
            'snippet_channelId': 'id_canal',
            'videoId': 'id_video',
            'snippet_textDisplay': 'texto_comentario',
            'id_comentario_x': 'id_comentario'
        })
        return df_resposta_comentarios_final

    @staticmethod
    def __unir_dataset(**kwargs) -> pd.DataFrame:
        df_comentarios_final = kwargs['df_comentarios_final']
        df_resposta_comentarios_final = kwargs['df_resposta_comentarios_final']
        df_comentarios_tratado_final = pd.concat([df_resposta_comentarios_final, df_comentarios_final])
        return df_comentarios_tratado_final

    def executar_processo(self, contexto: Contexto) -> bool:
        df_comentarios = self.__obter_dataset_comentarios()
        df_resposta_comentarios = self.__obter_dataset_resposta_comentarios(dataset=df_comentarios)
        dataset_comentarios_tratado = self.__unir_dataset(
            df_comentarios_final=df_comentarios,
            df_resposta_comentarios_final=df_resposta_comentarios
        )
        contexto.dataframe_prata = dataset_comentarios_tratado


        return True
