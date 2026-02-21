import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco


class TratamentoComentariosCorrente(Corrente):

    def __init__(self, operacoes_banco: IoperacoesBanco):
        super().__init__()
        self.__banco_analitico = operacoes_banco
        self.__camino_consulta_comentarios = f's3://extracao/youtube/bronze/comentarios_youtube/*/*/comentarios.json'

    def __obter_dataset_comentarios(self) -> pd.DataFrame:
        dataframe_comentarios = self.__banco_analitico.consultar_dados('1=1', self.__camino_consulta_comentarios)
        df_snippet_comentarios = pd.json_normalize(
            dataframe_comentarios['snippet'].tolist(),
            sep='_'
        )

        df_comentarios_final = df_snippet_comentarios[
            ['channelId', 'videoId', 'topLevelComment_id', 'topLevelComment_snippet_textDisplay']]

        df_comentarios_final.rename(
            columns={
                "channelId": 'id_canal',
                "videoId": "id_video",
                "topLevelComment_id": "id_comentario",
                "topLevelComment_snippet_textDisplay": "texto_comentario"

            },
            inplace=True
        )
        return df_comentarios_final

    def __obter_dataset_resposta_comentarios(self, dataset: pd.DataFrame):


    def executar_processo(self, contexto: Contexto) -> bool:
        df_comentarios = self.__obter_dataset_comentarios()
        print(df_comentarios)
        return True
