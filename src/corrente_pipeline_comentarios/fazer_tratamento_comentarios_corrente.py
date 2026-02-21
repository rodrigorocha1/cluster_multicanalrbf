
from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.estrategia_tratamento.processador_texto import ProcessadorTexto
from src.servicos.estrategia_tratamento.tratamento_simples import TratamentoSimples
import numpy as np

from src.servicos.estrategia_tratamento.tratamento_spacy import TratamentoSpacy


class FazerTratamentoComentariosCorrente(Corrente):

    def __init__(self, operacoes_banco: IoperacoesBanco):
        super().__init__()
        self.__servico_banco_analitico = operacoes_banco
        self.__estrategia = ProcessadorTexto()

    def executar_processo(self, contexto: Contexto) -> bool:
        base_original = contexto.dataframe_prata
        self.__estrategia.estrategia = TratamentoSimples()
        vetorizar_tratamento = np.vectorize(self.__estrategia.processar)
        base_original['texto_comentario_tratado'] = vetorizar_tratamento(base_original['texto_comentario'])
        self.__estrategia.estrategia = TratamentoSpacy()
        tokens_list, entidades_list, comentarios_limpos, embeddings_list = self.__estrategia.processar(
            base_original['texto_comentario_tratado'].tolist()
        )
        base_original['tokens'] = tokens_list
        base_original['entidades'] = entidades_list
        base_original['comentario_limpo'] = comentarios_limpos
        base_original['embeddings'] = embeddings_list

        df_copy = base_original.copy()

        # Converte listas/tuplas em JSON ou string
        for col in ['tokens', 'entidades', 'embeddings']:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: str(x))

        print(df_copy.dtypes)

        self.__servico_banco_analitico.guardar_dados(df_copy)

        return True
