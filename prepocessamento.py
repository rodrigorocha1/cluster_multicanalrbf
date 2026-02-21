import pandas as pd
import numpy as np
from src.servicos.estrategia_tratamento.tratamento_simples import TratamentoSimples
from src.servicos.estrategia_tratamento.tratamento_spacy import TratamentoSpacy
tratamento_simples = TratamentoSimples()
tratamamento_spacy = TratamentoSpacy()

base_original = pd.read_csv('df_comentarios_tratado_final.csv', sep='|')

vetorizar_tratamento = np.vectorize(tratamento_simples.executar_tratamento)
base_original['texto_comentário_tratado'] = vetorizar_tratamento(base_original['texto_comentario'])


tokens_list, entidades_list, comentarios_limpos, embeddings_list = tratamamento_spacy.executar_tratamento(
    base_original['texto_comentário_tratado'].tolist()
    )


base_original['tokens'] = tokens_list
base_original['entidades'] = entidades_list
base_original['comentario_limpo'] = comentarios_limpos
base_original['embeddings'] = embeddings_list
base_original.to_csv('dados_prata.csv', sep='|')