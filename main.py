import datetime
import os

from datetime import datetime
from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.obter_lista_comentarios_corrente import ObterListaComentariosCorrente
from src.corrente_pipeline_comentarios.obter_ultima_data_publicacao_corrente import ObterUltimaDataPublicacaoCorrente
from src.servicos.api_youtube.api_youtube import YoutubeAPI

contexto = Contexto( data_publicacao=datetime.now())
caminho_banco = os.path.join(os.getcwd(), 'logs', 'logs.db')
lista_canais = ['@jogatinaepica', '@ChratosGameplay']
servico_youtube = YoutubeAPI()
p1 = ObterUltimaDataPublicacaoCorrente(caminho_banco=caminho_banco)
p2 = ObterListaComentariosCorrente(lista_canais=lista_canais, servico_youtube=servico_youtube)
p1.set_proxima_corrente(p2)
p1.corrente(contexto=contexto)


