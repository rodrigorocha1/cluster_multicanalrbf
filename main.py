import os

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.obter_lista_video_corrente import ObterListaVideoCorrente
from src.corrente_pipeline_comentarios.obter_ultima_data_publicacao_corrente import ObterUltimaDataPublicacaoCorrente
from src.servicos.api_youtube.api_youtube import YoutubeAPI

contexto = Contexto()
caminho_banco = os.path.join(os.getcwd(), 'logs', 'logs.db')
lista_canais = ['@jogatinaepica']
servico_youtube = YoutubeAPI()
p1 = ObterUltimaDataPublicacaoCorrente(caminho_banco=caminho_banco)
p2 = ObterListaVideoCorrente(lista_canais=lista_canais, servico_youtube=servico_youtube)
p1.set_proxima_corrente(p2)
p1.corrente(contexto=contexto)
