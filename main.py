import os
from datetime import datetime

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.guardar_comentarios_s3_corrente import GuardarComentariosS3Corrente
from src.corrente_pipeline_comentarios.obter_lista_canais_corrente import ObterListaCanaisCorrente
from src.corrente_pipeline_comentarios.obter_lista_comentarios_corrente import ObterListacomentariosCorrente
from src.corrente_pipeline_comentarios.obter_lista_videos_corrente import ObterListaVideosCorrente
# from src.corrente_pipeline_comentarios.obter_lista_canais_corrente import ObterListaComentariosCorrente
from src.corrente_pipeline_comentarios.obter_ultima_data_publicacao_corrente import ObterUltimaDataPublicacaoCorrente
from src.servicos.api_youtube.api_youtube import YoutubeAPI
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb
from src.servicos.servico_s3.sevicos3 import ServicoS3

contexto = Contexto(data_publicacao=datetime.now())
caminho_banco = os.path.join(os.getcwd(), 'logs', 'logs.db')
lista_canais = ['@jogatinaepica']
servico_youtube = YoutubeAPI()
servico_s3 = ServicoS3()
servico_banco_analitico = OperacoesBancoDuckDb()
p1 = ObterUltimaDataPublicacaoCorrente(caminho_banco=caminho_banco)
p2 = ObterListaCanaisCorrente(lista_canais=lista_canais, servico_youtube=servico_youtube)
p3 = ObterListaVideosCorrente(lista_canais=lista_canais, servico_youtube=servico_youtube, servico_s3=servico_s3)
p4 = ObterListacomentariosCorrente(servico_youtube=servico_youtube)
p5 = GuardarComentariosS3Corrente(servico_youtube=servico_youtube, servico_banco_analitico=servico_banco_analitico, servico_s3=servico_s3)
p1.set_proxima_corrente(p2).set_proxima_corrente(p3).set_proxima_corrente(p4)
p1.corrente(contexto=contexto)
