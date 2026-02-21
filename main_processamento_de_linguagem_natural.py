import os
from datetime import datetime

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.fim_cadeia import FimCadeia
from src.corrente_pipeline_comentarios.guardar_comentarios_s3_corrente import GuardarComentariosS3Corrente
from src.corrente_pipeline_comentarios.guardar_resposta_comentarios_s3_corrente import \
    GuardarDadosYoutubeRespostaComentariosS3Corrente
from src.corrente_pipeline_comentarios.obter_lista_canais_corrente import ObterListaCanaisCorrente
from src.corrente_pipeline_comentarios.obter_lista_comentarios_corrente import ObterListacomentariosCorrente
from src.corrente_pipeline_comentarios.obter_lista_resposta_comentarios_corrente import \
    ObterListaRespostaComentariosCorrente
from src.corrente_pipeline_comentarios.obter_lista_videos_corrente import ObterListaVideosCorrente
from src.corrente_pipeline_comentarios.obter_ultima_data_publicacao_corrente import ObterUltimaDataPublicacaoCorrente
from src.corrente_pipeline_comentarios.criacao_dataframe_comentarios_corrente import CriacaoDataframeComentariosCompletoCorrente
from src.servicos.api_youtube.api_youtube import YoutubeAPI
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb
from src.servicos.servico_s3.sevicos3 import ServicoS3

contexto = Contexto(data_publicacao=datetime.now())
caminho_banco = os.path.join(os.getcwd(), 'logs', 'logs.db')

servico_banco_analitico = OperacoesBancoDuckDb()

p1 = CriacaoDataframeComentariosCompletoCorrente(operacoes_banco=servico_banco_analitico)
# p1.set_proxima_corrente(p2) \
#     .set_proxima_corrente(p3) \
#     .set_proxima_corrente(p4) \
#     .set_proxima_corrente(p5) \
#     .set_proxima_corrente(p6) \
#     .set_proxima_corrente(p7) \
#     .set_proxima_corrente(p8)
p1.corrente(contexto=contexto)
