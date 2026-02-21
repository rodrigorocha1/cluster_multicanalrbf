import os
from datetime import datetime

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.criacao_dataframe_comentarios_corrente import \
    CriacaoDataframeComentariosCompletoCorrente
from src.corrente_pipeline_comentarios.fazer_tratamento_comentarios_corrente import FazerTratamentoComentariosCorrente
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb

contexto = Contexto(data_publicacao=datetime.now())
caminho_banco = os.path.join(os.getcwd(), 'logs', 'logs.db')

servico_banco_analitico = OperacoesBancoDuckDb()

p1 = CriacaoDataframeComentariosCompletoCorrente(operacoes_banco=servico_banco_analitico)
p2 = FazerTratamentoComentariosCorrente(
    operacoes_banco=servico_banco_analitico
)
p1.set_proxima_corrente(p2) \
#     .set_proxima_corrente(p3) \
#     .set_proxima_corrente(p4) \
#     .set_proxima_corrente(p5) \
#     .set_proxima_corrente(p6) \
#     .set_proxima_corrente(p7) \
#     .set_proxima_corrente(p8)
p1.corrente(contexto=contexto)
