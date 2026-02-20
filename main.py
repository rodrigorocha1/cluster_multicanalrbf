import datetime
import os

from datetime import datetime
from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.obter_lista_comentarios_corrente import ObterListaComentariosCorrente
from src.corrente_pipeline_comentarios.obter_ultima_data_publicacao_corrente import ObterUltimaDataPublicacaoCorrente
from src.servicos.api_youtube.api_youtube import YoutubeAPI

contexto = Contexto( data_publicacao=datetime.now())
caminho_banco = os.path.join(os.getcwd(), 'logs', 'logs.db')
lista_canais = ['@jogatinaepica']
servico_youtube = YoutubeAPI()
p1 = ObterUltimaDataPublicacaoCorrente(caminho_banco=caminho_banco)
p2 = ObterListaComentariosCorrente(lista_canais=lista_canais, servico_youtube=servico_youtube)
p1.set_proxima_corrente(p2)
p1.corrente(contexto=contexto)


# from datetime import datetime, timezone
#
# from src.servicos.api_youtube.api_youtube import YoutubeAPI
#
# if __name__ == '__main__':
#     youtube_api = YoutubeAPI()
#
#     data_inicio = datetime(2026, 2, 19, tzinfo=timezone.utc)
#     print(data_inicio)
#
#     id_canal, _ = youtube_api.obter_id_canal('@jogatinaepica')
#     print(id_canal)
#     for video in youtube_api.obter_video_por_data(id_canal=id_canal, data_inicio=data_inicio):
#         print(video)
#         break
