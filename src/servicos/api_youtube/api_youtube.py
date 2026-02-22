from datetime import datetime
from typing import Generator, Dict

from googleapiclient.discovery import build  # type: ignore

from src.servicos.config.config import Config
from src.utlls.log_sqlite import logger


class YoutubeAPI:

    def __init__(self):
        self.__youtube = build('youtube', 'v3', developerKey=Config.CHAVE_API_YOUTUBE)

    def obter_id_canal(self, id_canal: str):
        try:
            request = self.__youtube.search().list(
                part="snippet",
                q=id_canal,
                type="channel",
                maxResults=1
            )

            response = request.execute()
            url_canal = f"https://www.youtube.com/channel/{id_canal}"

            logger.info(f'Sucesso ao recuperar lista de videos do canal {id_canal}', extra={
                "descricao": "Consulta canal YouTube",
                "url": url_canal,
                "codigo": 200,
                'requisicao': response
            })
            if 'items' in response and len(response['items']) > 0:
                return response['items'][0]['id']['channelId'], response['items'][0]['snippet']['title']
            return None
        except Exception as e:
            logger.error(f'Erro ao consultar id do canal {id_canal}', extra={
                'exception': e
            })
            return None

    def obter_video_por_data(self, id_canal: str, data_inicio: datetime):
        data_inicio_string = data_inicio.strftime("%Y-%m-%dT%H:%M:%SZ")
        flag_token = True
        token = ''

        while flag_token:
            request = self.__youtube.search().list(
                part="snippet",
                channelId=id_canal,
                order="date",
                publishedAfter=data_inicio_string,
                pageToken=token,

            )

            response = request.execute()

            logger.info(f'Sucesso ao recuperar o vídeo do canal {id_canal}', extra={
                "descricao": "Consulta vídeo YouTube",
                "url": 'url Vídeo',
                "codigo": 200,
                'requisicao': response
            })

            try:
                yield from  response['items']
                token = response['nextPageToken']
                flag_token = True
            except KeyError:
                flag_token = False

    def obter_comentarios_youtube(self, id_video: str) -> Generator[Dict, None, None]:
        """
            Método para obter comentários de um vídeo do youtube
            :param id_video: id do vídeo
            :type id_video: str
            :return: Gerador dos comentários
            :rtype: Generator[Dict, None, None]
        """
        next_page_token = None
        while True:
            try:
                request = self.__youtube.commentThreads().list(
                    part="snippet",
                    videoId=id_video,

                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()
                logger.info(f'Sucesso ao comentarios o vídeo  {id_video}', extra={
                    "descricao": "Consulta comentários YouTube",
                    "url": 'url_canal',
                    "codigo": 200,
                    'requisicao': response
                })

                if len(response['items']) == 0:
                    break


                yield from response["items"]
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            except Exception as e:
                logger.error(f'erro ao recuperar_comentarios {e}', extra={
                    'exception': str(e)
                })
                break

    def obter_resposta_comentarios(self, id_comentario: str) -> Generator[Dict, None, None]:
        """
        Recupera todas as respostas de um comentário no YouTube.

        Args:
            id_comentario (str): ID do comentário principal.

        Yields:
            Dict: Cada resposta do comentário.
        """
        next_page_token = None

        while True:
            try:
                request = self.__youtube.comments().list(
                    part="snippet",
                    parentId=id_comentario,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=next_page_token  # plainText ou html
                )

                response = request.execute()

                if len(response['items']) == 0:
                    break

                yield from response.get("items", [])
                logger.info(f'Sucesso ao pegar a resposta do comentário   {id_comentario}', extra={
                    "descricao": "Consulta comentários YouTube",
                    "url": 'url_canal',
                    "codigo": 200,
                    'requisicao': response
                })

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            except Exception as e:
                logger.error('erro ao recuperar_comentarios', extra={
                    'exception': str(e)
                })
                break
