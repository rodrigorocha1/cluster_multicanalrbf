from datetime import datetime, timezone
from typing import Generator, Dict

from googleapiclient.discovery import build  # type: ignore

from src.servicos.config.config import Config


class YoutubeAPI:

    def __init__(self):
        self.__youtube = build('youtube', 'v3', developerKey=Config.CHAVE_API_YOUTUBE)

    def obter_id_canal(self, id_canal: str):
        request = self.__youtube.search().list(
            part="snippet",
            q=id_canal,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        if 'items' in response and len(response['items']) > 0:
            return response['items'][0]['id']['channelId'], response['items'][0]['snippet']['title']
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
                maxResults=50
            )

            response = request.execute()

            for item in response['items']:
                video_id = item['id']['videoId']
                video_title = item['snippet']['title']
                yield video_id, video_title

            try:
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
            request = self.__youtube.commentThreads().list(
                part="snippet",
                videoId=id_video,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            yield from response["items"]
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
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
            request = self.__youtube.comments().list(
                part="snippet",
                parentId=id_comentario,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token  # plainText ou html
            )

            response = request.execute()

            yield from response.get("items", [])

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break


if __name__ == '__main__':
    youtube_api = YoutubeAPI()

    data_inicio = datetime(2026, 2, 19, tzinfo=timezone.utc)

    id_canal, _ = youtube_api.obter_id_canal('@jogatinaepica')
    print(id_canal)
    for video in youtube_api.obter_video_por_data(id_canal=id_canal, data_inicio=data_inicio):
        print(video)
        break
