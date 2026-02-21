from datetime import datetime
from typing import Dict, Generator, Protocol, Tuple, Optional


class IApiYoutube(Protocol):

    def obter_id_canal(self, id_canal: str) -> Optional[Tuple[str, str]]:
        ...

    def obter_video_por_data(
            self,
            id_canal: str,
            data_inicio: datetime
    ) -> Generator[Dict, None, None]:
        ...

    def obter_comentarios_youtube(
            self,
            id_video: str
    ) -> Generator[Dict, None, None]:
        ...

    def obter_resposta_comentarios(
            self,
            id_comentario: str
    ) -> Generator[Dict, None, None]:
        ...
