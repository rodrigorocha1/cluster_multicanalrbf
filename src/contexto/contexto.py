from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Any, List, Tuple, Optional, Generator

import pandas as pd


@dataclass
class Contexto:
    data_publicacao: Optional[datetime]
    gerador_reviews_steam: Iterable[Any] = field(default_factory=list)
    gerador_comentarios_youtube: Iterable[Any] = field(default_factory=list)
    lista_id_comentarios: List[Tuple[str, ...]] = field(default_factory=list)
    gerador_resposta_comentarios: Iterable[Any] = field(default_factory=list)
    dataframe_original: pd.DataFrame = field(default_factory=pd.DataFrame)
    lista_video: List[Generator[tuple[str, str], None, None]] = field(default_factory=list)
    lista_canais: List[str] = field(default_factory=list)
