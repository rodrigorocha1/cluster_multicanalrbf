from abc import ABC, abstractmethod
from typing import TypeVar, Generic

# Define um tipo genérico T
T = TypeVar("T")
R = TypeVar("R")


class IoperacoesBanco(ABC, Generic[T, R]):
    @abstractmethod
    def consultar_dados(self, id_consulta: str, caminho_consulta: str) -> R:
        """
        Método para consultar registros.
        Retorna um tipo genérico R.
        """
        pass

    @abstractmethod
    def guardar_dados(self, dados: T):
        """
        Método para guardar dados.
        Aceita um tipo genérico T.
        """
        pass
