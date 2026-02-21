import json
import pickle
from typing import Any, Dict, List, Union

import boto3
import pandas as pd
import s3fs
from botocore.client import Config

from src.servicos.config.config import Config as c
from src.servicos.servico_s3.iservicos3 import Iservicos3

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 300)
pd.set_option("display.max_rows", 20)


class ServicoS3(Iservicos3):

    def __init__(self):
        self.__cliente_s3 = self._criar_cliente()
        self.__fs = self.criar_filesystem()

    @staticmethod
    def criar_filesystem():
        return s3fs.S3FileSystem(
            key=c.MINIO_ACCESS_KEY,
            secret=c.MINIO_SECRET_KEY,
            client_kwargs={"endpoint_url": c.MINIO_ENDPOINT},
        )

    def guardar_dados(self, dados: Any, caminho_arquivo: str, tipo: str = "json") -> None:

        if tipo == "json":
            self._guardar_json(dados, caminho_arquivo)
        elif tipo == "dataframe":
            self._guardar_dataframe(dados, caminho_arquivo)
        elif tipo == "pickle":
            self._guardar_pickle(dados, caminho_arquivo)
        else:
            raise ValueError(f"Tipo desconhecido: {tipo}. Use 'json', 'dataframe' ou 'pickle'.")

    def _guardar_json(self, dados: Union[Dict, List], caminho_arquivo: str) -> None:
        linhas = self._obter_linhas_existentes(caminho_arquivo)
        nova_linha = json.dumps(dados, ensure_ascii=False)
        linhas.append(nova_linha)
        self._salvar_linhas(caminho_arquivo, linhas, content_type="application/json")

    def _guardar_dataframe(self, df: pd.DataFrame, caminho_arquivo: str) -> None:

        csv_buffer = df.to_csv(index=False, encoding="utf-8")
        self.__cliente_s3.put_object(
            Bucket=c.MINIO_BUCKET_PLN,
            Key=caminho_arquivo,
            Body=csv_buffer.encode('utf-8'),
            ContentType="text/csv"
        )

    def _guardar_pickle(self, obj: Any, caminho_arquivo: str) -> None:
        pickle_bytes = pickle.dumps(obj)
        self.__cliente_s3.put_object(
            Bucket=c.MINIO_BUCKET_PLN,
            Key=caminho_arquivo,
            Body=pickle_bytes,
            ContentType="application/octet-stream"
        )

    @staticmethod
    def _criar_cliente():
        return boto3.client(
            "s3",
            endpoint_url=c.MINIO_ENDPOINT,
            aws_access_key_id=c.MINIO_ACCESS_KEY,
            aws_secret_access_key=c.MINIO_SECRET_KEY,
            region_name=c.AWS_REGION,
            config=Config(signature_version="s3v4")
        )

    def _obter_linhas_existentes(self, caminho_arquivo: str) -> List[str]:
        try:
            obj = self.__cliente_s3.get_object(
                Bucket=c.MINIO_BUCKET_PLN,
                Key=caminho_arquivo
            )
            conteudo = obj['Body'].read().decode('utf-8')
            return [linha for linha in conteudo.splitlines() if linha.strip()]
        except self.__cliente_s3.exceptions.NoSuchKey:
            return []

    def _salvar_linhas(self, caminho_arquivo: str, linhas: List[str], content_type="application/json") -> None:
        novo_conteudo = "\n".join(linhas)
        self.__cliente_s3.put_object(
            Bucket=c.MINIO_BUCKET_PLN,
            Key=caminho_arquivo,
            Body=novo_conteudo.encode('utf-8'),
            ContentType=content_type
        )