import duckdb
import pandas as pd
from src.servicos.config.config import Config  # sua configuração

# Conexão com DuckDB (em memória ou arquivo .db)
con = duckdb.connect()

# Configuração das credenciais S3 (MinIO)
con.execute(f"""
    SET s3_region='{Config.AWS_REGION}';
    SET s3_access_key_id='{Config.MINIO_ACCESS_KEY}';
    SET s3_secret_access_key='{Config.MINIO_SECRET_KEY}';
    SET s3_endpoint='{Config.MINIO_HOST_URL_DUCKDB}';
    SET s3_use_ssl=false;
    SET s3_url_style='path';
""")

# Caminho do CSV no MinIO
s3_path = f"s3://extracao/prata/comentarios_youtube_prata_2026_02_21_22_15_46.csv"

# Ler CSV direto do MinIO para pandas DataFrame
df = con.execute(f"""
    SELECT * FROM read_csv_auto('{s3_path}')
""").df()

# Visualizar as primeiras linhas
print(df.head())

df['comentario_limpo'].isna().sum()