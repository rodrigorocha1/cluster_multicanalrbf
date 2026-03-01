import ast
import base64
import io
import json
import os
from collections import Counter
from io import BytesIO

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.cluster import KMeans
from sklearn.cluster._hdbscan import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from servicos.modelo.rmb_youtube import RBMYoutube
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb


# -----------------------------
# Configuração MLflow
# -----------------------------

def configurar_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri


configurar_mlflow("http://localhost:5000", experiment_name="RBM_Youtube_Cluster")

# -----------------------------
# Leitura e preparação dos dados
# -----------------------------
obddb = OperacoesBancoDuckDb()
caminho_consulta = "s3://extracao/prata/comentarios_youtube_prata_2026_02_22_14_43_29.csv"
df_original = obddb.consultar_dados(id_consulta='1=1', caminho_consulta=caminho_consulta).drop_duplicates()

df_original["embeddings"] = df_original["embeddings"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
embeddings_array = np.array(df_original["embeddings"].tolist(), dtype=np.float32)
scaler = StandardScaler()
embeddings_normalizados = scaler.fit_transform(embeddings_array)
dados_tensor = torch.tensor(embeddings_normalizados, dtype=torch.float32)

# -----------------------------
# Treinamento RBM
# -----------------------------
n_visiveis = dados_tensor.shape[1]
n_ocultos = 16
epocas = 100
taxa_aprendizado = 0.02
tamanho_batch = 32

rbm = RBMYoutube(n_visiveis=n_visiveis, n_ocultos=n_ocultos)
rbm.treinar(dados_tensor, epocas=epocas, taxa_aprendizado=taxa_aprendizado, tamanho_batch=tamanho_batch)
ativacoes_latentes = rbm.transformar_latente(dados_tensor).detach().numpy()


# -----------------------------
# Funções utilitárias
# -----------------------------
def log_cluster_counts(clusters, nome_arquivo):
    contagem = {int(k): int(v) for k, v in Counter(clusters).items()}
    mlflow.log_dict(contagem, nome_arquivo)


def log_wordclouds(clusters, df, prefix="kmeans"):
    for cluster_id in set(clusters):
        if cluster_id == -1:
            continue
        textos = df.loc[clusters == cluster_id, "texto_comentario"].tolist()
        if not textos:
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(textos))
        buffer = BytesIO()
        wc.to_image().save(buffer, format="PNG")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        html_content = f'<img src="data:image/png;base64,{img_b64}"/>'
        mlflow.log_text(html_content, artifact_file=f"visualizacoes/wordclouds_{prefix}/cluster_{cluster_id}.html")
        buffer.close()


# -----------------------------
# KMEANS
# -----------------------------
n_clusters_kmeans = 4
with mlflow.start_run(run_name="RBM_KMeans") as run_kmeans:
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
    clusters = kmeans.fit_predict(ativacoes_latentes)
    score_silhouette = silhouette_score(ativacoes_latentes, clusters)
    mlflow.log_metric("silhouette_score", score_silhouette)

    # Artefatos t-SNE
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    coords_tsne = tsne.fit_transform(ativacoes_latentes)
    df_viz = pd.DataFrame({
        "tsne_1": coords_tsne[:, 0],
        "tsne_2": coords_tsne[:, 1],
        "cluster_kmeans": clusters,
        "id_canal": df_original["id_canal"],
        "comentario": df_original["texto_comentario"]
    })
    fig = px.scatter(df_viz, x="tsne_1", y="tsne_2", color="cluster_kmeans",
                     symbol="id_canal", hover_data=["comentario", "id_canal"],
                     title="Clusterização KMeans")
    html_buffer = io.StringIO()
    fig.write_html(html_buffer)
    html_buffer.seek(0)
    mlflow.log_text(html_buffer.getvalue(), artifact_file="visualizacoes/pca_tsne_kmeans.html")
    html_buffer.close()

    log_cluster_counts(clusters, "kmeans_cluster_counts.json")
    log_wordclouds(clusters, df_original, prefix="kmeans")

# -----------------------------
# HDBSCAN
# -----------------------------
from sklearn.cluster._hdbscan import hdbscan

min_cluster_size_hdbscan = 5
min_samples_hdbscan = 20
metrica_hdbscan = 'euclidean'

with mlflow.start_run(run_name="RBM_HDBSCAN") as run_hdbscan:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size_hdbscan,
                                min_samples=min_samples_hdbscan,
                                metric=metrica_hdbscan)
    clusters_hdbscan = clusterer.fit_predict(ativacoes_latentes)

    mask = clusters_hdbscan != -1
    if np.sum(mask) > 1:
        score_silhouette = silhouette_score(ativacoes_latentes[mask], clusters_hdbscan[mask])
    else:
        score_silhouette = -1.0
    mlflow.log_metric("silhouette_score_hdbscan", score_silhouette)

    tsne = TSNE(n_components=2, random_state=42, init="pca")
    coords_tsne = tsne.fit_transform(ativacoes_latentes)
    df_viz = pd.DataFrame({
        "tsne_1": coords_tsne[:, 0],
        "tsne_2": coords_tsne[:, 1],
        "cluster_hdbscan": clusters_hdbscan,
        "id_canal": df_original["id_canal"],
        "comentario": df_original["texto_comentario"]
    })
    fig = px.scatter(df_viz, x="tsne_1", y="tsne_2", color="cluster_hdbscan",
                     symbol="id_canal", hover_data=["comentario", "id_canal"],
                     title="Clusterização HDBSCAN")
    html_buffer = io.StringIO()
    fig.write_html(html_buffer)
    html_buffer.seek(0)
    mlflow.log_text(html_buffer.getvalue(), artifact_file="visualizacoes/pca_tsne_hdbscan.html")
    html_buffer.close()

    log_cluster_counts(clusters_hdbscan, "hdbscan_cluster_counts.json")
    log_wordclouds(clusters_hdbscan, df_original, prefix="hdbscan")
