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
def configurar_mlflow(tracking_uri: str, experiment_name: str = "RBM_Youtube"):
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
print(df_original.columns)
# -----------------------------
# Treinamento RBM e registro no MLflow
# -----------------------------
n_visiveis = dados_tensor.shape[1]
n_ocultos = 16
epocas = 100
taxa_aprendizado = 0.02
tamanho_batch = 32
n_componentes = 0
min_cluster_size_hdbscan = 5
metrica_hdbscan = 'euclidean'
min_samples_hdbscan = 20
n_clusters_kmeans = 4

with mlflow.start_run(run_name="RBM_Youtube_Cluster") as run:
    def log_cluster_counts(clusters, nome_arquivo):
        contagem = {int(k): int(v) for k, v in Counter(clusters).items()}
        mlflow.log_dict(contagem, nome_arquivo)


    from PIL import Image
    import io


    def log_wordclouds(clusters, df, prefix="kmeans"):
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue
            textos = df.loc[clusters == cluster_id, "comentario_limpo"].tolist()
            if not textos:
                continue

            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(textos))
            buffer = BytesIO()
            wc.to_image().save(buffer, format="PNG")
            buffer.seek(0)

            # converte para base64
            img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
            html_content = f'<img src="data:image/png;base64,{img_b64}"/>'

            # loga HTML no MLflow
            mlflow.log_text(html_content, artifact_file=f"visualizacoes/wordclouds_{prefix}/cluster_{cluster_id}.html")
            buffer.close()



    rbm = RBMYoutube(n_visiveis=n_visiveis, n_ocultos=n_ocultos)
    rbm.treinar(dados_tensor, epocas=epocas, taxa_aprendizado=taxa_aprendizado, tamanho_batch=tamanho_batch)

    ativacoes_latentes = rbm.transformar_latente(dados_tensor).detach().numpy()

    comentarios = df_original["texto_comentario"].tolist()
    dados_json = [{"comentario": c, "ativacoes_latentes": l.tolist()}
                  for c, l in zip(comentarios, ativacoes_latentes)]

    json_buffer = io.StringIO()
    json.dump(dados_json, json_buffer, ensure_ascii=False, indent=2)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file="embeddings_latentes/comentarios_ativacoes_rbm.json")
    json_buffer.close()

    # pca = PCA(n_components=n_componentes, random_state=42)
    # ativacoes_pca = pca.fit_transform(ativacoes_latentes)

    ativacoes_pca = ativacoes_latentes
    wcss = []
    for i in range(1, 11):
        kmeans_cartao_mais = KMeans(n_clusters=i, random_state=0)
        kmeans_cartao_mais.fit(ativacoes_pca)
        wcss.append(kmeans_cartao_mais.inertia_)

    mlflow.log_params({
        "n_visiveis": n_visiveis,
        "n_ocultos": n_ocultos,
        "epocas": epocas,
        "taxa_aprendizado": taxa_aprendizado,
        "tamanho_batch": tamanho_batch,
        "n_componentes": n_componentes,
        "n_clusters_kmeans": n_clusters_kmeans,
        "min_cluster_size_hdbscan": min_cluster_size_hdbscan,
        "min_samples_hdbscan": min_cluster_size_hdbscan
    })

    grafico = px.line(x=range(1, 11), y=wcss)
    html_buffer = io.StringIO()
    grafico.write_html(html_buffer)
    html_buffer.seek(0)
    mlflow.log_text(html_buffer.getvalue(), artifact_file="visualizacoes/kmeans_cluster.html")
    html_buffer.close()

    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
    clusters = kmeans.fit_predict(ativacoes_pca)

    score_silhouette = silhouette_score(ativacoes_pca, clusters)
    mlflow.log_metric("silhouette_score", score_silhouette)

    tsne = TSNE(n_components=2, random_state=42, init="pca")
    coords_tsne = tsne.fit_transform(ativacoes_pca)

    df_viz_hdbscan = pd.DataFrame({
        "tsne_1": coords_tsne[:, 0],
        "tsne_2": coords_tsne[:, 1],
        "cluster_kmeans": clusters,
        "id_canal": df_original["id_canal"],
        "comentario": df_original["texto_comentario"]
    })

    cores_cluster = {
        0: "blue",
        1: "red",
        2: "green",
        3: "orange",
        4: "purple"
    }

    fig = px.scatter(
        df_viz_hdbscan,
        x="tsne_1",
        y="tsne_2",
        color="cluster_kmeans",
        symbol="id_canal",  # Formas diferentes por canal
        hover_data=["comentario", "id_canal"],
        title="Clusterização de Comentários (RBM + PCA + KMeans) por Canal",
        color_discrete_map=cores_cluster
    )
    fig.update_layout(showlegend=False)

    html_buffer = io.StringIO()
    fig.write_html(html_buffer)
    html_buffer.seek(0)
    mlflow.log_text(html_buffer.getvalue(), artifact_file="visualizacoes/pca_tsne_kmeans.html")
    html_buffer.close()

    # -----------------------------
    # HDBSCAN
    # -----------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size_hdbscan,
        metric=metrica_hdbscan,
        min_samples=min_samples_hdbscan
    )
    clusters_hdbscan = clusterer.fit_predict(ativacoes_pca)

    # Silhouette Score (apenas para pontos atribuídos a clusters, ignora -1)
    mask = clusters_hdbscan != -1
    if np.sum(mask) > 1:  # necessário pelo menos 2 pontos para silhouette
        score_silhouette = silhouette_score(ativacoes_pca[mask], clusters_hdbscan[mask])
    else:
        score_silhouette = -1.0
    mlflow.log_metric("silhouette_score_hdscan", score_silhouette)
    print(f"Silhouette Score HDBSCAN: {score_silhouette:.4f}")

    cluster_probabilities = clusterer.probabilities_

    n_clusters = len(set(clusters_hdbscan)) - (1 if -1 in clusters_hdbscan else 0)
    n_outliers = np.sum(clusters_hdbscan == -1)

    # Log no MLflow

    mlflow.log_metric("mean_probability", np.mean(cluster_probabilities))

    # -----------------------------
    # Visualização t-SNE
    # -----------------------------

    coords_tsnehdbscan = tsne.fit_transform(ativacoes_pca)

    df_viz_hdbscan = pd.DataFrame({
        "tsne_1": coords_tsnehdbscan[:, 0],
        "tsne_2": coords_tsnehdbscan[:, 1],
        "clusters_hdbscan": clusters_hdbscan,
        "id_canal": df_original["id_canal"],
        "comentario": df_original["texto_comentario"]
    })

    fig = px.scatter(
        df_viz_hdbscan,
        x="tsne_1",
        y="tsne_2",
        color="clusters_hdbscan",
        symbol="id_canal",
        hover_data=["comentario", "id_canal"],
        title="Clusterização de Comentários (RBM + PCA + HDBSCAN)"
    )

    # Salvar gráfico no MLflow
    html_buffer = io.StringIO()
    fig.write_html(html_buffer)
    html_buffer.seek(0)
    mlflow.log_text(html_buffer.getvalue(), artifact_file="visualizacoes/pca_tsne_hdbscan.html")
    html_buffer.close()

    print(f"Run registrada no MLflow: {mlflow.get_tracking_uri()}, run_id: {run.info.run_id}")
    input_example = embeddings_normalizados[:1].astype(np.float32)

    contagem_kmeans = Counter(clusters)


    contagem_hdb = Counter(clusters_hdbscan)


    log_cluster_counts(clusters, "kmeans_cluster_counts.json")
    log_cluster_counts(clusters_hdbscan, "hdbscan_cluster_counts.json")
    log_wordclouds(clusters, df_original, prefix="kmeans")
    log_wordclouds(clusters_hdbscan, df_original, prefix="hdbscan")




    mlflow.pytorch.log_model(
        pytorch_model=rbm,
        name="rbm_youtube",
        registered_model_name="RBM_Youtube_Model",
        export_model=False,
        input_example=input_example
    )
    print(f"Modelo registrado no MLflow: run_id={run.info.run_id}")
