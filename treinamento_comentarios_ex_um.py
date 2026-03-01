import ast
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from servicos.modelo.rmb_youtube import RBMYoutube
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb


# -----------------------------
# Configuração de display
# -----------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)
pd.set_option('display.precision', 2)
pd.set_option('display.colheader_justify', 'center')


# -----------------------------
# Leitura dos dados
# -----------------------------
obddb = OperacoesBancoDuckDb()

caminho_consulta = "s3://extracao/prata/comentarios_youtube_prata_2026_02_22_14_43_29.csv"

df_original = (
    obddb.consultar_dados(id_consulta='1=1', caminho_consulta=caminho_consulta)
    .drop_duplicates()

)

# Converter embeddings string → lista
df_original["embeddings"] = df_original["embeddings"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

embeddings_array = np.array(df_original["embeddings"].tolist(), dtype=np.float32)
print("Shape embeddings:", embeddings_array.shape)
print(df_original.shape)

# -----------------------------
# PCA com variância mínima
# -----------------------------
# def aplicar_pca(embeddings: np.ndarray,
#                 n_componentes_max: int = 50,
#                 variancia_minima: float = 0.90):
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(embeddings)
#
#     # evitar erro se n_componentes_max > limite permitido
#     limite = min(n_componentes_max, X_scaled.shape[0], X_scaled.shape[1])
#
#     pca_full = PCA(n_components=limite, random_state=42)
#     X_pca_full = pca_full.fit_transform(X_scaled)
#
#     variancia_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
#     k = np.argmax(variancia_acumulada >= variancia_minima) + 1
#
#     print(f"Dimensão ideal (k): {k}")
#     print(f"Variância acumulada: {variancia_acumulada[k - 1]:.4f}")
#
#     # refaz PCA com k componentes
#     pca = PCA(n_components=k, random_state=42)
#     X_pca = pca.fit_transform(X_scaled)
#
#     return torch.tensor(X_pca, dtype=torch.float32), pca, scaler, k
#
#
# # chamada correta
# x_pca, pca, scaler, k = aplicar_pca(
#     embeddings_array,
#     n_componentes_max=30,
#     variancia_minima=0.90
# )
#
#
# # -----------------------------
# # Treinamento RBM
# # -----------------------------
# n_ocultos = 12
#
# rbm = RBMYoutube(
#     neuronios_visiveis=k,
#     n_ocultos=n_ocultos
# )
#
# rbm.treinar_rede(
#     x_pca,
#     epocas=400,
#     taxa=0.005,
#     tamanho_batch=64
# )
#
# h_latente = rbm.propagacao_direta(x_pca).detach().cpu().numpy()
#
# # padronizar ativações antes do clustering
# scaler_latente = StandardScaler()
# h_latente = scaler_latente.fit_transform(h_latente)
#
# print("Shape espaço latente:", h_latente.shape)
#
#
# # -----------------------------
# # Elbow Method
# # -----------------------------
# wcss = []
# for i in range(1, 11):
#     kmeans_temp = KMeans(
#         n_clusters=i,
#         random_state=42,
#         n_init=20
#     )
#     kmeans_temp.fit(h_latente)
#     wcss.append(kmeans_temp.inertia_)
#
# fig_wcss = px.line(
#     x=list(range(1, 11)),
#     y=wcss,
#     markers=True,
#     title="Elbow Method - WCSS",
#     labels={"x": "Número de clusters", "y": "WCSS"}
# )
# fig_wcss.write_html("kmeans_elbow.html")
#
#
# # -----------------------------
# # Cluster final
# # -----------------------------
# kmeans = KMeans(
#     n_clusters=4,
#     random_state=42,
#     n_init=20
# )
#
# clusters = kmeans.fit_predict(h_latente)
#
# # métrica objetiva
# sil_score = silhouette_score(h_latente, clusters)
# print("Silhouette Score:", round(sil_score, 4))
#
#
# # -----------------------------
# # DataFrame final
# # -----------------------------
# df = pd.DataFrame(h_latente)
# df["cluster"] = clusters
# df["comentario"] = df_original["texto_comentario"].tolist()
# df["id_canal"] = df_original["id_canal"].tolist()
#
#
# # -----------------------------
# # t-SNE
# # -----------------------------
# tsne = TSNE(
#     n_components=2,
#     perplexity=min(30, len(h_latente) // 3),
#     learning_rate="auto",
#     init="pca",
#     random_state=42
# )
#
# coords = tsne.fit_transform(h_latente)
#
# df["x"] = coords[:, 0]
# df["y"] = coords[:, 1]
#
#
# # -----------------------------
# # Visualização interativa
# # -----------------------------
# fig = px.scatter(
#     df,
#     x="x",
#     y="y",
#     color="id_canal",
#     symbol="cluster",
#     hover_data=["comentario"],
#     title="t-SNE: Ativações RBM (Espaço Latente)",
#     template="plotly_dark",
#     opacity=0.85
# )
#
# fig.update_traces(marker=dict(size=9))
# fig.write_html("rbm_tsne_interativo_dark.html")
#
# print("Pipeline executado com sucesso.")
#
# comentarios = df_original["texto_comentario"].tolist()
# # Criar colunas h0..h{n_ocultos-1}
# colunas = [f"h{i}" for i in range(n_ocultos)]
#
# # Criar DataFrame
# df_ativacoes = pd.DataFrame(h_latente, columns=colunas)
# df_ativacoes["comentario"] = comentarios
#
# # Mostrar as primeiras linhas
# print(df_ativacoes.head())
#
# import plotly.graph_objects as go
#
#
# def radar_neuronio(df_ativacoes, neuronio="h0"):
#     valores = df_ativacoes[neuronio].values
#     categorias = [f"C{i}" for i in range(len(valores))]
#
#     fig = go.Figure()
#
#     fig.add_trace(go.Scatterpolar(
#         r=valores,
#         theta=categorias,
#         fill='toself'
#     ))
#
#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True)),
#         title=f"Padrão de Ativação - {neuronio}"
#     )
#
#     fig.show()
#
# radar_neuronio(df_ativacoes, "h0")