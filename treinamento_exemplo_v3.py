import numpy as np
import pandas as pd
import torch
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from typing import List

from servicos.modelo.rmb_youtube import RBMYoutube

# Certifique-se de que a classe RBMYoutube atualizada está acessível
# from servicos.modelo.rmb_youtube import RBMYoutube

nlp = spacy.load("pt_core_news_lg")

# -----------------------------
# 1. Dados de Amostra
# -----------------------------
comentarios_satisfactory = [
    "A otimização dessa fábrica está impecável!", "Gostei das dicas de produção, muito útil.",
    "Não sabia que podia automatizar dessa forma.", "Muito bom, mas poderia mostrar mais caminhos alternativos.",
    "A estratégia de transporte de materiais é genial.",
    "Excelente vídeo, aprendi a montar linhas de produção melhores.",
    "A expansão da base ficou muito organizada, gostei.", "Tutorial claro, deu para entender bem os mods.",
    "Ótima explicação sobre logística e fluxo de itens.", "Conteúdo incrível, ajudou a otimizar minha fábrica."
]
comentarios_nms = [
    "Adorei o vídeo, consegui encontrar planetas incríveis!", "As novas atualizações do No Man's Sky são sensacionais.",
    "Faltou mostrar como construir bases eficientes.", "Explicação clara sobre exploração interplanetária.",
    "Muito bom, aprendi truques que não conhecia.", "Gostei das dicas de recursos raros, muito útil.",
    "O vídeo ajudou a entender o sistema de naves.", "Excelente conteúdo, exploração e sobrevivência bem explicadas.",
    "Não sabia que podia personalizar tanto a base, valeu!", "Vídeo detalhado sobre mods e upgrades, adorei."
]
comentarios_tech = [
    "Excelente explicação sobre GPUs e performance.", "Gostei do passo a passo do tutorial de Python.",
    "Vídeo informativo, mas faltou abordar benchmarks.", "Muito técnico, adorei os exemplos de código.",
    "Parabéns pelo conteúdo, aprendi bastante.", "Tutorial de Linux muito útil, consegui aplicar direto.",
    "Explicação clara sobre inteligência artificial e ML.", "Adorei os exemplos práticos de programação.",
    "Vídeo detalhado sobre hardware e otimização.", "Conteúdo técnico profundo, recomendo a todos que estudam TI."
]

comentarios = comentarios_satisfactory + comentarios_nms + comentarios_tech
canais = ["Satisfactory"] * 10 + ["No Man's Sky"] * 10 + ["Tech Insights"] * 10
ids_canal = [c.lower().replace(" ", "_") for c in canais]

# -----------------------------
# 2. Processamento e PCA
# -----------------------------
print("Gerando Embeddings e aplicando PCA...")
embeddings = np.array([nlp(texto).vector for texto in comentarios])

# Para 30 exemplos, 10 componentes capturam bem a variância
pca = PCA(n_components=10)
dados_reduzidos = pca.fit_transform(embeddings)
dados_tensor = torch.tensor(dados_reduzidos, dtype=torch.float32)

# -----------------------------
# 3. Treino da RBM
# -----------------------------
n_visiveis = dados_tensor.shape[1]
n_ocultos = 8
rbm = RBMYoutube(n_visiveis=n_visiveis, n_ocultos=n_ocultos)
rbm.treinar_rede(dados=dados_tensor, epocas=300, taxa=0.02, tamanho_batch=5)

# Obter ativações (H)
H_latente = rbm.propagacao_direta(dados_tensor).detach().cpu().numpy()


# -----------------------------
# 4. Mapeamento de Palavras (V2)
# -----------------------------
def mapear_palavras_v2(pca_model, rbm_model, vocabulario, top_k=5):
    pesos_pca = pca_model.components_
    pesos_rbm = rbm_model.pesos.data.cpu().numpy()
    influencia_no_embedding = pesos_pca.T @ pesos_rbm
    vetores_vocab = np.array([nlp(p).vector for p in vocabulario])
    relacao_final = vetores_vocab @ influencia_no_embedding

    palavras_por_h = {}
    for h in range(pesos_rbm.shape[1]):
        indices_top = np.argsort(relacao_final[:, h])[::-1][:top_k]
        palavras_por_h[h] = [vocabulario[i] for i in indices_top]
    return palavras_por_h


vocabulario = list({t.text.lower() for c in comentarios for t in nlp(c) if t.is_alpha and not t.is_stop})
top_words = mapear_palavras_v2(pca, rbm, vocabulario)

# -----------------------------
# 5. Geração de Gráficos
# -----------------------------

# Criar DataFrame para visualização
df = pd.DataFrame(H_latente, columns=[f"h{i}" for i in range(n_ocultos)])
df["comentario"] = comentarios
df["canal"] = canais
df["cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(H_latente)


wcss = []
for i in range(1, 11):
    kmeans_temp = KMeans(n_clusters=i, random_state=42)
    kmeans_temp.fit(H_latente)
    wcss.append(kmeans_temp.inertia_)

# --- Gráfico WCSS ---
fig_wcss = px.line(
    x=list(range(1, 11)),
    y=wcss,
    markers=True,
    title="Elbow Method - WCSS por número de clusters",
    labels={"x": "Número de clusters", "y": "WCSS"}
)
fig_wcss.write_html('kmeans_comentarios.html')


# A: t-SNE Interativo
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
coords = tsne.fit_transform(H_latente)
df["x"], df["y"] = coords[:, 0], coords[:, 1]

fig = px.scatter(
    df,
    x="x", y="y",
    color="canal",
    symbol="cluster",
    hover_data=["comentario"],
    title="t-SNE: Ativações RBM (Espaço Latente)",
    template="plotly_dark"  # <<< modo dark ativado
)

# Salvar o HTML interativo
fig.write_html("teste_rbm_interativo_dark.html")
print("Gráfico Plotly (Dark Mode) salvo: teste_rbm_interativo_dark.html")

# B: Heatmap de Especialistas
plt.figure(figsize=(12, 6))
df_mean = df.groupby(["canal", "cluster"]).mean(numeric_only=True).drop(columns=['x', 'y'])
sns.heatmap(df_mean, annot=True, cmap="coolwarm", fmt=".2f")

# Adicionar as palavras-chave no eixo X para facilitar a leitura
labels_x = [f"h{i}\n({', '.join(top_words[i][:2])})" for i in range(n_ocultos)]
plt.xticks(np.arange(n_ocultos) + 0.5, labels_x, rotation=45, ha='right')
plt.title("Mapa de Calor: Neurónios vs Temas")
plt.tight_layout()
plt.savefig("heatmap_especialistas_teste.png")
plt.show()

# -----------------------------
# 6. Output de Texto Final
# -----------------------------
print("\n=== RESUMO DOS NEURÓNIOS (CONCEITOS APRENDIDOS) ===")
for h, palavras in top_words.items():
    print(f"h{h}: {', '.join(palavras)}")


print("\n=== VALORES DOS NEURÔNIOS ESPECIALISTAS (CONCEITOS APRENDIDOS) ===")
for h, palavras in top_words.items():
    print(f"Neurônio h{h}: {', '.join(palavras[:10])}")  # Mostrando até 10 palavras mais influentes


# Criar DataFrame detalhado com comentários e valores dos neurônios
df_detalhado = pd.DataFrame(H_latente, columns=[f"h{i}" for i in range(n_ocultos)])
df_detalhado["comentario"] = comentarios
df_detalhado["canal"] = canais
df_detalhado["cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(H_latente)
# Exibir todas as colunas e linhas
pd.set_option('display.max_columns', None)   # mostra todas as colunas
pd.set_option('display.max_rows', None)      # mostra todas as linhas
pd.set_option('display.width', 300)          # largura máxima do display
pd.set_option('display.precision', 2)        # casas decimais para float
pd.set_option('display.colheader_justify', 'center')  # centralizar cabeçalho

# Exibir os 10 primeiros para conferência
print(df_detalhado.head(10))