import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from servicos.modelo.rmb_youtube import RBMYoutube  # Classe consistente

# -----------------------------
# Configurações de display
# -----------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.colheader_justify", "left")

# -----------------------------
# Carregar modelo spaCy
# -----------------------------
nlp = spacy.load("pt_core_news_lg")

# -----------------------------
# Comentários de exemplo
# -----------------------------
comentarios = [
    "Excelente aula, muito clara e detalhada.",
    "Adorei a explicação, muito útil!",
    "Fantástico conteúdo, aprendi bastante.",
    "Muito bom, obrigado por compartilhar!",
    "Ótima didática, consegui entender tudo.",
    "Amei o vídeo, conteúdo de qualidade.",
    "Parabéns pelo esforço, aula excelente!",
    "Muito claro e direto, adorei!",
    "Conteúdo incrível, recomendo para todos.",
    "Excelente abordagem, aprendi novos conceitos.",
    "Não gostei, ficou confuso.",
    "O vídeo é ruim, explicação fraca.",
    "Áudio péssimo e sem clareza.",
    "Não recomendo, muito superficial.",
    "Faltou organização no conteúdo.",
    "Muito complicado de entender.",
    "Explicação ruim, perdi tempo.",
    "Não explicou direito o assunto.",
    "Decepcionante, esperava mais.",
    "Aula confusa e mal estruturada.",
    "Ótima análise técnica, exemplos muito bons.",
    "Muito detalhado, gostei dos cálculos.",
    "Explicação matemática clara e precisa.",
    "Conteúdo técnico profundo, aprendi muito.",
    "Exemplos práticos ajudam a entender a teoria.",
    "Abordagem detalhada sobre redes neurais.",
    "Muito informativo, ótimo para estudo avançado.",
    "Detalhamento correto dos conceitos técnicos.",
    "Exemplos de código ajudam bastante.",
    "Explicação passo a passo, muito técnico.",
    "Legal, mas poderia ser mais curto.",
    "Vídeo ok, nada demais.",
    "Interessante, mas prefiro outros vídeos.",
    "Informativo, mas poderia ter mais exemplos.",
    "Conteúdo mediano, não me surpreendeu.",
    "Ok, mas esperava algo mais prático.",
    "Tudo bem, mas faltou clareza em algumas partes.",
    "Comentário neutro sobre o vídeo.",
    "Assisti inteiro, mas nada novo.",
    "Poderia melhorar, mas serve como referência."
]

# -----------------------------
# Gerar embeddings via spaCy
# -----------------------------
def gerar_embeddings(textos):
    return np.array([nlp(texto).vector for texto in textos])

embeddings = gerar_embeddings(comentarios)

# -----------------------------
# Reduzir dimensionalidade com PCA
# -----------------------------
pca = PCA(n_components=20)
dados = torch.tensor(pca.fit_transform(embeddings), dtype=torch.float32)

# -----------------------------
# Inicializar RBM Gaussiana
# -----------------------------
n_visiveis = dados.shape[1]
n_ocultos = 8
# Inicializar RBM Gaussiana
rbm = RBMYoutube(n_visiveis=n_visiveis, n_ocultos=n_ocultos)

# Treinar RBM com mini-batch
rbm.treinar_rede(dados=dados, epocas=100, taxa=0.01, tamanho_batch=16)

# Obter ativações latentes
ativacoes = rbm.obter_ativacoes(dados)

with torch.no_grad():
    H_latente = rbm.propagacao_direta(dados).numpy()

print("Shape latente:", H_latente.shape)

# Mostrar ativações
for i, comentario in enumerate(comentarios):
    print(f"\nComentário {i + 1}: {comentario}")
    print(f"Ativações (h0..h{n_ocultos - 1}): {H_latente[i].tolist()}")


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(ativacoes.detach().numpy())


for c in range(3):
    print(f"\n=== Cluster {c} ===")
    for i, comentario in enumerate(comentarios):
        if clusters[i] == c:
            print(f"{i+1}: {comentario}")
import umap
mapper = umap.UMAP(n_components=2, random_state=42
                   , n_jobs=1).fit_transform(ativacoes.detach().numpy())
import matplotlib.pyplot as plt

# Cria scatter plot das coordenadas UMAP
plt.figure(figsize=(10,6))
plt.scatter(mapper[:,0], mapper[:,1], c=clusters, cmap='tab10', s=100)  # usa clusters como cores
for i, txt in enumerate(range(len(comentarios))):
    plt.annotate(str(i+1), (mapper[i,0], mapper[i,1]), fontsize=8, alpha=0.7)
plt.title("Visualização 2D das ativações RBM via UMAP")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.colorbar(label="Cluster K-Means")
plt.savefig('umap.png')
plt.close()



tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, random_state=42)
ativacoes_2d = tsne.fit_transform(H_latente)
# Clustering KMeans para visualização (opcional)
n_clusters = 4  # você pode ajustar
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(H_latente)

# Cores diferentes por cluster
colors = plt.cm.get_cmap('tab10', n_clusters)

# Plot
plt.figure(figsize=(14, 10))
for i in range(n_clusters):
    mask = clusters == i
    plt.scatter(
        ativacoes_2d[mask, 0],
        ativacoes_2d[mask, 1],
        color=colors(i),
        alpha=0.7,
        label=f'Cluster {i+1}',
        s=50
    )

# Adiciona alguns comentários próximos aos pontos (apenas os primeiros 30)
for i, comentario in enumerate(comentarios[:30]):
    plt.text(
        ativacoes_2d[i, 0],
        ativacoes_2d[i, 1],
        comentario[:30] + ('...' if len(comentario) > 30 else ''),
        fontsize=9
    )

plt.title("t-SNE das ativações latentes da RBM por Cluster")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.grid(True)
plt.legend()
plt.savefig('t-SNE.png')
plt.close()


import plotly.express as px
import pandas as pd

# Criar DataFrame para plotagem
df_tsne = pd.DataFrame({
    "x": ativacoes_2d[:, 0],
    "y": ativacoes_2d[:, 1],
    "comentario": comentarios,
    "cluster": clusters
})

# Criar gráfico interativo
fig = px.scatter(
    df_tsne,
    x="x",
    y="y",
    color="cluster",
    hover_data=["comentario"],
    title="t-SNE interativo das ativações latentes da RBM",
    width=1000,
    height=700,
    color_continuous_scale=px.colors.qualitative.Set1
)

# Salvar como arquivo HTML interativo
fig.write_html("tsne_rbm_interativo.html")

print("Gráfico interativo salvo em: tsne_rbm_interativo.html")