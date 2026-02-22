import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Comentários simulados
# -----------------------------
comentarios = [
    "Construí uma base enorme com layout perfeito",
    "Minha fábrica está super eficiente agora",
    "Adorei organizar as linhas de produção",

    "Descobri um planeta exótico incrível",
    "Esse planeta tem minerais raros",
    "Explorar galáxias é a melhor parte do jogo",

    "O jogo está travando muito",
    "Encontrei vários bugs nessa atualização",
    "Performance caiu depois do patch",

    "Base modular ficou linda",
    "Sistema de trade está lucrativo",
    "Bug crítico quando uso muitas máquinas"
]

# -----------------------------
# Simular embeddings (10 dimensões)
# Em cenário real você usaria spaCy ou sentence-transformers
# -----------------------------
np.random.seed(42)
X = np.random.randn(len(comentarios), 10)

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Comentários simulados
# -----------------------------
comentarios = [
    "Construí uma base enorme com layout perfeito",
    "Minha fábrica está super eficiente agora",
    "Adorei organizar as linhas de produção",

    "Descobri um planeta exótico incrível",
    "Esse planeta tem minerais raros",
    "Explorar galáxias é a melhor parte do jogo",

    "O jogo está travando muito",
    "Encontrei vários bugs nessa atualização",
    "Performance caiu depois do patch",

    "Base modular ficou linda",
    "Sistema de trade está lucrativo",
    "Bug crítico quando uso muitas máquinas"
]

# -----------------------------
# Simular embeddings (10 dimensões)
# Em cenário real você usaria spaCy ou sentence-transformers
# -----------------------------
np.random.seed(42)
X = np.random.randn(len(comentarios), 10)

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)


visible_dim = X.shape[1]
hidden_dim = 5

class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))

    def sample_prob(self, probs):
        return torch.bernoulli(probs)

    def forward(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        h_sample = self.sample_prob(h_prob)
        v_prob = torch.sigmoid(torch.matmul(h_sample, self.W.t()) + self.v_bias)
        return v_prob, h_prob

rbm = RBM(visible_dim, hidden_dim)
optimizer = optim.Adam(rbm.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    v_recon, _ = rbm(X)
    loss = criterion(v_recon, X)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")


with torch.no_grad():
    _, hidden = rbm(X)

latent = hidden.numpy()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(latent)

for comentario, cluster in zip(comentarios, clusters):
    print(f"[Cluster {cluster}] {comentario}")
