import torch
import torch.nn as nn
from tqdm import tqdm  # Import do tqdm


class RBMYoutube(nn.Module):
    def __init__(self, n_visiveis: int, n_ocultos: int):
        """
        RBM customizada para embeddings de comentários do YouTube.
        """
        super().__init__()
        # Pesos entre camada visível e oculta
        self.W = nn.Parameter(torch.randn(n_visiveis, n_ocultos) * 0.01)
        # Bias da camada oculta
        self.h_bias = nn.Parameter(torch.zeros(n_ocultos))
        # Bias da camada visível
        self.v_bias = nn.Parameter(torch.zeros(n_visiveis))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula as ativações latentes (probabilidade dos neurônios ocultos)
        """
        return torch.sigmoid(torch.matmul(x, self.W) + self.h_bias)

    def treinar(
            self,
            dados: torch.Tensor,
            epocas: int = 100,
            taxa_aprendizado: float = 0.01,
            tamanho_batch: int = 32
    ):
        """
        Treinamento da RBM com tqdm mostrando loss média por batch e progresso da época.

        Args:
            dados (torch.Tensor): entradas de treinamento (batch_size x n_visiveis)
            epocas (int): número de épocas
            taxa_aprendizado (float): learning rate
            tamanho_batch (int): tamanho do batch
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=taxa_aprendizado)

        for ep in range(epocas):
            permutation = torch.randperm(dados.size(0))
            batch_losses = []

            # Barra de progresso por batch
            with tqdm(range(0, dados.size(0), tamanho_batch), desc=f"Época {ep + 1}/{epocas}", unit="batch") as pbar:
                for i in pbar:
                    batch_idx = permutation[i:i + tamanho_batch]
                    v0 = dados[batch_idx]

                    # Forward
                    h_prob = self.forward(v0)

                    # Reconstrução
                    v_recon = torch.sigmoid(torch.matmul(h_prob, self.W.t()) + self.v_bias)

                    # Loss
                    loss = ((v0 - v_recon) ** 2).mean()
                    batch_losses.append(loss.item())

                    # Atualiza pesos e bias
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Atualiza a barra com a loss atual
                    pbar.set_postfix(
                        {
                            "batch_loss": f"{loss.item():.6f}",
                            "loss_média": f"{sum(batch_losses) / len(batch_losses):.6f}"
                        }
                    )

    def transformar_latente(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtém as ativações latentes (features) do RBM.

        Args:
            x (torch.Tensor): entradas (batch_size x n_visiveis)

        Returns:
            torch.Tensor: ativações latentes (batch_size x n_ocultos)
        """
        with torch.no_grad():
            return self.forward(x)
