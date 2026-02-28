from typing import Any, Tuple
import torch.nn as nn
import torch
from torch import Tensor
from tqdm import trange


class RBMYoutube(nn.Module):
    # Definindo os tipos explicitamente como sugerido no teu código
    pesos: nn.Parameter
    vies_visivel: nn.Parameter
    vies_oculto: nn.Parameter

    def __init__(self, n_visiveis: int, n_ocultos: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Inicialização levemente ajustada para melhor convergência
        self.pesos = nn.Parameter(torch.randn(n_visiveis, n_ocultos) * 0.1)
        self.vies_visivel = nn.Parameter(torch.zeros(n_visiveis))
        self.vies_oculto = nn.Parameter(torch.zeros(n_ocultos))

    def ativar_unidades_ocultas(self, tensor_visivel: Tensor) -> Tuple[Tensor, Tensor]:
        """Calcula a probabilidade de ativação das camadas ocultas (h)"""
        prob = torch.sigmoid(tensor_visivel @ self.pesos + self.vies_oculto)
        amostra = torch.bernoulli(prob)
        return prob, amostra

    def reconstruir_unidade_visivel(self, tensor_oculto: Tensor) -> Tensor:
        """Reconstrói a entrada (v). Removido o Sigmoide para suportar valores de PCA"""
        media = tensor_oculto @ self.pesos.t() + self.vies_visivel
        return media

    def divergencia_constrastiva(self, dados: Tensor, taxa: float = 0.01):
        """Implementação do algoritmo CD-1 para atualização de pesos"""
        # Desativamos o autograd para operações manuais de pesos (mais eficiente)
        with torch.no_grad():
            v0 = dados
            prob_h0, h0 = self.ativar_unidades_ocultas(v0)

            # Fase de reconstrução (passo negativo)
            v1 = self.reconstruir_unidade_visivel(h0)
            prob_h1, _ = self.ativar_unidades_ocultas(v1)

            # Cálculo do gradiente simplificado e atualização
            batch_size = v0.size(0)

            # Atualização dos Pesos (W)
            # Diferença entre a correlação inicial (v0, h0) e a reconstruída (v1, h1)
            pos_gradient = v0.t() @ prob_h0
            neg_gradient = v1.t() @ prob_h1
            self.pesos += taxa * (pos_gradient - neg_gradient) / batch_size

            # Atualização dos Biases (vieses)
            self.vies_visivel += taxa * torch.mean(v0 - v1, dim=0)
            self.vies_oculto += taxa * torch.mean(prob_h0 - prob_h1, dim=0)

    def treinar_rede(self, dados: Tensor, epocas: int = 100, taxa: float = 0.01, tamanho_batch: int = None):
        """Loop de treinamento"""
        for epoca in trange(epocas, desc="Treinamento RBM"):
            if tamanho_batch:
                # Embaralhar os dados para um treino mais robusto
                indices = torch.randperm(dados.size(0))
                dados_shuffled = dados[indices]

                for i in range(0, dados.size(0), tamanho_batch):
                    batch = dados_shuffled[i:i + tamanho_batch]
                    self.divergencia_constrastiva(batch, taxa=taxa)
            else:
                self.divergencia_constrastiva(dados, taxa=taxa)

    def obter_ativacoes(self, dados: Tensor) -> Tensor:
        """Retorna apenas as probabilidades das unidades ocultas"""
        prob, _ = self.ativar_unidades_ocultas(dados)
        return prob

    def propagacao_direta(self, dados: Tensor) -> Tensor:
        """Alias para manter compatibilidade com o resto do teu script"""
        return self.obter_ativacoes(dados)