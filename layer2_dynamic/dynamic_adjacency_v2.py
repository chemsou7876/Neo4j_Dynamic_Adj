# ============================================================
# dynamic_adjacency_v2.py
# Layer 2 — Calcul de l'adjacence dynamique enrichie
#
# Formule finale :
#
#   A_dyn(b,i,j) = [ α(b)·W_semantic(i,j) + β(b)·ρ_local(b,i,j) ]
#                  × q_static(j)
#                  × q_avail(b,j)
#                  × q_anomaly(b,j)
#
# où :
#   W_semantic  = prior structurel filtré par type de relation
#                 (Neo4j, Layer 1)
#   ρ_local     = corrélation de Pearson sur le batch courant
#   q_static(j) = fiabilité historique du capteur j (Neo4j, L1)
#   q_avail     = disponibilité dans la fenêtre courante
#   q_anomaly   = proportion de valeurs dans [μ±3σ]
#   α(b), β(b)  = poids adaptatifs selon la disponibilité du batch
# ============================================================

import numpy as np
import torch
from pathlib import Path


class DynamicAdjacencyV2:
    """
    Layer 2 — Adjacence dynamique guidée par Neo4j (Layer 1).

    Améliorations par rapport à DynamicAdjacency v1 :
      1. W_semantic remplace A_static (masque sémantique inclus)
      2. q_static depuis Neo4j (fiabilité historique)
      3. α/β adaptatifs selon la disponibilité du batch
      4. quality = q_static × q_avail × q_anomaly (3 facteurs)
    """

    def __init__(self, bridge, alpha_max=0.8, alpha_min=0.4, device=None):
        """
        Args:
            bridge     : instance de Neo4jBridgeV2 (Layer 1)
            alpha_max  : poids max de W_semantic (batch vide)
            alpha_min  : poids min de W_semantic (batch plein)
            device     : 'cuda' ou 'cpu'
        """
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.device    = device or torch.device('cpu')

        # Charger tous les tenseurs depuis Layer 1
        (self.W_semantic,
         self.q_static,
         self.var_mean,
         self.var_std) = bridge.get_tensors(self.device)

        self.K = self.W_semantic.shape[0]

        print(f"[DynamicAdjacencyV2] K={self.K}, "
              f"α∈[{alpha_min},{alpha_max}]")
        print(f"  W_semantic non-zero : {(self.W_semantic > 0).sum().item()}")
        print(f"  q_static   mean     : {self.q_static.mean().item():.4f}")

    # ── Composants de la formule ──────────────────────────────

    def _adaptive_alpha(self, cond_mask):
        """
        Calcule α(b) adaptatif pour chaque exemple du batch.

        Quand le batch est très incomplet (faible disponibilité
        moyenne), on fait davantage confiance au prior structurel
        W_semantic. Quand les observations sont denses, on laisse
        ρ_local prendre plus de poids.

        α(b) = α_max - (α_max - α_min) × q̄_avail(b)

        Returns:
            alpha : (B,)  valeurs dans [α_min, α_max]
            beta  : (B,)  = 1 - alpha
        """
        q_avail_mean = cond_mask.float().mean(dim=(1, 2))  # (B,)
        alpha = (self.alpha_max
                 - (self.alpha_max - self.alpha_min) * q_avail_mean)
        return alpha, 1.0 - alpha

    def _q_avail(self, cond_mask):
        """
        Disponibilité par capteur dans la fenêtre courante.

        Returns: (B, K)
        """
        return cond_mask.float().mean(dim=-1)

    def _q_anomaly(self, observed_data, cond_mask):
        """
        Proportion de valeurs dans la plage [μ_var ± 3σ_var].
        Calculée uniquement sur les timesteps conditionnels.

        Returns: (B, K)
        """
        mean_k = self.var_mean.unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
        std_k  = self.var_std.unsqueeze(0).unsqueeze(-1)   # (1, K, 1)

        z     = (observed_data - mean_k) / std_k.clamp(min=1e-8)
        ok    = (z.abs() <= 3.0).float()                   # (B, K, L)
        mask  = cond_mask.float()
        n_obs = mask.sum(dim=-1).clamp(min=1)              # (B, K)
        return (ok * mask).sum(dim=-1) / n_obs

    def _quality_gate(self, observed_data, cond_mask):
        """
        Score de qualité composite (3 facteurs) pour chaque capteur.

        q(b, j) = q_static(j) × q_avail(b, j) × q_anomaly(b, j)

        Returns: (B, K)
        """
        q_stat = self.q_static.unsqueeze(0)          # (1, K) → broadcast
        q_av   = self._q_avail(cond_mask)             # (B, K)
        q_an   = self._q_anomaly(observed_data, cond_mask)  # (B, K)
        return q_stat * q_av * q_an

    def _local_pearson(self, observed_data, cond_mask):
        """
        Corrélation de Pearson locale calculée sur les timesteps
        conditionnels du batch courant.

        Returns: (B, K, K)  — valeurs ∈ [0, 1], diagonale = 0
        """
        B, K, L = observed_data.shape
        mask    = cond_mask.float()
        masked  = observed_data * mask

        n_obs   = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_x  = masked.sum(dim=-1, keepdim=True) / n_obs
        centered = (masked - mean_x) * mask

        co_obs   = torch.bmm(mask, mask.transpose(1, 2)).clamp(min=1)
        cov      = torch.bmm(centered, centered.transpose(1, 2)) / co_obs

        std      = torch.sqrt(
            (centered ** 2).sum(dim=-1) / n_obs.squeeze(-1).clamp(min=1)
        ).clamp(min=1e-8)

        std_outer = std.unsqueeze(-1) * std.unsqueeze(-2)
        corr      = (cov / std_outer.clamp(min=1e-8)).clamp(0.0, 1.0)

        eye  = torch.eye(K, device=self.device).unsqueeze(0)
        return corr * (1.0 - eye)

    # ── Point d'entrée principal ──────────────────────────────

    def compute_dynamic_adj(self, observed_data, cond_mask):
        """
        Calcule A_dyn(b) pour chaque exemple du batch.

        Formule :
          A_dyn(b,i,j) =
            [ α(b)·W_semantic(i,j) + β(b)·ρ_local(b,i,j) ]
            × q_static(j) × q_avail(b,j) × q_anomaly(b,j)

        Args:
            observed_data : (B, K, L)
            cond_mask     : (B, K, L)  — masque conditionnel UNIQUEMENT
                            (jamais observed_mask)
        Returns:
            A_dyn : (B, K, K)
        """
        B, K, L = observed_data.shape

        # ── 1. Poids adaptatifs ───────────────────────────────
        alpha, beta = self._adaptive_alpha(cond_mask)   # (B,), (B,)

        # ── 2. Matrices source ────────────────────────────────
        W_sem = self.W_semantic.unsqueeze(0).expand(B, -1, -1)  # (B,K,K)
        rho   = self._local_pearson(observed_data, cond_mask)   # (B,K,K)

        # Combinaison avec α/β adaptatifs (broadcast sur K×K)
        alpha_3d = alpha.view(B, 1, 1)
        beta_3d  = beta.view(B, 1, 1)
        adj_combined = alpha_3d * W_sem + beta_3d * rho          # (B,K,K)

        # ── 3. Quality gate (3 facteurs) ─────────────────────
        quality = self._quality_gate(observed_data, cond_mask)  # (B, K)
        # Appliquer sur la dimension j (capteur source d'information)
        quality_gate = quality.unsqueeze(1).expand(-1, K, -1)   # (B,K,K)

        A_dyn = adj_combined * quality_gate                      # (B,K,K)
        return A_dyn

    def normalize(self, adj):
        """
        Normalisation D^{-1}·A par ligne (row-stochastic).
        Protégée contre les lignes nulles.

        Args:  adj : (B, K, K)
        Returns:     (B, K, K)
        """
        rs = adj.sum(dim=-1, keepdim=True)
        rs = torch.where(rs == 0, torch.ones_like(rs), rs)
        return adj / rs

    def get_support(self, observed_data, cond_mask):
        """
        Interface principale appelée par diff_models.py.
        Retourne [A_fwd, A_bwd] normalisées, comme avant.

        Returns: [A_fwd (B,K,K), A_bwd (B,K,K)]
        """
        A = self.compute_dynamic_adj(observed_data, cond_mask)
        return [self.normalize(A), self.normalize(A.transpose(1, 2))]


# ── Test standalone ───────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../layer1_bridge')
    from neo4j_bridge_v2 import Neo4jBridgeV2

    bridge = Neo4jBridgeV2(
        metadata_dir='../neo4j_setup/metadata_metrla',
        mode='offline'
    )
    dyn = DynamicAdjacencyV2(bridge, alpha_max=0.8, alpha_min=0.4)

    B, K, L = 4, 207, 24
    torch.manual_seed(42)
    obs   = torch.randn(B, K, L)
    mask  = (torch.rand(B, K, L) > 0.25).float()

    A_fwd, A_bwd = dyn.get_support(obs, mask)
    print(f"\nA_fwd shape : {A_fwd.shape}")
    print(f"A_fwd range : [{A_fwd.min():.4f}, {A_fwd.max():.4f}]")
    print(f"A_fwd non-zero : {(A_fwd > 0).sum().item()}")

    # Vérifier que α est bien adaptatif
    alpha, beta = dyn._adaptive_alpha(mask)
    print(f"\nα adaptatif (4 batchs) : {alpha.tolist()}")
    print(f"β adaptatif (4 batchs) : {beta.tolist()}")

    # Vérifier q_static
    q = dyn.q_static
    print(f"\nq_static min={q.min():.3f} max={q.max():.3f}")
    print("✓ DynamicAdjacencyV2 OK")