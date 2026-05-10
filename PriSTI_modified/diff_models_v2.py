# ============================================================
# diff_models_v2.py
# Layer 3 — PriSTI modifié pour DynamicAdjacencyV2
#
# Changements par rapport à diff_models.py v1 :
#   - Import de Neo4jBridgeV2 + DynamicAdjacencyV2
#   - Guide_diff.__init__ : instancie les deux layers
#   - compute_dynamic_support : délègue à DynamicAdjacencyV2.get_support()
#   - forward : signature inchangée, interface identique
# ============================================================

import sys
import math
from pathlib import Path
from layers import *

# ── Layer 1 + Layer 2 ─────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent / "layer1_bridge"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "layer2_dynamic"))
from neo4j_bridge_v2 import Neo4jBridgeV2
from dynamic_adjacency_v2 import DynamicAdjacencyV2


class Guide_diff(nn.Module):
    def __init__(self, config, inputdim=1, target_dim=36, is_itp=False):
        super().__init__()
        self.channels = config["channels"]
        self.is_itp   = is_itp
        self.device   = config["device"]

        if self.is_itp:
            self.itp_channels    = config["channels"]
            self.itp_projection  = Conv1d_with_init(
                inputdim - 1, self.itp_channels, 1)
            self.itp_modeling    = GuidanceConstruct(
                channels=self.itp_channels,
                nheads=config["nheads"],
                target_dim=target_dim,
                order=2, include_self=True,
                device=config["device"],
                is_adp=config["is_adp"],
                adj_file=config["adj_file"],
                proj_t=config["proj_t"],
            )
            self.cond_projection  = Conv1d_with_init(
                config["side_dim"], self.itp_channels, 1)
            self.itp_projection2  = Conv1d_with_init(
                self.itp_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # ── Layer 1 : pont Neo4j ──────────────────────────────
        # mode='offline' par défaut (Kaggle).
        # Pour le mode online, passer neo4j_mode='online',
        # neo4j_uri et neo4j_auth dans la config.
        bridge = Neo4jBridgeV2(
            metadata_dir=config.get(
                "metadata_dir",
                "./neo4j_setup/metadata_metrla"
            ),
            mode=config.get("neo4j_mode", "offline"),
            neo4j_uri=config.get("neo4j_uri",   None),
            neo4j_auth=config.get("neo4j_auth",  None),
        )

        # ── Layer 2 : adjacence dynamique V2 ─────────────────
        self.dynamic_adj = DynamicAdjacencyV2(
            bridge=bridge,
            alpha_max=config.get("alpha_max", 0.8),
            alpha_min=config.get("alpha_min", 0.4),
            device=config["device"],
        )

        self.input_projection   = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList([
            NoiseProject(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
                target_dim=target_dim,
                proj_t=config["proj_t"],
                is_adp=config["is_adp"],
                device=config["device"],
                adj_file=config["adj_file"],
                is_cross_t=config["is_cross_t"],
                is_cross_s=config["is_cross_s"],
            )
            for _ in range(config["layers"])
        ])

    def compute_dynamic_support(self, observed_data, cond_mask):
        """
        Délègue à DynamicAdjacencyV2.get_support().
        Retourne [A_fwd, A_bwd], chacun (B, K, K).
        """
        return self.dynamic_adj.get_support(observed_data, cond_mask)

    def forward(self, x, side_info, diffusion_step, itp_x, cond_mask,
                observed_data, observed_mask):
        """
        Signature identique à v1 — aucun changement côté appelant.
        observed_mask conservé en paramètre pour compatibilité mais
        non utilisé ici (cond_mask suffit pour Layer 2).
        """
        if self.is_itp:
            x = torch.cat([x, itp_x], dim=1)
        B, inputdim, K, L = x.shape

        x = self.input_projection(
            x.reshape(B, inputdim, K * L))
        x = F.relu(x).reshape(B, self.channels, K, L)

        # Layer 2 — support dynamique (toujours cond_mask)
        support = self.compute_dynamic_support(observed_data, cond_mask)

        if self.is_itp:
            itp_x2 = self.itp_projection(
                itp_x.reshape(B, inputdim - 1, K * L))
            itp_cond = self.cond_projection(
                side_info.reshape(B, -1, K * L))
            itp_x2 = F.relu(
                self.itp_modeling(
                    itp_x2 + itp_cond,
                    [B, self.itp_channels, K, L], support
                )
            ).reshape(B, self.itp_channels, K, L)
        else:
            itp_x2 = itp_x

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, s = layer(x, side_info, diffusion_emb, itp_x2, support)
            skip.append(s)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(
            len(self.residual_layers))
        x = F.relu(self.output_projection1(
            x.reshape(B, self.channels, K * L)))
        return self.output_projection2(x).reshape(B, K, L)


class NoiseProject(nn.Module):
    """Inchangé par rapport à PriSTI original."""
    def __init__(self, side_dim, channels, diffusion_embedding_dim,
                 nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None,
                 is_cross_t=False, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(
            diffusion_embedding_dim, channels)
        self.cond_projection  = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection   = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.forward_time    = TemporalLearning(
            channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(
            channels=channels, nheads=nheads, target_dim=target_dim,
            order=order, include_self=include_self, device=device,
            is_adp=is_adp, adj_file=adj_file, proj_t=proj_t,
            is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base = x.shape
        x    = x.reshape(B, channel, K * L)
        y    = x + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y    = self.forward_time(y, base, itp_info)
        y    = self.forward_feature(y, base, support, itp_info)
        y    = self.mid_projection(y)
        si   = self.cond_projection(
            side_info.reshape(B, -1, K * L))
        y    = y + si
        gate, filt = torch.chunk(y, 2, dim=1)
        y    = self.output_projection(
            torch.sigmoid(gate) * torch.tanh(filt))
        res, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base)
        return (x + res.reshape(base)) / math.sqrt(2.0), skip.reshape(base)