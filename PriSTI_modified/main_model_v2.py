# ============================================================
# main_model_v2.py
# Identique à main_model.py — seule différence :
#   from diff_models_v2 import Guide_diff
# Toute la logique Layer 1 + Layer 2 est dans diff_models_v2.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from diff_models_v2 import Guide_diff   # ← seul changement vs v1


class PriSTI(nn.Module):
    def __init__(self, target_dim, seq_len, config, device):
        super().__init__()
        self.device           = device
        self.target_dim       = target_dim
        self.seq_len          = seq_len
        self.emb_time_dim     = config["model"]["timeemb"]
        self.emb_feature_dim  = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy  = config["model"]["target_strategy"]
        self.use_guide        = config["model"]["use_guide"]

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer   = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim
        )

        config_diff             = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"]   = device

        input_dim      = 2
        self.diffmodel = Guide_diff(
            config_diff, input_dim, target_dim, self.use_guide)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5,
                config_diff["beta_end"]   ** 0.5,
                self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"],
                config_diff["beta_end"],
                self.num_steps
            )

        self.alpha_hat   = 1 - self.beta
        self.alpha       = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float()
            .to(self.device).unsqueeze(1).unsqueeze(1)
        )

    # ── Embeddings ────────────────────────────────────────────

    def time_embedding(self, pos, d_model=128):
        pe       = torch.zeros(pos.shape[0], pos.shape[1], d_model
                               ).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device))
        feature_embed = (feature_embed.unsqueeze(0).unsqueeze(0)
                         .expand(B, L, -1, -1))
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        return side_info.permute(0, 3, 2, 1)

    # ── Loss ──────────────────────────────────────────────────

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask,
                        side_info, itp_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask,
                side_info, itp_info, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask,
                  side_info, itp_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        t = ((torch.ones(B) * set_t).long().to(self.device)
             if is_train != 1
             else torch.randint(0, self.num_steps, [B]).to(self.device))

        current_alpha = self.alpha_torch[t]
        noise         = torch.randn_like(observed_data)
        noisy_data    = ((current_alpha ** 0.5) * observed_data
                         + (1.0 - current_alpha) ** 0.5 * noise)
        total_input   = self.set_input_to_diffmodel(
            noisy_data, observed_data, cond_mask)

        if not self.use_guide:
            itp_info = cond_mask * observed_data

        # cond_mask passé comme adj_cond_mask — jamais observed_mask
        predicted = self.diffmodel(
            total_input, side_info, t, itp_info,
            cond_mask, observed_data, cond_mask
        )

        target_mask = observed_mask - cond_mask
        residual    = (noise - predicted) * target_mask
        num_eval    = target_mask.sum()
        return (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        if not self.use_guide:
            cond_obs    = (cond_mask * observed_data).unsqueeze(1)
            noisy_tgt   = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            return torch.cat([cond_obs, noisy_tgt], dim=1)
        return ((1 - cond_mask) * noisy_data).unsqueeze(1)

    # ── Imputation ────────────────────────────────────────────

    def impute(self, observed_data, cond_mask, side_info,
               n_samples, itp_info, observed_mask):
        B, K, L    = observed_data.shape
        imputed    = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            if self.is_unconditional:
                noisy_obs  = observed_data
                noisy_hist = []
                for t in range(self.num_steps):
                    noise     = torch.randn_like(noisy_obs)
                    noisy_obs = ((self.alpha_hat[t] ** 0.5) * noisy_obs
                                 + self.beta[t] ** 0.5 * noise)
                    noisy_hist.append(noisy_obs * cond_mask)

            current = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (cond_mask * noisy_hist[t]
                                  + (1.0 - cond_mask) * current)
                    diff_input = diff_input.unsqueeze(1)
                else:
                    if not self.use_guide:
                        cond_obs   = (cond_mask * observed_data).unsqueeze(1)
                        noisy_tgt  = ((1 - cond_mask) * current).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_tgt], dim=1)
                    else:
                        diff_input = ((1 - cond_mask) * current).unsqueeze(1)

                predicted = self.diffmodel(
                    diff_input, side_info,
                    torch.tensor([t]).to(self.device),
                    itp_info, cond_mask,
                    observed_data, cond_mask   # cond_mask, jamais observed_mask
                )

                c1      = 1 / self.alpha_hat[t] ** 0.5
                c2      = ((1 - self.alpha_hat[t])
                           / (1 - self.alpha[t]) ** 0.5)
                current = c1 * (current - c2 * predicted)

                if t > 0:
                    sigma    = ((1.0 - self.alpha[t - 1])
                                / (1.0 - self.alpha[t])
                                * self.beta[t]) ** 0.5
                    current += sigma * torch.randn_like(current)

            imputed[:, i] = current.detach()
        return imputed

    # ── forward / evaluate ────────────────────────────────────

    def forward(self, batch, is_train=1):
        (observed_data, observed_mask, observed_tp, gt_mask,
         for_pattern_mask, _, coeffs, cond_mask) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info  = coeffs.unsqueeze(1) if self.use_guide else None

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask,
                         side_info, itp_info, is_train)

    def evaluate(self, batch, n_samples):
        (observed_data, observed_mask, observed_tp, gt_mask,
         _, cut_length, coeffs, _) = self.process_data(batch)

        with torch.no_grad():
            cond_mask   = gt_mask
            target_mask = observed_mask - cond_mask
            side_info   = self.get_side_info(observed_tp, cond_mask)
            itp_info    = coeffs.unsqueeze(1) if self.use_guide else None
            samples     = self.impute(
                observed_data, cond_mask, side_info,
                n_samples, itp_info, observed_mask)
            for i in range(len(cut_length)):
                target_mask[i, ..., :cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp


# ── Sous-classes inchangées ───────────────────────────────────

class PriSTI_aqi36(PriSTI):
    def __init__(self, config, device, target_dim=36, seq_len=36):
        super().__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        dev  = self.device
        od   = batch["observed_data"].to(dev).float()
        om   = batch["observed_mask"].to(dev).float()
        tp   = batch["timepoints"].to(dev).float()
        gm   = batch["gt_mask"].to(dev).float()
        cl   = batch["cut_length"].to(dev).long()
        fpm  = batch["hist_mask"].to(dev).float()
        co   = (batch["coeffs"].to(dev).float()
                if self.config["model"]["use_guide"] else None)
        cm   = batch["cond_mask"].to(dev).float()
        for t in [od, om, gm, fpm, cm]:
            t.data = t.permute(0, 2, 1)
        if co is not None:
            co = co.permute(0, 2, 1)
        return od, om, tp, gm, fpm, cl, co, cm


class PriSTI_MetrLA(PriSTI):
    def __init__(self, config, device, target_dim=207, seq_len=24):
        super().__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        dev  = self.device
        od   = batch["observed_data"].to(dev).float()
        om   = batch["observed_mask"].to(dev).float()
        tp   = batch["timepoints"].to(dev).float()
        gm   = batch["gt_mask"].to(dev).float()
        cl   = batch["cut_length"].to(dev).long()
        co   = (batch["coeffs"].to(dev).float()
                if self.config["model"]["use_guide"] else None)
        cm   = batch["cond_mask"].to(dev).float()
        for t in [od, om, gm, cm]:
            t.data = t.permute(0, 2, 1)
        if co is not None:
            co = co.permute(0, 2, 1)
        fpm = om
        return od, om, tp, gm, fpm, cl, co, cm


class PriSTI_PemsBAY(PriSTI):
    def __init__(self, config, device, target_dim=325, seq_len=24):
        super().__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        dev  = self.device
        od   = batch["observed_data"].to(dev).float()
        om   = batch["observed_mask"].to(dev).float()
        tp   = batch["timepoints"].to(dev).float()
        gm   = batch["gt_mask"].to(dev).float()
        cl   = batch["cut_length"].to(dev).long()
        co   = (batch["coeffs"].to(dev).float()
                if self.config["model"]["use_guide"] else None)
        cm   = batch["cond_mask"].to(dev).float()
        for t in [od, om, gm, cm]:
            t.data = t.permute(0, 2, 1)
        if co is not None:
            co = co.permute(0, 2, 1)
        fpm = om
        return od, om, tp, gm, fpm, cl, co, cm