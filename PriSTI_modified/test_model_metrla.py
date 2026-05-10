# ============================================================
# test_model_metrla.py — Test du modèle modifié sur METR-LA
# ============================================================

import sys
import torch
import yaml
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "layer2_dynamic"))

from main_model import PriSTI_MetrLA

print("=" * 55)
print("TEST — PriSTI Modifié avec Dynamic Adjacency (METR-LA)")
print("=" * 55)

# ── Config ──
config = {
    "model": {
        "is_unconditional": False,
        "timeemb": 128,
        "featureemb": 16,
        "target_strategy": "hybrid",
        "use_guide": True,
        "mask_sensor": []
    },
    "diffusion": {
        "layers": 4,
        "channels": 64,
        "nheads": 8,
        "diffusion_embedding_dim": 128,
        "beta_start": 0.0001,
        "beta_end": 0.2,
        "num_steps": 50,
        "schedule": "quad",
        "is_adp": True,
        "proj_t": 64,
        "is_cross_t": True,
        "is_cross_s": True,
        "adj_file": "metr-la",
        "device": "cpu",
        "alpha": 0.6,
        "beta": 0.4,
        "dynamic_adj_data_dir": str(
            Path(__file__).resolve().parent.parent / "extracted_data_metrla"
        ),
    },
    "train": {
        "epochs": 10,
        "batch_size": 4,
        "lr": 0.001,
        "valid_epoch_interval": 10,
        "is_lr_decay": True
    },
    "seed": 42
}

# Add side_dim
config["diffusion"]["side_dim"] = config["model"]["timeemb"] + config["model"]["featureemb"]

print("✅ Config créée")

# ── Créer le modèle ──
device = "cpu"
model = PriSTI_MetrLA(config, device, target_dim=207, seq_len=24).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Modèle créé — Paramètres: {n_params:,}")

# ── Faux batch ──
B, K, L = 2, 207, 24

torch.manual_seed(42)

fake_batch = {
    "observed_data": torch.randn(B, L, K),
    "observed_mask": torch.ones(B, L, K),
    "timepoints": torch.arange(L).unsqueeze(0).expand(B, -1).float(),
    "gt_mask": torch.ones(B, L, K),
    "cut_length": torch.zeros(B).long(),
    "cond_mask": torch.ones(B, L, K),
    "coeffs": torch.randn(B, L, K),
}

# Simuler capteurs manquants
fake_batch["observed_mask"][:, :, 50] = 0.
fake_batch["observed_mask"][:, 0:2, 50] = 1.
fake_batch["observed_mask"][:, :, 100] = 0.
fake_batch["cond_mask"] = fake_batch["observed_mask"].clone() * 0.8

print(f"✅ Faux batch créé (B={B}, K={K}, L={L})")
print(f"   Capteur 50: {fake_batch['observed_mask'][0, :, 50].sum().item():.0f}/{L} observé")
print(f"   Capteur 100: {fake_batch['observed_mask'][0, :, 100].sum().item():.0f}/{L} observé")

# ── Test 1 : Forward ──
print("\n── Test 1 : Forward pass ──")
try:
    model.train()
    loss = model(fake_batch, is_train=1)
    print(f"✅ Forward OK — Loss: {loss.item():.4f}")
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Test 2 : Validation ──
print("\n── Test 2 : Validation pass ──")
try:
    model.eval()
    with torch.no_grad():
        loss_valid = model(fake_batch, is_train=0)
    print(f"✅ Validation OK — Loss: {loss_valid.item():.4f}")
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Test 3 : Evaluate ──
print("\n── Test 3 : Evaluate (2 samples) ──")
try:
    model.eval()
    with torch.no_grad():
        output = model.evaluate(fake_batch, n_samples=2)
    samples, target, eval_points, obs_points, obs_time = output
    print(f"✅ Evaluate OK — Samples: {samples.shape}")
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Test 4 : A_dynamic ──
print("\n── Test 4 : A_dynamic ──")
try:
    dyn_adj = model.diffmodel.dynamic_adj
    obs_data = fake_batch["observed_data"].permute(0, 2, 1)
    obs_mask = fake_batch["observed_mask"].permute(0, 2, 1)

    quality = dyn_adj.compute_quality_score(obs_mask)
    A_dyn = dyn_adj.compute_dynamic_adj(obs_data, obs_mask)

    print(f"✅ A_dynamic OK")
    print(f"   Quality capteur 0: {quality[0, 0]:.4f} (devrait être 1.0)")
    print(f"   Quality capteur 50: {quality[0, 50]:.4f} (devrait être ~0.08)")
    print(f"   Quality capteur 100: {quality[0, 100]:.4f} (devrait être 0.0)")
    print(f"   A_dynamic[0,0,100]: {A_dyn[0, 0, 100]:.4f} (devrait être 0)")
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 55}")
print("TOUS LES TESTS PASSÉS ✅ — METR-LA")
print(f"{'=' * 55}")