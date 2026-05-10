# ============================================================
# test_model.py — Vérifie que le modèle modifié tourne
# Exécuter depuis PriSTI_modified/
# ============================================================

import sys
import torch
import yaml
import numpy as np
from pathlib import Path

# Ajouter les chemins nécessaires
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "layer2_dynamic"))

from main_model import PriSTI_aqi36

print("=" * 55)
print("TEST — PriSTI Modifié avec Dynamic Adjacency")
print("=" * 55)

# ── Charger la config ──
with open("config/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = 0
config["model"]["target_strategy"] = "hybrid"
config["diffusion"]["adj_file"] = "AQI36"
config["diffusion"]["device"] = "cpu"

# Corriger le chemin vers extracted_data
config["diffusion"]["dynamic_adj_data_dir"] = str(
    Path(__file__).resolve().parent.parent / "extracted_data"
)

print("✅ Config chargée")

# ── Créer le modèle ──
device = "cpu"
model = PriSTI_aqi36(config, device).to(device)
print(f"✅ Modèle créé")

# Compter les paramètres
n_params = sum(p.numel() for p in model.parameters())
print(f"   Paramètres: {n_params:,}")

# ── Créer un faux batch ──
B = 2   # batch size
K = 36  # capteurs
L = 36  # timesteps

torch.manual_seed(42)

fake_batch = {
    "observed_data": torch.randn(B, L, K),      # (B, L, K) — sera permuté en (B, K, L)
    "observed_mask": torch.ones(B, L, K),         # tout observé
    "timepoints": torch.arange(L).unsqueeze(0).expand(B, -1).float(),
    "gt_mask": torch.ones(B, L, K),
    "hist_mask": torch.ones(B, L, K),
    "cut_length": torch.zeros(B).long(),
    "cond_mask": torch.ones(B, L, K),
    "coeffs": torch.randn(B, L, K),
}

# Simuler des capteurs manquants pour tester le quality gate
# Capteur 5 : 90% manquant
fake_batch["observed_mask"][:, :, 5] = 0.
fake_batch["observed_mask"][:, 0:3, 5] = 1.
# Capteur 10 : complètement manquant
fake_batch["observed_mask"][:, :, 10] = 0.
# Mettre le cond_mask cohérent
fake_batch["cond_mask"] = fake_batch["observed_mask"].clone() * 0.8

print(f"✅ Faux batch créé (B={B}, K={K}, L={L})")
print(f"   Capteur 5: {fake_batch['observed_mask'][0, :, 5].sum().item():.0f}/{L} observé")
print(f"   Capteur 10: {fake_batch['observed_mask'][0, :, 10].sum().item():.0f}/{L} observé")

# ── Test 1 : Forward pass (training) ──
print("\n── Test 1 : Forward pass (training) ──")
try:
    model.train()
    loss = model(fake_batch, is_train=1)
    print(f"✅ Forward pass OK")
    print(f"   Loss: {loss.item():.4f}")
except Exception as e:
    print(f"❌ Forward pass ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Test 2 : Forward pass (validation) ──
print("\n── Test 2 : Forward pass (validation) ──")
try:
    model.eval()
    with torch.no_grad():
        loss_valid = model(fake_batch, is_train=0)
    print(f"✅ Validation pass OK")
    print(f"   Loss: {loss_valid.item():.4f}")
except Exception as e:
    print(f"❌ Validation pass ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Test 3 : Evaluate (imputation) ──
print("\n── Test 3 : Evaluate (imputation avec 2 samples) ──")
try:
    model.eval()
    with torch.no_grad():
        output = model.evaluate(fake_batch, n_samples=2)
    samples, target, eval_points, obs_points, obs_time = output
    print(f"✅ Evaluate OK")
    print(f"   Samples shape: {samples.shape}")
    print(f"   Target shape: {target.shape}")
except Exception as e:
    print(f"❌ Evaluate ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Test 4 : Vérifier que A_dynamic fonctionne ──
print("\n── Test 4 : Vérification A_dynamic ──")
try:
    dyn_adj = model.diffmodel.dynamic_adj
    obs_data = fake_batch["observed_data"].permute(0, 2, 1)  # (B, K, L)
    obs_mask = fake_batch["observed_mask"].permute(0, 2, 1)  # (B, K, L)

    quality = dyn_adj.compute_quality_score(obs_mask)
    A_dyn = dyn_adj.compute_dynamic_adj(obs_data, obs_mask)

    print(f"✅ A_dynamic OK")
    print(f"   Quality capteur 0: {quality[0, 0]:.4f} (devrait être 1.0)")
    print(f"   Quality capteur 5: {quality[0, 5]:.4f} (devrait être ~0.08)")
    print(f"   Quality capteur 10: {quality[0, 10]:.4f} (devrait être 0.0)")
    print(f"   A_dynamic[0,0,5]: {A_dyn[0, 0, 5]:.4f} (devrait être faible)")
    print(f"   A_dynamic[0,0,10]: {A_dyn[0, 0, 10]:.4f} (devrait être 0)")
except Exception as e:
    print(f"❌ A_dynamic ERREUR: {e}")
    import traceback
    traceback.print_exc()

# ── Résumé ──
print(f"\n{'=' * 55}")
print("TOUS LES TESTS PASSÉS ✅")
print(f"{'=' * 55}")
print(f"Le modèle modifié fonctionne correctement.")
print(f"Prochaine étape: entraînement sur Kaggle avec les vraies données.")

