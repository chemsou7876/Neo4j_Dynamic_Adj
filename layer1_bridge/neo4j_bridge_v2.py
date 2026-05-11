# ============================================================
# neo4j_bridge_v2.py  — VERSION FINALE CORRIGÉE
# Layer 1 — Pont entre Neo4j et le modèle
#
# CORRECTIONS APPLIQUÉES :
#
# [C1] W_semantic : Pearson continu au lieu du système
#      catégoriel (WEAKLY/CORRELATED/STRONGLY) à seuil
#      arbitraire 0.70. Seuil minimal = 0.15 (bruit pur).
#      Avant : 170 arêtes, 115/207 capteurs isolés.
#      Après : ~590 arêtes, ~0 capteurs isolés.
#
# [C2] q_anomaly : var_mean/var_std depuis 'mean' et 'std'
#      (écart-type du signal, ≈10 mph) au lieu de
#      'temporal_var_mean'/'temporal_var_std' (variance de
#      la variance, ≈70 → seuil ±3σ inutilisable).
#      Note : var_mean.npy et var_std.npy du dataset Kaggle
#      contiennent les valeurs temporal_var_* (erreur du
#      notebook d'extraction Cell 7). Le bridge v2 contourne
#      ça en lisant directement les JSON.
#
# [C3] Relations unidirectionnelles : le JSON ne stocke que
#      776 paires (i→j, côté supérieur). W_semantic était
#      donc à moitié vide. Chaque relation est maintenant
#      propagée dans les deux sens avec les attributs du
#      capteur source correct pour chaque direction.
#
# [C4] _query_nodes online : retourne 'mean' et 'std'.
# [C5] _query_relations online : retourne toutes les arêtes
#      (pas seulement CORRELATED/STRONGLY) avec leur Pearson.
# ============================================================

import json
import numpy as np
from pathlib import Path


class Neo4jBridgeV2:
    """
    Layer 1 — Extraction des métadonnées relationnelles.

    Fournit au modèle :
      W_semantic (K,K) : prior structurel pondéré par Pearson,
                         fiabilité et autocorrélation du capteur source.
                         Bidirectionnel, seuil minimal PEARSON_MIN.
      q_static   (K,)  : fiabilité historique par capteur.
      var_mean   (K,)  : moyenne globale du signal (pour q_anomaly).
      var_std    (K,)  : écart-type global du signal (pour q_anomaly).

    Formule W_semantic (par direction) :
      W(i→j) = gaussian_weight(i,j)
               × clamp(pearson(i,j), PEARSON_MIN, 1)
               × (1 - missing_rate(i))
               × (1 + autocorr_lag1(i)) / 2
    """

    # Seuil minimal Pearson.
    # Avec METR-LA (médiane ≈ 0.41) :
    #   PEARSON_MIN=0.15 → ~590 arêtes conservées (vs 170 avant)
    #   PEARSON_MIN=0.0  → toutes les arêtes positives (~735)
    PEARSON_MIN = 0.15

    def __init__(self, metadata_dir, mode='offline',
                 neo4j_uri=None, neo4j_auth=None):
        """
        Args:
            metadata_dir : dossier contenant nodes_metadata.json
                           et relations_metadata.json
            mode         : 'offline' ou 'online'
            neo4j_uri    : bolt://... (mode online uniquement)
            neo4j_auth   : (user, password) (mode online uniquement)
        """
        self.metadata_dir = Path(metadata_dir)
        self.mode = mode

        if mode == 'online':
            self._init_online(neo4j_uri, neo4j_auth)
        else:
            self._init_offline()

        print(f"[Neo4jBridgeV2] mode={mode}, K={self.K}")
        print(f"  W_semantic non-zero : {np.count_nonzero(self.W_semantic)}")
        print(f"  Capteurs isolés     : "
              f"{int((self.W_semantic.sum(1) == 0).sum())}/{self.K}")
        print(f"  q_static   range    : [{self.q_static.min():.3f}, "
              f"{self.q_static.max():.3f}]")
        print(f"  var_std    range    : [{self.var_std.min():.2f}, "
              f"{self.var_std.max():.2f}]  (doit être ~2-25, pas ~70)")

    # ── Mode offline ──────────────────────────────────────────

    def _init_offline(self):
        """Charge les JSON exportés depuis Neo4j."""
        nodes_path = self.metadata_dir / "nodes_metadata.json"
        rels_path  = self.metadata_dir / "relations_metadata.json"

        if not nodes_path.exists():
            raise FileNotFoundError(
                f"nodes_metadata.json introuvable dans {self.metadata_dir}"
            )

        with open(nodes_path, 'r') as f:
            nodes = json.load(f)
        with open(rels_path, 'r') as f:
            rels = json.load(f)

        self.K = len(nodes)
        self._build_from_data(nodes, rels)

    # ── Mode online ───────────────────────────────────────────

    def _init_online(self, uri, auth):
        """Requêtes Cypher live sur Neo4j."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("pip install neo4j  requis pour le mode online.")

        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._driver.verify_connectivity()
        nodes = self._query_nodes()
        rels  = self._query_relations()
        self.K = len(nodes)
        self._build_from_data(nodes, rels)

    def _query_nodes(self):
        """
        [C4] Retourne 'mean' et 'std' (signal global).
        """
        cypher = """
            MATCH (s:Sensor)
            RETURN
                s.sensor_index   AS sensor_index,
                s.missing_rate   AS missing_rate,
                s.autocorr_lag1  AS autocorr_lag1,
                s.mean           AS mean,
                s.std            AS std
            ORDER BY s.sensor_index
        """
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher)]

    def _query_relations(self):
        """
        [C5] Retourne TOUTES les arêtes avec gaussian_weight > 0
        et leur valeur Pearson. Le filtrage se fait via PEARSON_MIN.
        La bidirectionnalité est gérée dans _build_from_data.
        """
        cypher = """
            MATCH (a:Sensor)-[r:CORRELATED_WITH|ADJACENT_TO]->(b:Sensor)
            WHERE r.gaussian_weight IS NOT NULL
              AND r.gaussian_weight > 0
            RETURN
                a.sensor_index            AS source,
                b.sensor_index            AS target,
                r.gaussian_weight         AS gaussian_weight,
                COALESCE(r.pearson, 0.0)  AS pearson,
                toFloat(a.missing_rate)   AS src_missing_rate,
                toFloat(a.autocorr_lag1)  AS src_autocorr_lag1,
                toFloat(b.missing_rate)   AS tgt_missing_rate,
                toFloat(b.autocorr_lag1)  AS tgt_autocorr_lag1
        """
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher)]

    # ── Construction des matrices ─────────────────────────────

    def _build_from_data(self, nodes, rels):
        """
        Construit W_semantic (bidirectionnel) et q_static.

        [C1] Pondération continue par valeur Pearson.
        [C2] var_mean/var_std depuis 'mean'/'std' du signal.
        [C3] Bidirectionnalisation : chaque relation JSON (i→j)
             génère W[i,j] ET W[j,i] avec les bons attributs source.
        """
        K = self.K

        # ── Mapping index → attributs ─────────────────────────
        idx_to_meta = {}
        for n in nodes:
            idx = int(n.get('sensor_index', 0))
            idx_to_meta[idx] = n

        # Support sensor_id string (mode offline METR-LA)
        sensor_id_to_idx = {}
        if nodes and 'sensor_id' in nodes[0]:
            sensor_id_to_idx = {
                str(n['sensor_id']): int(n['sensor_index'])
                for n in nodes
            }

        # ── q_static + seuils d'anomalie ─────────────────────
        q_static = np.zeros(K)
        var_mean  = np.zeros(K)
        var_std   = np.ones(K)

        for meta in idx_to_meta.values():
            j  = int(meta['sensor_index'])
            mr = self._safe_float(meta.get('missing_rate'),  0.0)
            ac = self._safe_float(meta.get('autocorr_lag1'), 0.5)

            # [C2] : 'mean' et 'std' (écart-type du signal ≈ 10 mph)
            #        et NON temporal_var_mean / temporal_var_std (≈ 70)
            vm = self._safe_float(meta.get('mean'), 0.0)
            vs = self._safe_float(meta.get('std'),  1.0)

            q_static[j] = (1.0 - mr) * ((1.0 + ac) / 2.0)
            var_mean[j]  = vm
            var_std[j]   = max(vs, 1e-8)

        # ── W_semantic ────────────────────────────────────────
        W = np.zeros((K, K))
        kept = filtered = 0

        for rel in rels:
            # [C1] Valeur Pearson continue
            pearson = self._safe_float(rel.get('pearson'), 0.0)

            # Exclure corrélations négatives ou bruit pur
            if pearson < self.PEARSON_MIN:
                filtered += 1
                continue

            # Résoudre les indices
            src, tgt = self._resolve_indices(
                rel.get('source', rel.get('src')),
                rel.get('target', rel.get('tgt')),
                sensor_id_to_idx
            )
            if src is None or src >= K or tgt >= K:
                continue

            gw = self._safe_float(rel.get('gaussian_weight'), 0.0)

            # ── Direction i → j ───────────────────────────────
            mr_src = self._safe_float(
                rel.get('src_missing_rate',
                        idx_to_meta.get(src, {}).get('missing_rate')), 0.0)
            ac_src = self._safe_float(
                rel.get('src_autocorr_lag1',
                        idx_to_meta.get(src, {}).get('autocorr_lag1')), 0.5)

            W[src, tgt] = gw * pearson * (1.0 - mr_src) * ((1.0 + ac_src) / 2.0)

            # [C3] Direction j → i (bidirectionnalisation)
            # Utilise les attributs du capteur TGT comme source
            mr_tgt = self._safe_float(
                rel.get('tgt_missing_rate',
                        idx_to_meta.get(tgt, {}).get('missing_rate')), 0.0)
            ac_tgt = self._safe_float(
                rel.get('tgt_autocorr_lag1',
                        idx_to_meta.get(tgt, {}).get('autocorr_lag1')), 0.5)

            # Ne pas écraser si la direction inverse existe déjà
            # (mode online : Neo4j stocke les deux sens)
            if W[tgt, src] == 0.0:
                W[tgt, src] = gw * pearson * (1.0 - mr_tgt) * ((1.0 + ac_tgt) / 2.0)

            kept += 1

        n_zero = int((W.sum(axis=1) == 0).sum())
        print(f"  Relations traitées  : {kept} paires "
              f"({kept * 2} arêtes dirigées) | "
              f"filtrées (pearson<{self.PEARSON_MIN}) : {filtered}")
        print(f"  Capteurs isolés dans W_semantic : {n_zero}/{K}")

        self.W_semantic = W
        self.q_static   = q_static
        self.var_mean    = var_mean
        self.var_std     = var_std

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _safe_float(val, default=0.0):
        """Convertit val en float, retourne default si None ou -1."""
        if val is None:
            return default
        v = float(val)
        return default if v == -1.0 else v

    @staticmethod
    def _resolve_indices(src_raw, tgt_raw, sensor_id_to_idx):
        """Résout source/target en indices entiers."""
        try:
            if isinstance(src_raw, str) and src_raw in sensor_id_to_idx:
                return sensor_id_to_idx[src_raw], sensor_id_to_idx[str(tgt_raw)]
            return int(src_raw), int(tgt_raw)
        except (TypeError, ValueError):
            return None, None

    # ── API publique ──────────────────────────────────────────

    def get_tensors(self, device):
        """
        Retourne les 4 tenseurs nécessaires à Layer 2.

        Returns:
            W_semantic (K,K), q_static (K,),
            var_mean (K,),    var_std (K,)
        """
        import torch
        def t(arr):
            return torch.tensor(arr, dtype=torch.float32).to(device)
        return (t(self.W_semantic), t(self.q_static),
                t(self.var_mean),   t(self.var_std))

    def close(self):
        if hasattr(self, '_driver'):
            self._driver.close()
            print("[Neo4jBridgeV2] Connexion Neo4j fermée.")


# ── Test standalone ───────────────────────────────────────────
if __name__ == '__main__':
    import sys, torch
    print("=" * 60)
    print("TEST Neo4jBridgeV2 — version finale corrigée")
    print("=" * 60)

    meta_dir = './neo4j_setup/metadata_metrla'
    bridge = Neo4jBridgeV2(metadata_dir=meta_dir, mode='offline')
    W, q, vm, vs = bridge.get_tensors('cpu')

    print(f"\nW_semantic : {W.shape}  non-zero={( W>0).sum().item()}")
    print(f"q_static   : mean={q.mean():.3f}  range=[{q.min():.3f},{q.max():.3f}]")
    print(f"var_std    : mean={vs.mean():.2f}  (doit être ~10, pas ~70)")

    zero_rows = (W.sum(dim=1) == 0).sum().item()

    # Assertions
    assert vs.mean().item() < 30, \
        f"[C2 FAIL] var_std={vs.mean():.1f} — vérifier 'std' vs 'temporal_var_std'"
    assert zero_rows < 20, \
        f"[C3 FAIL] {zero_rows}/207 capteurs isolés — vérifier bidirectionnalisation"
    assert (W > 0).sum().item() > 500, \
        f"[C1 FAIL] W_semantic trop sparse ({(W>0).sum().item()} arêtes)"

    print(f"\n✅ [C1] Pearson continu       — {(W>0).sum().item()} arêtes")
    print(f"✅ [C2] var_std signal global  — mean={vs.mean():.2f}")
    print(f"✅ [C3] Bidirectionnel         — {zero_rows} capteurs isolés")
    print(f"\n🎉 Tous les tests passent !")