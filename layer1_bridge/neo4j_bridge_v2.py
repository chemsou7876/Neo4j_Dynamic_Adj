# ============================================================
# neo4j_bridge_v2.py
# Layer 1 — Pont entre Neo4j et le modèle
#
# Deux modes de fonctionnement :
#   1. OFFLINE (Kaggle / production) : lit les JSON exportés
#      depuis Neo4j, construit W_semantic et q_static sans
#      connexion active à la base.
#   2. ONLINE  (local) : requête Cypher live sur Neo4j,
#      résultats identiques au mode offline.
#
# Dans les deux cas, le modèle reçoit exactement les mêmes
# objets. La commutation est transparente.
# ============================================================

import json
import numpy as np
from pathlib import Path


class Neo4jBridgeV2:
    """
    Layer 1 — Extraction et gestion des métadonnées relationnelles.

    Fournit au modèle :
      - W_semantic  (K, K) : matrice structurelle filtrée par
                             type de relation (masque sémantique)
      - q_static    (K,)   : fiabilité historique par capteur
      - var_mean    (K,)   : moyenne globale (anomaly detection)
      - var_std     (K,)   : écart-type global (anomaly detection)

    Paramètres de filtrage sémantique :
      rel_type_weights = {
          'STRONGLY_CORRELATED': 1.5,   # bonus sémantique fort
          'CORRELATED':          1.0,   # relation standard
          'WEAKLY_CORRELATED':   0.0,   # exclu du graphe actif
      }
    """

    REL_WEIGHTS = {
        'STRONGLY_CORRELATED': 1.5,
        'CORRELATED':          1.0,
        'WEAKLY_CORRELATED':   0.0,   # masque sémantique : exclu
    }

    def __init__(self, metadata_dir, mode='offline',
                 neo4j_uri=None, neo4j_auth=None):
        """
        Args:
            metadata_dir : chemin vers le dossier contenant
                           nodes_metadata.json et
                           relations_metadata.json
            mode         : 'offline' (JSON) ou 'online' (Neo4j live)
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
        print(f"  q_static   range    : [{self.q_static.min():.3f}, "
              f"{self.q_static.max():.3f}]")

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

    # ── Mode online (Neo4j live) ──────────────────────────────

    def _init_online(self, uri, auth):
        """Interroge Neo4j via Cypher pour construire les matrices."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Package 'neo4j' requis pour le mode online.\n"
                "Installez-le avec : pip install neo4j"
            )

        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._driver.verify_connectivity()

        nodes = self._query_nodes()
        rels  = self._query_relations()
        self.K = len(nodes)
        self._build_from_data(nodes, rels)

    def _query_nodes(self):
        """Récupère les métadonnées de tous les capteurs."""
        cypher = """
            MATCH (s:Sensor)
            RETURN
                s.sensor_index    AS sensor_index,
                s.missing_rate    AS missing_rate,
                s.missing_category AS missing_category,
                s.autocorr_lag1   AS autocorr_lag1,
                s.stability       AS stability,
                s.temporal_var_mean AS temporal_var_mean,
                s.temporal_var_std  AS temporal_var_std
            ORDER BY s.sensor_index
        """
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher)]

    def _query_relations(self):
        """
        Récupère les relations filtrées par type sémantique.
        WEAKLY_CORRELATED est exclu au niveau de la requête.
        """
        cypher = """
            MATCH (a:Sensor)-[r:CORRELATED_WITH|ADJACENT_TO]->(b:Sensor)
            WHERE r.rel_type IN ['STRONGLY_CORRELATED', 'CORRELATED']
              AND r.gaussian_weight IS NOT NULL
            RETURN
                a.sensor_index    AS source,
                b.sensor_index    AS target,
                r.gaussian_weight AS gaussian_weight,
                r.rel_type        AS rel_type,
                toFloat(a.missing_rate)  AS src_missing_rate,
                toFloat(a.autocorr_lag1) AS src_autocorr_lag1
        """
        with self._driver.session() as session:
            return [dict(r) for r in session.run(cypher)]

    # ── Construction commune des matrices ─────────────────────

    def _build_from_data(self, nodes, rels):
        """
        Construit W_semantic et q_static depuis les données
        (indépendamment de leur source JSON ou Cypher).

        W_semantic(i, j) = gaussian_weight(i→j)
                           × rel_bonus(rel_type)
                           × (1 - missing_rate(i))
                           × (1 + autocorr_lag1(i)) / 2

        q_static(j) = (1 - missing_rate(j))
                      × (1 + autocorr_lag1(j)) / 2
        """
        K = self.K

        # ── Construire le mapping sensor_id → index ───────────
        # Supporte les deux formats : {'sensor_index': int, ...}
        # et {'source': int, 'target': int, ...}
        if isinstance(nodes[0].get('sensor_index', None), int):
            idx_to_meta = {n['sensor_index']: n for n in nodes}
        else:
            # Mode online : sensor_index peut être retourné comme int
            idx_to_meta = {int(n['sensor_index']): n for n in nodes}

        # Supporte le format offline avec sensor_id string
        sensor_id_to_idx = {}
        if 'sensor_id' in nodes[0]:
            sensor_id_to_idx = {
                str(n['sensor_id']): n['sensor_index'] for n in nodes
            }

        # ── q_static : fiabilité historique par capteur ───────
        q_static = np.zeros(K)
        var_mean  = np.zeros(K)
        var_std   = np.ones(K)

        for meta in idx_to_meta.values():
            j = int(meta['sensor_index'])
            mr  = float(meta.get('missing_rate',   0.0))
            ac  = float(meta.get('autocorr_lag1',  0.5))
            vm  = float(meta.get('temporal_var_mean', 0.0))
            vs  = float(meta.get('temporal_var_std',  1.0))
            # Gérer les valeurs sentinelle -1
            if vm == -1: vm = 0.0
            if vs == -1: vs = 1.0
            q_static[j] = (1.0 - mr) * ((1.0 + ac) / 2.0)
            var_mean[j]  = vm
            var_std[j]   = max(vs, 1e-8)

        # ── W_semantic : matrice structurelle filtrée ─────────
        W = np.zeros((K, K))
        kept = filtered = 0

        for rel in rels:
            rel_type = rel.get('rel_type', 'WEAKLY_CORRELATED')
            bonus    = self.REL_WEIGHTS.get(rel_type, 0.0)

            if bonus == 0.0:
                filtered += 1
                continue

            # Résoudre les indices source/target
            src_raw = rel.get('source', rel.get('src'))
            tgt_raw = rel.get('target', rel.get('tgt'))

            # Supports sensor_id (string) et sensor_index (int)
            if isinstance(src_raw, str) and src_raw in sensor_id_to_idx:
                src = sensor_id_to_idx[src_raw]
                tgt = sensor_id_to_idx[tgt_raw]
            else:
                src = int(src_raw)
                tgt = int(tgt_raw)

            if src >= K or tgt >= K:
                continue

            gw  = float(rel.get('gaussian_weight', 0.0))
            # Attributs de fiabilité de la source (depuis la relation
            # ou depuis idx_to_meta)
            src_meta = idx_to_meta.get(src, {})
            mr  = float(rel.get('src_missing_rate',
                                 src_meta.get('missing_rate', 0.0)))
            ac  = float(rel.get('src_autocorr_lag1',
                                 src_meta.get('autocorr_lag1', 0.5)))

            W[src, tgt] = gw * bonus * (1.0 - mr) * ((1.0 + ac) / 2.0)
            kept += 1

        print(f"  Relations conservées : {kept} | filtrées : {filtered}")

        self.W_semantic = W
        self.q_static   = q_static
        self.var_mean    = var_mean
        self.var_std     = var_std

    # ── API publique ──────────────────────────────────────────

    def get_tensors(self, device):
        """
        Retourne tous les tenseurs nécessaires à Layer 2,
        déplacés sur le device cible.

        Returns:
            W_semantic : torch.Tensor (K, K)
            q_static   : torch.Tensor (K,)
            var_mean   : torch.Tensor (K,)
            var_std    : torch.Tensor (K,)
        """
        import torch
        def to_t(arr):
            return torch.tensor(arr, dtype=torch.float32).to(device)

        return (
            to_t(self.W_semantic),
            to_t(self.q_static),
            to_t(self.var_mean),
            to_t(self.var_std),
        )

    def close(self):
        """Ferme la connexion Neo4j (mode online uniquement)."""
        if hasattr(self, '_driver'):
            self._driver.close()
            print("[Neo4jBridgeV2] Connexion Neo4j fermée.")


# ── Test standalone ───────────────────────────────────────────
if __name__ == '__main__':
    bridge = Neo4jBridgeV2(
        metadata_dir='./neo4j_setup/metadata_metrla',
        mode='offline'
    )
    W, q, vm, vs = bridge.get_tensors('cpu')
    print(f"\nW_semantic shape : {W.shape}")
    print(f"q_static   shape : {q.shape}")
    print(f"W_semantic non-zero : {(W > 0).sum().item()}")
    print(f"q_static   mean     : {q.mean().item():.4f}")
    print("✓ Neo4jBridgeV2 OK")