# ============================================================
# METRLA_upload.py — Setup Neo4j pour METR-LA
# ============================================================

import numpy as np
import pandas as pd
import json
from neo4j import GraphDatabase
from pathlib import Path
import sys

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_DIR = SCRIPT_DIR / "metadata_metrla"

NODES_PATH = METADATA_DIR / "nodes_metadata.json"
RELATIONS_PATH = METADATA_DIR / "relations_metadata.json"

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "chemsou123")  # adapte le mot de passe

# ============================================================
# CHARGER LES MÉTADONNÉES
# ============================================================
print("=" * 55)
print("SETUP NEO4J — METR-LA (207 sensors)")
print("=" * 55)

with open(NODES_PATH, 'r') as f:
    nodes = json.load(f)

with open(RELATIONS_PATH, 'r') as f:
    relations = json.load(f)

print(f"✅ {len(nodes)} nœuds chargés")
print(f"✅ {len(relations)} relations chargées")

# ============================================================
# CONNEXION NEO4J
# ============================================================
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    driver.verify_connectivity()
    print(f"✅ Connexion Neo4j OK")
except Exception as e:
    print(f"\n❌ Impossible de se connecter à Neo4j: {e}")
    sys.exit(1)

# ============================================================
# NETTOYER LA BASE
# ============================================================
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("✅ Base nettoyée")

# ============================================================
# CRÉER LES 207 NŒUDS SENSOR
# ============================================================
with driver.session() as session:
    for node in nodes:
        clean_node = {}
        for k, v in node.items():
            if v is None:
                clean_node[k] = -1
            else:
                clean_node[k] = v

        session.run("""
            CREATE (s:Sensor {
                sensor_id:         $sensor_id,
                sensor_index:      $sensor_index,
                latitude:          $latitude,
                longitude:         $longitude,
                missing_rate:      $missing_rate,
                missing_category:  $missing_category,
                mean:              $mean,
                std:               $std,
                temporal_var_mean: $temporal_var_mean,
                temporal_var_std:  $temporal_var_std,
                stability:         $stability,
                autocorr_lag1:     $autocorr_lag1,
                smoothness_label:  $smoothness_label
            })
        """, **clean_node)

    count = session.run("MATCH (s:Sensor) RETURN count(s) AS c").single()['c']
    print(f"✅ {count} nœuds Sensor créés")

# ============================================================
# CRÉER LES RELATIONS (avec gaussian_weight INCLUS)
# ============================================================
with driver.session() as session:
    for rel in relations:
        session.run("""
            MATCH (a:Sensor {sensor_id: $source})
            MATCH (b:Sensor {sensor_id: $target})
            CREATE (a)-[:CORRELATED_WITH {
                pearson:          $pearson,
                weight:           $weight,
                distance_km:      $distance_km,
                gaussian_weight:  $gaussian_weight,
                rel_type:         $rel_type
            }]->(b)
        """, **rel)

    count = session.run(
        "MATCH ()-[r:CORRELATED_WITH]->() RETURN count(r) AS c"
    ).single()['c']
    print(f"✅ {count} relations créées (avec gaussian_weight)")

# ============================================================
# AJOUTER LES RELATIONS INVERSES (pour symétrie)
# ============================================================
print("\n⚙️ Ajout des relations inverses...")

# Charger A_static pour trouver les paires manquantes
A_static = np.load(METADATA_DIR / "A_static.npy")

# Index mapping
id_to_idx = {str(n['sensor_id']): n['sensor_index'] for n in nodes}
idx_to_id = {n['sensor_index']: str(n['sensor_id']) for n in nodes}

# Trouver les relations existantes
existing_pairs = set()
for rel in relations:
    existing_pairs.add((rel['source'], rel['target']))

with driver.session() as session:
    created = 0
    N = len(nodes)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            gw = float(A_static[i, j])
            if gw == 0.:
                continue

            src = idx_to_id[i]
            tgt = idx_to_id[j]

            if (src, tgt) in existing_pairs:
                continue

            session.run("""
                MATCH (a:Sensor {sensor_id: $src})
                MATCH (b:Sensor {sensor_id: $tgt})
                CREATE (a)-[:CORRELATED_WITH {
                    gaussian_weight: $gw,
                    pearson: 0.0,
                    weight: $gw,
                    distance_km: -1.0,
                    rel_type: 'SPATIAL_ONLY'
                }]->(b)
            """, src=src, tgt=tgt, gw=gw)
            created += 1

    print(f"✅ {created} relations inverses ajoutées")

# ============================================================
# VÉRIFICATION FINALE
# ============================================================
with driver.session() as session:
    n_nodes = session.run("MATCH (s:Sensor) RETURN count(s) AS c").single()['c']
    n_rels = session.run("MATCH ()-[r:CORRELATED_WITH]->() RETURN count(r) AS c").single()['c']
    n_gw = session.run("""
        MATCH ()-[r:CORRELATED_WITH]->()
        WHERE r.gaussian_weight IS NOT NULL
        RETURN count(r) AS c
    """).single()['c']
    n_var = session.run("""
        MATCH (s:Sensor)
        WHERE s.temporal_var_mean IS NOT NULL AND s.temporal_var_mean <> -1
        RETURN count(s) AS c
    """).single()['c']

print(f"\n{'=' * 55}")
print(f"VÉRIFICATION FINALE — METR-LA")
print(f"{'=' * 55}")
print(f"  Nœuds Sensor          : {n_nodes}")
print(f"  Relations totales     : {n_rels}")
print(f"  Avec gaussian_weight  : {n_gw}")
print(f"  Avec seuils anomalie  : {n_var}/207")
print(f"{'=' * 55}")

if n_nodes == 207 and n_gw > 0 and n_var == 207:
    print("✅ NEO4J PRÊT — METR-LA Layer 1 complète")
else:
    print("⚠️  Vérifier les données manquantes")

driver.close()
print("\n🎉 Setup METR-LA terminé !")