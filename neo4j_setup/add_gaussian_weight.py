# ============================================================
# ÉTAPE 1.1 : Calculer A_static (identique à PriSTI) et
#             mettre à jour Neo4j avec gaussian_weight
# ============================================================

import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import haversine_distances
from neo4j import GraphDatabase
from pathlib import Path
import sys

# ── Reproduire EXACTEMENT la fonction de PriSTI ──

def geographical_distance(x, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)
    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM
    return distances

def thresholded_gaussian_kernel(x, theta=None, threshold=None):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = weights < threshold
        weights[mask] = 0.
    return weights

def get_adj_AQI36_from_latlng(latlng_path):
    df = pd.read_csv(latlng_path)
    df_coords = df[['latitude', 'longitude']]
    dist = geographical_distance(df_coords, to_rad=False)
    theta = np.std(dist[:36, :36])
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=0.1)
    np.fill_diagonal(adj, 0.)
    return adj, df['sensor_id'].astype(str).tolist()

# ── Calculer A_static ──
# LATLNG_PATH = "../PriSTI_modified/data/pm25/SampleData/pm25_latlng.txt"  # adapte le chemin
# Resolve the lat/lng file path robustly relative to this script, with a CWD fallback.
SCRIPT_DIR = Path(__file__).resolve().parent
LATLNG_PATH = SCRIPT_DIR.parent / "PriSTI_modified" / "data" / "pm25" / "SampleData" / "pm25_latlng.txt"
if not LATLNG_PATH.exists():
    alt = Path.cwd() / "PriSTI_modified" / "data" / "pm25" / "SampleData" / "pm25_latlng.txt"
    if alt.exists():
        LATLNG_PATH = alt
    else:
        raise FileNotFoundError(f"LatLng file not found. Tried: {LATLNG_PATH} and {alt}")

A_static, sensor_ids = get_adj_AQI36_from_latlng(str(LATLNG_PATH))

print(f"A_static shape: {A_static.shape}")
print(f"Non-zero entries: {np.count_nonzero(A_static)}")
print(f"Value range: [{A_static[A_static > 0].min():.4f}, {A_static.max():.4f}]")

# ── Connexion Neo4j ──
URI = "bolt://127.0.0.1:7687"  # prefer IPv4 to avoid ::1 resolution issues on Windows
AUTH = ("neo4j", "chemsou123")
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    # quick connectivity check to fail fast with a clear message
    driver.verify_connectivity()
except Exception as e:
    print("\nERROR: Unable to connect to Neo4j at", URI)
    print("Reason:", str(e))
    print("Make sure Neo4j is running and listening on port 7687, and update AUTH if needed.")
    sys.exit(1)

# ── Mettre à jour les relations existantes avec gaussian_weight ──
# ── ET ajouter les relations manquantes (celles où pearson < seuil
#    mais gaussian_weight > 0) ──

with driver.session() as session:
    updated = 0
    created = 0
    
    for i in range(36):
        for j in range(36):
            if i == j:
                continue
            gw = float(A_static[i, j])
            if gw == 0.:
                continue
            
            src = sensor_ids[i]
            tgt = sensor_ids[j]
            
            # Vérifier si la relation existe déjà
            result = session.run("""
                MATCH (a:Sensor {sensor_id: $src})-[r:CORRELATED_WITH]->(b:Sensor {sensor_id: $tgt})
                RETURN r
            """, src=src, tgt=tgt)
            
            if result.peek():
                # Mettre à jour avec gaussian_weight
                session.run("""
                    MATCH (a:Sensor {sensor_id: $src})-[r:CORRELATED_WITH]->(b:Sensor {sensor_id: $tgt})
                    SET r.gaussian_weight = $gw
                """, src=src, tgt=tgt, gw=gw)
                updated += 1
            else:
                # Créer la relation si elle n'existe pas
                session.run("""
                    MATCH (a:Sensor {sensor_id: $src})
                    MATCH (b:Sensor {sensor_id: $tgt})
                    CREATE (a)-[:CORRELATED_WITH {
                        gaussian_weight: $gw,
                        pearson: 0.0,
                        weight: $gw,
                        distance_km: -1,
                        rel_type: 'SPATIAL_ONLY'
                    }]->(b)
                """, src=src, tgt=tgt, gw=gw)
                created += 1
    
    print(f"\n✅ Relations mises à jour: {updated}")
    print(f"✅ Relations créées: {created}")

driver.close()