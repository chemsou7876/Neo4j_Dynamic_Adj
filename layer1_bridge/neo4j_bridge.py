# ============================================================
# neo4j_bridge.py
# Extrait les données de Neo4j et les sauvegarde en .npy
# À exécuter UNE SEULE FOIS en local
# ============================================================

import numpy as np
from neo4j import GraphDatabase
from pathlib import Path
import json

class Neo4jBridge:
    def __init__(self, uri="bolt://127.0.0.1:7687", auth=("neo4j", "chemsou123")):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()
        print("✅ Connecté à Neo4j")

    def extract_all(self, output_dir="./extracted_data"):
        """Extrait tout de Neo4j et sauvegarde en fichiers."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        A_static = self._extract_adj_static()
        var_mean, var_std = self._extract_anomaly_thresholds()

        # Sauvegarder
        np.save(output_dir / "A_static.npy", A_static)
        np.save(output_dir / "var_mean.npy", var_mean)
        np.save(output_dir / "var_std.npy", var_std)

        # Résumé
        print(f"\n{'=' * 50}")
        print(f"DONNÉES EXTRAITES → {output_dir}")
        print(f"{'=' * 50}")
        print(f"  A_static.npy  : {A_static.shape}, {np.count_nonzero(A_static)} non-zero")
        print(f"  var_mean.npy  : {var_mean.shape}")
        print(f"  var_std.npy   : {var_std.shape}")
        print(f"{'=' * 50}")

        return A_static, var_mean, var_std

    def _extract_adj_static(self):
        """Extrait la matrice A_static (36×36) depuis Neo4j."""
        with self.driver.session() as session:
            # D'abord, trouver le nombre de capteurs
            n = session.run("MATCH (s:Sensor) RETURN count(s) AS c").single()['c']
            A = np.zeros((n, n))

            result = session.run("""
                MATCH (a:Sensor)-[r:CORRELATED_WITH]->(b:Sensor)
                WHERE r.gaussian_weight IS NOT NULL
                RETURN a.sensor_index AS src, 
                       b.sensor_index AS tgt, 
                       r.gaussian_weight AS gw
            """)

            count = 0
            for record in result:
                A[record['src'], record['tgt']] = record['gw']
                count += 1

            print(f"✅ A_static extraite: {n}×{n}, {count} entrées")
            return A

    def _extract_anomaly_thresholds(self):
        """Extrait μ_var et σ_var par capteur pour la détection d'anomalie."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor)
                RETURN s.sensor_index AS idx,
                       s.temporal_var_mean AS var_mean,
                       s.temporal_var_std AS var_std
                ORDER BY s.sensor_index
            """)

            var_mean_list = []
            var_std_list = []

            for record in result:
                vm = record['var_mean']
                vs = record['var_std']
                # Gérer les valeurs -1 (None dans les données originales)
                var_mean_list.append(vm if vm != -1 else 0.0)
                var_std_list.append(vs if vs != -1 else 1e10)

            var_mean = np.array(var_mean_list)
            var_std = np.array(var_std_list)

            print(f"✅ Seuils anomalie extraits: {len(var_mean)} capteurs")
            return var_mean, var_std

    def close(self):
        self.driver.close()


# ============================================================
# Exécution directe
# ============================================================
if __name__ == "__main__":
    bridge = Neo4jBridge()
    A_static, var_mean, var_std = bridge.extract_all("./extracted_data")

    # Validation rapide
    print("\n── Validation ──")
    print(f"A_static symétrique ? {np.allclose(A_static, A_static.T)}")
    print(f"A_static diagonale = 0 ? {np.all(np.diag(A_static) == 0)}")
    print(f"Seuils anomalie (capteur 0) : μ={var_mean[0]:.1f}, σ={var_std[0]:.1f}, seuil={var_mean[0] + 3*var_std[0]:.1f}")

    bridge.close()
    print("\n🎉 Extraction terminée !")