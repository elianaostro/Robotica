"""
Experimento del ejercicio 3.4: Weighted A*.
Corre A* con f(n) = g(n) + w·h(n) para w ∈ {1, 2, 5, 10} sobre el mapa duplicado y
reporta celdas expandidas y longitud real del camino, para mostrar empíricamente el
trade-off velocidad (menos nodos) vs. optimalidad (camino más largo).
Backend Agg (sin ventanas). Respalda numéricamente el análisis del INFORME.
"""
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # no bloquear

import planning_framework as pf

occ = np.loadtxt("map.txt")
occ = np.kron(occ, np.ones((2, 2)))
start = np.array([44, 66])
goal = np.array([80, 30])

rows = []
for w in [1, 2, 5, 10]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        expanded, cost, length, _ = pf.run_astar(occ, start, goal, w=w, plot=False)
    rows.append((w, expanded, cost, length))

opt_len = rows[0][3]  # longitud con w=1 (óptima sobre la grilla) como referencia

print()
print(f"{'w':>3} | {'celdas expandidas':>18} | {'costo g':>10} | {'longitud':>9} | {'exceso vs óptimo':>16}")
print("-" * 72)
for w, expanded, cost, length in rows:
    excess = (length / opt_len - 1.0) * 100.0
    print(f"{w:>3} | {expanded:>18} | {cost:>10.2f} | {length:>9.2f} | {excess:>15.1f}%")
