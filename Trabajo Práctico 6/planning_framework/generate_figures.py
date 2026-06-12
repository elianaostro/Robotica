"""
Genera las figuras de los caminos finales para cada algoritmo y las guarda como PNG.
Usa backend Agg (sin ventanas). Reutiliza planning_framework.py tal cual: solo
neutraliza las pausas interactivas y redirige plt.show()/waitforbuttonpress() a savefig.
"""
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Acelerar: las pausas interactivas no aportan al PNG final
plt.pause = lambda *a, **k: None

import planning_framework as pf

occ = np.loadtxt("map.txt")
occ = np.kron(occ, np.ones((2, 2)))
start = np.array([44, 66])
goal = np.array([80, 30])

_real_show = plt.show
_real_wait = plt.waitforbuttonpress


def save_as(target):
    """Redirige show()/waitforbuttonpress() para que guarden la figura actual."""
    def _save(*a, **k):
        plt.title(target)
        plt.legend(loc="upper right", fontsize=8)
        plt.savefig(target, dpi=130, bbox_inches="tight")
    plt.show = _save
    plt.waitforbuttonpress = _save


runs = [
    ("dijkstra.png", lambda: pf.run_dijkstra(occ, start, goal)),
    ("astar.png",    lambda: pf.run_astar(occ, start, goal)),
    ("thetastar.png",lambda: pf.run_thetastar(occ, start, goal)),
    ("apf.png",      lambda: pf.run_potential_fields(occ, start, goal)),
    ("rrt.png",      lambda: (random.seed(0), pf.run_rrt(occ, start, goal, step_size=3.0))[1]),
    ("rrtstar.png",  lambda: (random.seed(0), pf.run_rrt_star(occ, start, goal,
                              max_nodes=4000, step_size=3.0, search_radius=6.0))[1]),
]

for fname, fn in runs:
    plt.figure(figsize=(6, 6))
    save_as(fname)
    try:
        fn()
    except Exception as e:
        print(f"[WARN] {fname}: {e}")
    plt.close("all")
    print(f"guardado {fname}")

print("Figuras generadas.")
