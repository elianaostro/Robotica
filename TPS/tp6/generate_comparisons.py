"""
Genera las figuras COMPARATIVAS del TP6:

  1) astar_vs_thetastar.png  -> ejercicio 4.3 (comparación VISUAL A* vs Theta* sobre el
     mismo mapa, con la longitud real de cada camino en el título).
  2) weighted_astar.png      -> ejercicio 3.4 (trade-off de Weighted A*: celdas expandidas
     y longitud del camino en función del peso w).

Estrategia: primero se corren los planificadores con el dibujo interno neutralizado
(stubs no-op) para quedarnos solo con los datos que DEVUELVEN (camino, celdas, longitud);
luego se restauran las funciones reales de matplotlib y se arman figuras limpias.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Guardar las funciones reales de matplotlib antes de neutralizarlas ---
_real = {name: getattr(plt, name) for name in
         ("clf", "imshow", "plot", "pause", "show", "draw", "waitforbuttonpress",
          "axis", "xlabel", "ylabel", "title", "legend", "savefig")}


def _silence():
    for name in ("clf", "imshow", "plot", "pause", "show", "draw",
                 "waitforbuttonpress", "axis", "xlabel", "ylabel"):
        setattr(plt, name, lambda *a, **k: None)


def _restore():
    for name, fn in _real.items():
        setattr(plt, name, fn)


import planning_framework as pf

occ = np.loadtxt("map.txt")
occ = np.kron(occ, np.ones((2, 2)))
start = np.array([44, 66])
goal = np.array([80, 30])

# --- 1) Correr A* (w=1) y Theta* en silencio, quedándonos con los caminos ---
_silence()
a_expanded, a_cost, a_len, a_path = pf.run_astar(occ, start, goal, w=1.0, plot=False)
t_expanded, t_cost, t_len, t_path = pf.run_thetastar(occ, start, goal)
# Weighted A*: barrido de w
weights = [1, 2, 5, 10]
w_expanded, w_length = [], []
for w in weights:
    e, c, l, _ = pf.run_astar(occ, start, goal, w=w, plot=False)
    w_expanded.append(e)
    w_length.append(l)
_restore()


def _draw_map_and_path(ax, path, title, color):
    ax.imshow(occ.T, cmap=plt.cm.gray, interpolation="none", origin="upper")
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color + "-", linewidth=2.2, zorder=5, label="camino")
    ax.plot([start[0]], [start[1]], "ro", label="inicio")
    ax.plot([goal[0]], [goal[1]], "go", label="meta")
    ax.set_xlim(0, occ.shape[0] - 1)
    ax.set_ylim(0, occ.shape[1] - 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


# === Figura 1: A* vs Theta* lado a lado (ejercicio 4.3) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
_draw_map_and_path(ax1, a_path, f"A* (45°)\nlongitud = {a_len:.2f} | celdas exp. = {a_expanded}", "b")
_draw_map_and_path(ax2, t_path, f"Theta* (any-angle)\nlongitud = {t_len:.2f} | celdas exp. = {t_expanded}", "c")
mejora = (1 - t_len / a_len) * 100
fig.suptitle(f"Comparación A* vs Theta* — Theta* es {mejora:.1f}% más corto sobre la misma grilla",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("astar_vs_thetastar.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"guardado astar_vs_thetastar.png  (A*={a_len:.2f}, Theta*={t_len:.2f}, mejora={mejora:.1f}%)")

# === Figura 2: Weighted A* — trade-off velocidad vs optimalidad (ejercicio 3.4) ===
fig, ax_left = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(weights))

color_e = "tab:blue"
ax_left.plot(x, w_expanded, "o-", color=color_e, linewidth=2, markersize=7, label="celdas expandidas")
ax_left.set_xlabel("peso de la heurística  w   (f = g + w·h)")
ax_left.set_ylabel("celdas expandidas", color=color_e)
ax_left.tick_params(axis="y", labelcolor=color_e)
ax_left.set_xticks(x)
ax_left.set_xticklabels([str(w) for w in weights])
for xi, e in zip(x, w_expanded):
    ax_left.annotate(str(e), (xi, e), textcoords="offset points", xytext=(0, 8),
                     ha="center", color=color_e, fontsize=9)

color_l = "tab:red"
ax_right = ax_left.twinx()
ax_right.plot(x, w_length, "s--", color=color_l, linewidth=2, markersize=7, label="longitud del camino")
ax_right.set_ylabel("longitud real del camino", color=color_l)
ax_right.tick_params(axis="y", labelcolor=color_l)
for xi, l in zip(x, w_length):
    ax_right.annotate(f"{l:.1f}", (xi, l), textcoords="offset points", xytext=(0, -14),
                      ha="center", color=color_l, fontsize=9)

ax_left.set_title("Weighted A*: más voraz (w↑) → menos celdas expandidas, camino más largo")
fig.tight_layout()
fig.savefig("weighted_astar.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"guardado weighted_astar.png  (expandidas={w_expanded}, longitudes={[round(l,2) for l in w_length]})")

print("Figuras comparativas generadas.")
