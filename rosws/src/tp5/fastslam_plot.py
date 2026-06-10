#!/usr/bin/env python3
"""TP5 - Parte 2: gráfico de resultados del FastSLAM (sin Gazebo).

Corre el FastSLAM contra un mundo sintético (mismo motor que el test offline)
y grafica:
  * trayectoria real del robot vs. trayectoria estimada (mejor partícula)
  * landmarks reales vs. landmarks estimados con sus elipses de covarianza

Genera fastslam_result.pdf, apto como gráfico de entrega.

Uso:  python3 fastslam_plot.py
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tp5.my_fastslam import FastSlamNode, normalize_angle  # noqa: E402

from custom_msgs.msg import DeltaOdom  # noqa: E402
from geometry_msgs.msg import Pose, PoseArray  # noqa: E402


# Las covarianzas convergen a escala de mm, así que se amplifican para poder
# visualizar su forma y orientación (igual que el código de markers de la
# consigna, que multiplica por 30).
COV_MAGNIF = 30.0


def cov_ellipse(ax, mu, sigma, n_std=3.0, **kwargs):
    """Dibuja la elipse de covarianza (n_std sigmas, amplificada) de un landmark."""
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    width, height = COV_MAGNIF * 2 * n_std * np.sqrt(np.maximum(vals, 0.0))
    e = Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(e)


def main():
    np.random.seed(42)
    rclpy.init()
    node = FastSlamNode()

    true_landmarks = np.array([
        [2.0, 1.0],
        [3.0, -1.5],
        [1.0, 2.0],
        [4.0, 0.5],
    ])

    x, y, theta = 0.0, 0.0, 0.0
    sigma_meas = 0.05
    true_path = [(x, y)]
    est_path = []

    for _ in range(60):
        dt_true, dr1_true, dr2_true = 0.1, 0.05, 0.05
        x += dt_true * math.cos(theta + dr1_true)
        y += dt_true * math.sin(theta + dr1_true)
        theta = normalize_angle(theta + dr1_true + dr2_true)
        true_path.append((x, y))

        d = DeltaOdom()
        d.dr1, d.dt, d.dr2 = dr1_true, dt_true, dr2_true
        node.delta_callback(d)

        pa = PoseArray()
        for lm in true_landmarks:
            dx, dy = lm[0] - x, lm[1] - y
            r = math.hypot(dx, dy)
            phi = normalize_angle(math.atan2(dy, dx) - theta)
            p = Pose()
            if r < 4.0:
                p.position.x = r + np.random.normal(0, sigma_meas)
                p.position.z = phi + np.random.normal(0, sigma_meas)
            pa.poses.append(p)
        node.obs_callback(pa)

        best = node._best_particle()
        est_path.append((best.x, best.y))

    best = node._best_particle()
    true_path = np.array(true_path)
    est_path = np.array(est_path)

    # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(true_path[:, 0], true_path[:, 1], "g-", lw=2,
            label="trayectoria real")
    ax.plot(est_path[:, 0], est_path[:, 1], "r--", lw=2,
            label="trayectoria estimada (FastSLAM)")

    ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], marker="*",
               s=220, c="green", edgecolors="k", zorder=5,
               label="landmarks reales")

    first = True
    for lm_id, (mu, sigma) in sorted(best.landmarks.items()):
        ax.scatter(mu[0], mu[1], marker="o", s=60, c="red", zorder=6,
                   label="landmarks estimados" if first else None)
        cov_ellipse(ax, mu, sigma, n_std=3.0, facecolor="blue",
                    alpha=0.25, edgecolor="blue",
                    label=f"covarianza 3σ (×{COV_MAGNIF:.0f})" if first else None)
        ax.annotate(f"{lm_id}", (mu[0], mu[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
        first = False

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("TP5 - FastSLAM: trayectoria y mapa de landmarks estimados")
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fastslam_result.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Figura guardada en: {out_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
