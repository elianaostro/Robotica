#!/usr/bin/env python3
"""Test offline de la matemática del FastSLAM (sin Gazebo).

Simula un robot moviéndose en un mapa con landmarks conocidos (solo para
generar las mediciones; el filtro NO conoce sus posiciones). Alimenta el nodo
con /delta y /observed_landmarks sintéticos y verifica que la pose estimada y
el mapa de landmarks converjan a la verdad.

Uso:  python3 test_fastslam_offline.py
"""

import math
import sys

import numpy as np
import rclpy

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tp5.my_fastslam import FastSlamNode, normalize_angle  # noqa: E402

from custom_msgs.msg import DeltaOdom  # noqa: E402
from geometry_msgs.msg import Pose, PoseArray  # noqa: E402


def main():
    np.random.seed(42)
    rclpy.init()
    node = FastSlamNode()
    node.num_particles = len(node.particles)

    # --- Mundo de verdad (desconocido para el filtro) ---
    true_landmarks = np.array([
        [2.0, 1.0],
        [3.0, -1.5],
        [1.0, 2.0],
        [4.0, 0.5],
    ])

    # Trayectoria de verdad: el robot avanza describiendo una curva suave
    x, y, theta = 0.0, 0.0, 0.0
    sigma_meas = 0.05

    for step in range(60):
        # Comando de movimiento de verdad
        dt_true = 0.1
        dr1_true = 0.05
        dr2_true = 0.05

        # Avanza la pose de verdad
        x += dt_true * math.cos(theta + dr1_true)
        y += dt_true * math.sin(theta + dr1_true)
        theta = normalize_angle(theta + dr1_true + dr2_true)

        # --- /delta (odometría perfecta para este test) ---
        d = DeltaOdom()
        d.dr1, d.dt, d.dr2 = dr1_true, dt_true, dr2_true
        node.delta_callback(d)

        # --- /observed_landmarks (rango y ángulo con ruido) ---
        pa = PoseArray()
        for lm in true_landmarks:
            dx = lm[0] - x
            dy = lm[1] - y
            r = math.hypot(dx, dy)
            phi = normalize_angle(math.atan2(dy, dx) - theta)
            p = Pose()
            if r < 4.0:  # solo landmarks "visibles"
                p.position.x = r + np.random.normal(0, sigma_meas)
                p.position.z = phi + np.random.normal(0, sigma_meas)
            else:
                p.position.x = 0.0
                p.position.z = 0.0
            pa.poses.append(p)
        node.obs_callback(pa)

    best = node._best_particle()

    print("=== Pose del robot ===")
    print(f"  verdad : x={x:.3f} y={y:.3f} theta={theta:.3f}")
    print(f"  estim. : x={best.x:.3f} y={best.y:.3f} theta={best.theta:.3f}")
    pose_err = math.hypot(best.x - x, best.y - y)
    print(f"  error posición: {pose_err:.3f} m")

    print("\n=== Landmarks ===")
    max_lm_err = 0.0
    for lm_id, (mu, sigma) in sorted(best.landmarks.items()):
        truth = true_landmarks[lm_id]
        err = math.hypot(mu[0] - truth[0], mu[1] - truth[1])
        max_lm_err = max(max_lm_err, err)
        print(f"  LM {lm_id}: estim=({mu[0]:.2f},{mu[1]:.2f}) "
              f"verdad=({truth[0]:.2f},{truth[1]:.2f}) err={err:.3f}")

    node.destroy_node()
    rclpy.shutdown()

    print("\n=== Veredicto ===")
    ok = pose_err < 0.30 and max_lm_err < 0.30 and len(best.landmarks) == 4
    print(f"  error pose={pose_err:.3f}  max error landmark={max_lm_err:.3f}  "
          f"landmarks={len(best.landmarks)}/4")
    print("  RESULTADO:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
