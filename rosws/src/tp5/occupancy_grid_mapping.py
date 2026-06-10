#!/usr/bin/env python3
"""TP5 - Parte 1: Mapeo de grilla de ocupación 1D con poses conocidas.

El robot está en c0 = 0 mirando hacia cn. Construye una grilla de ocupación
1D (celdas de 0 a 200 cm, paso 10 cm) integrando una secuencia de mediciones
de un sensor de distancia con el modelo inverso del sensor, en log-odds.

Modelo inverso del sensor (z = medición, d = distancia de la celda al robot):
  * d < z             -> p = 0.3  (celda libre, delante del obstáculo)
  * z <= d <= z + 20  -> p = 0.6  (celda ocupada, sobre el obstáculo)
  * d > z + 20        -> sin cambio (prob. a priori)

Belief a priori = 0.5  =>  l0 = log(0.5 / 0.5) = 0

Uso:
    python3 occupancy_grid_mapping.py
Genera occupancy_grid_1d.png con el belief final.
"""

import os

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla, para guardar la figura
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración ---
CELL_RES = 10              # resolución de la grilla [cm]
MAP_LEN = 200              # longitud del mapa [cm]
P_FREE = 0.3               # prob. de ocupación delante de la medición
P_OCC = 0.6                # prob. de ocupación sobre la medición
OCC_BAND = 20              # banda de ocupación detrás de la medición [cm]
PRIOR = 0.5                # belief a priori

MEASUREMENTS = [101, 82, 91, 112, 99, 151, 96, 85, 99, 105]


def prob_to_logodds(p):
    return np.log(p / (1.0 - p))


def logodds_to_prob(l):
    return 1.0 - 1.0 / (1.0 + np.exp(l))


def inverse_sensor_model(d, z):
    """Probabilidad de ocupación que el modelo inverso asigna a una celda a
    distancia d dada una medición z."""
    if d < z:
        return P_FREE
    elif d <= z + OCC_BAND:
        return P_OCC
    else:
        return None  # sin cambio


def main():
    # Coordenadas de las celdas: 0, 10, ..., 200 (ambos incluidos)
    c = np.arange(0, MAP_LEN + 1, CELL_RES)

    l0 = prob_to_logodds(PRIOR)          # = 0
    log_odds = np.full(c.shape, l0)      # belief inicial en log-odds

    # Integración de las mediciones (el robot está fijo en c0 = 0, así que la
    # distancia de cada celda al robot es su propia coordenada).
    for z in MEASUREMENTS:
        for i, d in enumerate(c):
            p = inverse_sensor_model(d, z)
            if p is None:
                continue  # celda más allá de z+20cm: no cambia
            log_odds[i] += prob_to_logodds(p) - l0

    m = logodds_to_prob(log_odds)        # belief final como probabilidad

    # --- Resultados por consola ---
    print("celda (cm) | P(ocupada)")
    for ci, mi in zip(c, m):
        print(f"   {ci:3d}     |   {mi:.3f}")

    # --- Gráfico ---
    plt.figure(figsize=(9, 5))
    plt.plot(c, m, marker="o")
    plt.axhline(PRIOR, color="gray", linestyle="--", label="a priori (0.5)")
    plt.xlabel("coordenada de la celda [cm]")
    plt.ylabel("P(ocupada)")
    plt.title("TP5 - Mapeo de grilla de ocupación 1D")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "occupancy_grid_1d.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nFigura guardada en: {out_path}")


if __name__ == "__main__":
    main()
