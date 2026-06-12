"""
Harness de verificacion headless (sin ventanas graficas) para planning_framework.py.
Stubea matplotlib (Agg + no-ops) y prueba cada funcion implementada:
 - tests unitarios de las funciones puras
 - corrida real de Dijkstra / A* / Theta* / RRT / RRT* sobre el mapa duplicado
No forma parte de la entrega; es solo para validar las implementaciones.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Neutralizar todo lo interactivo/grafico para que las corridas no bloqueen ni dibujen
plt.pause = lambda *a, **k: None
plt.show = lambda *a, **k: None
plt.draw = lambda *a, **k: None
plt.waitforbuttonpress = lambda *a, **k: None
plt.clf = lambda *a, **k: None
plt.plot = lambda *a, **k: None
plt.imshow = lambda *a, **k: None
plt.pause = lambda *a, **k: None

import planning_framework as pf

SQRT2 = np.sqrt(2)


def test_neighborhood():
    shape = (10, 10)
    # interior -> 8 vecinos
    assert len(pf.get_neighborhood([5, 5], shape)) == 8
    # esquina -> 3 vecinos
    assert len(pf.get_neighborhood([0, 0], shape)) == 3
    # borde -> 5 vecinos
    assert len(pf.get_neighborhood([0, 5], shape)) == 5
    print("OK get_neighborhood")


def test_heuristic():
    assert abs(pf.get_heuristic([0, 0], [3, 4]) - 5.0) < 1e-9
    assert pf.get_heuristic([2, 2], [2, 2]) == 0.0
    print("OK get_heuristic (euclidea)")


def test_edge_cost():
    occ = np.zeros((5, 5))
    occ[2, 2] = 1.0          # obstaculo
    occ[3, 3] = 0.2          # libre pero algo ocupado
    # arco a obstaculo -> inf
    assert pf.get_edge_cost([1, 2], [2, 2], occ) == np.inf
    # arco recto en celda libre -> 1.0
    assert abs(pf.get_edge_cost([0, 0], [0, 1], occ) - 1.0) < 1e-9
    # arco diagonal -> sqrt(2)
    assert abs(pf.get_edge_cost([0, 0], [1, 1], occ) - SQRT2) < 1e-9
    # celda con ocupacion intermedia cuesta mas que una libre equivalente
    c_occ = pf.get_edge_cost([2, 3], [3, 3], occ)
    c_free = pf.get_edge_cost([2, 0], [3, 0], occ)
    assert c_occ > c_free
    print("OK get_edge_cost (umbral 0.4 + penalizacion ocupacion)")


def test_forces():
    # Atractiva: apunta de la celda hacia la meta
    f = pf.get_attractive_force([0, 0], [10, 0])
    assert f[0] > 0 and abs(f[1]) < 1e-9
    # Repulsiva: obstaculo a la derecha empuja hacia la izquierda
    occ = np.zeros((20, 20))
    occ[12, 10] = 1.0
    fr = pf.get_repulsive_force(np.array([10.0, 10.0]), occ, d_0=5.0)
    assert fr[0] < 0, fr            # empuja en -x (alejandose del obstaculo)
    # Fuera de la distancia de influencia -> sin fuerza
    fr_far = pf.get_repulsive_force(np.array([0.0, 0.0]), occ, d_0=3.0)
    assert np.linalg.norm(fr_far) < 1e-9
    print("OK fuerzas atractiva/repulsiva")


import io
import contextlib


def _run_capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def test_searches():
    occ = np.loadtxt("map.txt")
    occ = np.kron(occ, np.ones((2, 2)))
    start = np.array([44, 66])
    goal = np.array([80, 30])

    for name, fn in [("Dijkstra", pf.run_dijkstra),
                     ("A*", pf.run_astar),
                     ("Theta*", pf.run_thetastar)]:
        out = _run_capture(fn, occ, start, goal)
        assert "No se encontró" not in out, f"{name} no encontro camino:\n{out}"
        assert "Longitud real" in out, f"{name} no reconstruyo camino:\n{out}"
        # extraer longitud reportada
        for line in out.splitlines():
            if "Longitud real" in line:
                print(f"OK {name}: {line.strip()}")

    # RRT / RRT* (acotamos nodos para que sea rapido)
    import random
    random.seed(0)
    out = _run_capture(pf.run_rrt, occ, start, goal, max_nodes=20000, step_size=3.0)
    assert "No se encontró" not in out, f"RRT no encontro camino:\n{out}"
    print("OK RRT: camino encontrado")
    random.seed(0)
    out = _run_capture(pf.run_rrt_star, occ, start, goal,
                       max_nodes=2000, step_size=3.0, search_radius=6.0)
    assert "No se encontró" not in out, f"RRT* no encontro camino:\n{out}"
    for line in out.splitlines():
        if "Longitud real" in line:
            print(f"OK RRT*: {line.strip()}")


if __name__ == "__main__":
    test_neighborhood()
    test_heuristic()
    test_edge_cost()
    test_forces()
    test_searches()
    print("\nTODOS LOS TESTS PASARON")
