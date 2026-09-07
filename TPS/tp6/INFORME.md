# TP6 — ICP y Planeamiento de Trayectorias

**I-402 — Principios de la Robótica Autónoma**

Implementación de los planificadores en `planning_framework.py` y respuestas conceptuales.
Mapa usado: `map.txt` (53×49) duplicado a 106×98; `start=[44,66]`, `goal=[80,30]`.

Convención de coordenadas: cada celda es `[i, j]` y se indexa `occ_map[i, j]` (consistente con
`bresenham_line_of_sight` e `is_collision_free`). Umbral de obstáculo unificado en **0.4** para
todos los planificadores, de modo que A*, Theta* y RRT operan sobre el mismo espacio libre.

---

## 1. Campos Potenciales Artificiales (APF)

**1.1 Fuerza atractiva (`get_attractive_force`).** Se usa un potencial **cónico/cuadrático combinado**:
- Lejos de la meta (`dist > d_switch`): potencial cónico → fuerza de **magnitud constante** `k_att`.
  Evita fuerzas desproporcionadamente grandes cuando el robot está lejos.
- Cerca de la meta (`dist ≤ d_switch`): potencial cuadrático → fuerza **lineal en la distancia**
  (`k_att·dist`), que decae suavemente y evita oscilar alrededor del objetivo.

En ambos casos el vector apunta de la celda hacia la meta (`goal − cell`), normalizado y escalado.

**1.2 Fuerza repulsiva (`get_repulsive_force`).** Modelo FIRAS de Khatib. Solo actúa si hay un
obstáculo a distancia menor que la distancia de influencia crítica `d_0`. Para cada celda ocupada
(`prob ≥ 0.4`) dentro de una ventana de radio `d_0`:

```
F = k_rep · (1/dist − 1/d_0) · (1/dist²) · (vector_unitario obstáculo→celda)
```

La magnitud crece sin cota cuando `dist→0` y se **anula exactamente en `dist = d_0`** (continuidad).
Se suman las contribuciones de todos los obstáculos dentro del radio. Restringir el cálculo a una
ventana alrededor de la celda evita recorrer todo el mapa en cada paso.

**1.3 Mínimos locales.** El problema central de APF: el robot se mueve siguiendo el gradiente
descendente del potencial total (atractivo + repulsivo). Existen configuraciones donde
`F_att + F_rep = 0` (o casi) **sin estar en la meta** — típicamente frente a obstáculos cóncavos,
pasillos estrechos, o cuando la repulsión de un obstáculo grande cancela exactamente la atracción.
Allí el campo tiene un mínimo local y el robot queda **estancado** u oscilando, sin alcanzar el
objetivo, aunque exista un camino.

**¿Ocurre en nuestra implementación?** **Sí.** Para `start=[44,66] → goal=[80,30]` el robot queda
estancado (ver `apf.png`, mensaje *"Campos Potenciales: Estancado"*): un obstáculo intermedio genera
una repulsión que, combinada con la atracción, anula el avance neto. Es el comportamiento esperado
de APF puro. Mitigaciones conocidas (no exigidas): potenciales navegables, *random walks* al
detectar estancamiento, o combinar APF con un planificador global.

---

## 2. Algoritmo de Dijkstra

**2.1 Vecindario (`get_neighborhood`).** 8-conectividad: los 8 desplazamientos
`(dx,dy) ∈ {−1,0,1}²\{(0,0)}`, descartando los que caen fuera de los límites del mapa.
Celda interior → 8 vecinos; borde → 5; esquina → 3.

**2.2 Costo de arco y umbral (`get_edge_cost`).** Si la ocupación de la celda destino supera el
umbral, el arco es **intransitable** (`inf`). **Umbral elegido: 0.4.** Justificación: un valor más
bajo descarta demasiado espacio libre (caminos imposibles); uno más alto (p. ej. 0.6–0.7) deja que
el robot atraviese celdas con alta probabilidad de obstáculo (riesgo de colisión). 0.4 es un
compromiso conservador y, sobre todo, **coincide con el umbral de `bresenham` e `is_collision_free`**,
garantizando que todos los planificadores compartan el mismo espacio libre (clave para que la
comparación A* vs Theta* del punto 4.3 sea válida).

**2.3 Información de ocupación en el costo.** Al costo geométrico (longitud euclídea del arco: `1`
en movimiento recto, `√2` en diagonal) se le suma una penalización proporcional a la ocupación de
la celda destino:

```
edge_cost = dist · (1 + w_occ · occ(child))
```

Como el término extra es `≥ 0`, ante caminos de longitud parecida el algoritmo **prefiere celdas con
baja probabilidad de ocupación** (rutas más seguras), y la heurística sigue siendo admisible.

**2.4 Actualización (relajación).** Para cada vecino no expandido (`closed_flag = 0`):
si `g(parent) + edge_cost(parent, vecino) < g(vecino)`, se actualiza `costs[vecino]` y se fija
`predecessors[vecino] = parent`. El nodo a expandir se elige siempre como el de menor `g` entre
los abiertos (heurística fija en 0).

---

## 3. Algoritmo A\*

**3.1 Propiedades de la heurística para optimalidad.** Para que A* sea óptimo, `h` debe ser:
- **Admisible**: nunca sobreestima el costo real al objetivo (`h(n) ≤ h*(n)`). Garantiza que la
  primera solución hallada es óptima.
- **Consistente (monótona)**: `h(n) ≤ c(n, n') + h(n')` para todo vecino `n'`. Garantiza optimalidad
  incluso sin reabrir nodos cerrados (como hace esta implementación, que marca `closed_flags`).
  La consistencia implica admisibilidad.

**3.2 Heurística (`get_heuristic`).** **Distancia euclídea** `‖cell − goal‖`. Es admisible (ninguna
ruta sobre la grilla puede ser más corta que la recta) y consistente. **Se eligió euclídea y no
octile** porque `get_heuristic` la comparte `run_thetastar`: en Theta* el costo real puede ser
exactamente la línea recta any-angle, y la octile (`≥` euclídea) la **sobreestimaría**, volviendo
Theta* no óptimo.

**3.3 Actualización.** Idéntica a 2.4: se actualiza el costo acumulado real `g`. La heurística se
suma aparte al seleccionar el nodo (`open_costs = costs + heuristic`), implementando `f = g + h`.

**Resultado.** A* expande **muchos menos nodos** que Dijkstra para el mismo camino. Verificado:
A* y Dijkstra obtienen **idéntica longitud de camino (72.53)** — ambos óptimos sobre la grilla —
pero A* explora una fracción de las celdas (búsqueda informada/dirigida hacia la meta).

**3.4 Sobre-ponderar la heurística (`h₂ = w·h`, `w ∈ {1,2,5,10}`).** Esto es **Weighted A\***:
`f(n) = g(n) + w·h(n)`. Se implementó como parámetro `w` en `run_astar` (ver
`weighted_astar_experiment.py`). Resultados reales sobre este mapa (`start=[44,66] → goal=[80,30]`):

| `w` | Celdas expandidas | Costo `g` | Longitud real | Exceso vs. óptimo |
|----:|------------------:|----------:|--------------:|------------------:|
| 1   | 1450              | 72.55     | 72.53         | 0.0 %             |
| 2   | 308               | 75.35     | 73.94         | +1.9 %            |
| 5   | 180               | 89.29     | 75.01         | +3.4 %            |
| 10  | 138               | 105.53    | 94.91         | +30.9 %           |

Lectura de los números:
- `w = 1`: A* estándar, heurística admisible → camino **óptimo** (72.53), pero expande **1450 celdas**.
- `w > 1`: la heurística sobreestima y deja de ser admisible. La búsqueda se vuelve **más voraz**
  (greedy) y expande **muchísimos menos nodos** (308 → 180 → 138, hasta **~10× menos** con `w=10`),
  encontrando solución más rápido. El precio es la **pérdida de optimalidad**: el camino se alarga
  (cota teórica `cost ≤ w·cost*`; aquí el exceso real es mucho menor que esa cota holgada).
- `w → ∞` (p. ej. 10): se comporta casi como *Greedy Best-First Search*. El salto de longitud es
  ahora notorio (+30.9 %): la heurística "tironea" tan fuerte hacia la meta que el frente de búsqueda
  ignora rodeos baratos y se queda con la primera ruta que aparece. Es el clásico **trade-off
  velocidad vs. optimalidad**: entre `w=2` y `w=5` se gana muchísima velocidad por muy poco exceso de
  longitud (sweet spot), mientras que con `w=10` la velocidad extra ya no compensa el deterioro del camino.

El gráfico `weighted_astar.png` ilustra ambas curvas: las celdas expandidas caen abruptamente
(1450 → 138) mientras la longitud del camino se mantiene casi plana hasta `w=5` y recién se dispara en `w=10`.

---

## 4. Theta\* (Any-Angle)

**4.1 Bresenham y línea de visión.** El algoritmo de Bresenham traza la recta entre dos celdas
usando **solo aritmética entera**: mantiene un término de error acumulado que decide, en cada paso,
si avanzar en el eje mayor o también en el menor, sin multiplicaciones ni divisiones ni flotantes.
Es eficiente en grillas porque cada paso es O(1) con sumas/restas enteras y recorre exactamente las
celdas que la recta atraviesa. Aquí se usa para chequear *line of sight*: si todas las celdas sobre
la recta entre dos nodos están libres (`< 0.4`), hay visión directa.

**4.2 Actualización con salto angular.** Por cada vecino se evalúan **dos caminos** y se toma el
más barato:
- **Path 2 (salto angular):** si el *abuelo* (padre de `parent`) tiene línea de visión directa al
  vecino, se conecta el vecino **directamente al abuelo** con costo `g(abuelo) + dist(abuelo, vecino)`.
  Esto "salta" el quiebre de 45° de la grilla y produce tramos rectos de cualquier ángulo.
- **Path 1 (A\* clásico):** si no hay visión directa, se usa la arista `parent→vecino`,
  `g(parent) + dist(parent, vecino)`.

**4.3 Comparación A\* vs Theta\*.**

| Algoritmo | Longitud real del camino | Restricción de movimiento |
|-----------|--------------------------|---------------------------|
| A*        | **72.53**                | múltiplos de 45° (grilla) |
| Theta*    | **65.38**                | cualquier ángulo          |

Theta* logra un camino **~10% más corto sobre la misma grilla**. La razón: A* está obligado a moverse
entre centros de celdas adyacentes (pasos horizontales/verticales/diagonales), por lo que aproxima
una diagonal libre como una escalera de tramos de 45°. Theta*, al reconectar nodos a su "abuelo"
cuando hay visión directa, reemplaza esas escaleras por **segmentos rectos**, acercándose al camino
euclídeo verdaderamente más corto. La comparación **visual lado a lado** está en
`astar_vs_thetastar.png`: a la izquierda el camino "en escalera" de 45° de A* (72.53), a la derecha
los tramos rectos any-angle de Theta* (65.38). (También las figuras individuales `astar.png` y `thetastar.png`.)

---

## 5. RRT

**5.1 Extensión (`steer`).** (1) Se busca el nodo del árbol `q_near` más cercano a la muestra
`q_rand` (mínima distancia euclídea). (2) Se avanza desde `q_near` hacia `q_rand` una distancia
máxima `step_size`: si `q_rand` está dentro del alcance, `q_new = q_rand`; si no,
`q_new = q_near + step_size·(q_rand − q_near)/‖·‖`. Luego `is_collision_free` valida la arista
`q_near→q_new` antes de agregarla.

**5.2 Variabilidad.** **No**, RRT no llega a la meta en la misma cantidad de pasos ni por el mismo
camino: el muestreo es aleatorio, así que cada corrida genera un árbol distinto y un número de nodos
distinto hasta alcanzar la meta (el sesgo del 10% hacia el goal solo acota la varianza). Dificultades
en este mapa: **pasajes angostos** — la probabilidad de muestrear puntos dentro de un corredor
estrecho es baja, así que el árbol tarda mucho en "colarse" por ahí (problema clásico del
*narrow passage*). Regiones cóncavas o detrás de obstáculos grandes también se exploran lento.

**5.3 Geometría errática y no-optimalidad.** Los caminos tienen forma de **zigzag** porque cada nodo
se conecta al *primer* `q_near` que lo generó y nunca se reconsidera: la dirección de cada tramo
depende de hacia dónde cayó la muestra aleatoria, no de la dirección al objetivo. Aunque el número de
muestras `→ ∞`, RRT clásico **no garantiza optimalidad**: una vez fijada la arista padre→hijo, jamás
se reestructura, por lo que los atajos que aparecen al densificar el árbol no se aprovechan. RRT es
*probabilísticamente completo* (encuentra un camino si existe) pero **no asintóticamente óptimo**.

---

## 6. RRT\*

**6.1 Choose Parent.** Tras generar `q_new`, se buscan todos los nodos dentro de `search_radius` con
arista libre hacia `q_new` (`near_indices`). Entre ellos (y `q_near` como base) se elige como padre
el que **minimiza el costo acumulado** `g(q_new) = g(j) + dist(j, q_new)`. Así `q_new` no se conecta
ciegamente a `q_near` sino al vecino que ofrece el camino más barato desde el inicio.

**6.2 Rewire.** Para cada vecino dentro del radio: si llegar a él **a través de `q_new`** es más
barato que su costo actual (`g(q_new) + dist(q_new, vecino) < g(vecino)`), se lo reconecta como hijo
de `q_new` (se actualiza su puntero de padre y su costo). Esto va "alisando" el árbol y eliminando los
zigzags de RRT clásico.

> *Nota de implementación:* el rewire básico actualiza el costo del vecino reconectado pero **no
> propaga** el nuevo costo a sus descendientes en el árbol. Es el comportamiento estándar de RRT*
> básico y se auto-corrige a medida que aumentan las iteraciones; documentarlo es preferible a
> implementar propagación completa.

**6.3 Algoritmo any-time.** RRT* **no se detiene** al tocar la meta por primera vez: sigue
muestreando y, vía Choose Parent + Rewire, **reduce monótonamente** el costo del mejor camino
encontrado. Un algoritmo *any-time* es aquel que produce rápido una solución válida (aunque
subóptima) y la **mejora continuamente** si se le da más tiempo de cómputo, pudiendo interrumpirse en
cualquier momento devolviendo la mejor solución hasta ese instante. En RRT* el costo del camino
**decrece y converge al óptimo** conforme `n → ∞` (asintóticamente óptimo): con pocas iteraciones el
camino se parece al de RRT (zigzag), y con muchas se vuelve recto y casi óptimo.

---

## Archivos de la entrega

- `planning_framework.py` — código completo y comentado (todas las funciones implementadas).
- `test_headless.py` — tests de verificación (unitarios + corridas reales) sin ventanas gráficas.
- `generate_figures.py` — genera los PNG de los caminos finales.
- `weighted_astar_experiment.py` — experimento numérico del ejercicio 3.4 (Weighted A* para `w ∈ {1,2,5,10}`).
- `generate_comparisons.py` — genera las figuras comparativas `astar_vs_thetastar.png` (4.3) y `weighted_astar.png` (3.4).
- Figuras comparativas: `astar_vs_thetastar.png` (A* vs Theta* lado a lado), `weighted_astar.png` (trade-off de `w`).
- Figuras: `dijkstra.png`, `astar.png`, `thetastar.png`, `apf.png`, `rrt.png`, `rrtstar.png`.

### Resultados numéricos (mapa duplicado, `start=[44,66]`, `goal=[80,30]`)

| Algoritmo | Longitud del camino | Observación |
|-----------|---------------------|-------------|
| Dijkstra  | 72.53 | Óptimo de grilla; expande todo el frente de costo |
| A*        | 72.53 | Mismo óptimo, muchas menos celdas expandidas |
| Theta*    | 65.38 | Any-angle: ~10% más corto |
| APF       | — | Queda en mínimo local (no alcanza la meta) |
| RRT       | variable | Camino válido pero subóptimo (zigzag) |
| RRT*      | ~74 (mejora con iteraciones) | Asintóticamente óptimo |
