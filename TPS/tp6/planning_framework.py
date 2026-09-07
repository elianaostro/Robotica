import numpy as np
import matplotlib.pyplot as plt
import random

# =============================================================================
# 1. SECCIÓN: CAMPOS POTENCIALES ARTIFICIALES (APF)
# =============================================================================

def get_attractive_force(cell, goal, k_att=1.0, d_switch=10.0):
    # (Ejercicio 1.1)
    # Modelo de potencial atractivo conico/cuadratico (combinado):
    #   - Lejos de la meta (dist > d_switch): potencial CONICO -> fuerza de magnitud
    #     constante (k_att). Esto evita fuerzas enormes cuando se esta lejos.
    #   - Cerca de la meta (dist <= d_switch): potencial CUADRATICO -> fuerza lineal
    #     en la distancia (k_att * dist), que decae suavemente al acercarse a la meta.
    # En ambos casos el vector apunta desde la celda actual hacia la meta.
    cell = np.asarray(cell, dtype=float)
    goal = np.asarray(goal, dtype=float)
    diff = goal - cell                      # vector celda -> meta
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return np.zeros(2)
    direction = diff / dist                 # vector unitario hacia la meta
    if dist <= d_switch:
        magnitude = k_att * dist            # zona cuadratica (fuerza proporcional a la distancia)
    else:
        magnitude = k_att * d_switch        # zona conica (magnitud constante)
    f_att = magnitude * direction
    return f_att

def get_repulsive_force(cell, occ_map, k_rep=100.0, d_0=5.0, occ_thresh=0.4):
    # (Ejercicio 1.2)
    # Fuerza repulsiva de tipo FIRAS (Khatib). Solo actua si la distancia al obstaculo
    # es menor que la distancia de influencia critica d_0. Para cada celda ocupada
    # (prob >= occ_thresh) dentro de una ventana de radio d_0 alrededor de la celda actual:
    #   F = k_rep * (1/dist - 1/d_0) * (1/dist^2) * (vector_unitario obstaculo->celda)
    # Esta magnitud crece sin limite cuando dist -> 0 y se anula exactamente en dist = d_0,
    # garantizando continuidad. Sumamos la contribucion de todos los obstaculos cercanos.
    f_rep = np.zeros(2)
    cell = np.asarray(cell, dtype=float)

    ci, cj = int(round(cell[0])), int(round(cell[1]))
    r = int(np.ceil(d_0))
    i_min, i_max = max(0, ci - r), min(occ_map.shape[0], ci + r + 1)
    j_min, j_max = max(0, cj - r), min(occ_map.shape[1], cj + r + 1)

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            if occ_map[i, j] < occ_thresh:
                continue
            obs = np.array([i, j], dtype=float)
            diff = cell - obs                  # vector obstaculo -> celda (empuja al robot)
            dist = np.linalg.norm(diff)
            if dist < 1e-9 or dist >= d_0:
                continue
            magnitude = k_rep * (1.0 / dist - 1.0 / d_0) * (1.0 / (dist * dist))
            f_rep += magnitude * (diff / dist)
    return f_rep

def run_potential_fields(occ_map, start, goal, max_steps=500, step_size=0.5):
    plot_map(occ_map, start, goal)
    current = np.array(start, dtype=float)
    path = [np.copy(current)]
    
    for _ in range(max_steps):
        if np.linalg.norm(current - goal) < 1.0:
            print("Campos Potenciales: ¡Meta alcanzada!")
            break
            
        f_att = get_attractive_force(current, goal)
        f_rep = get_repulsive_force(current, occ_map)
        f_total = f_att + f_rep
        
        # Normalizar y avanzar paso
        if np.linalg.norm(f_total) > 0.01:
            current += (f_total / np.linalg.norm(f_total)) * step_size
        path.append(np.copy(current))
        plt.plot(current[0], current[1], 'bo', markersize=2)
        plt.pause(0.001)

        if len(path) >= 10 and np.all(np.abs(path[-10:] - path[-1]) < 1.0):  # Si no hay progreso, romper para evitar bucle infinito
            print("Campos Potenciales: Estancado, terminando.")
            break
      
      
        
    plt.waitforbuttonpress()

# =============================================================================
# 2. FUNCIONES AUXILIARES Y DE INFRAESTRUCTURA (A* / DIJKSTRA)
# =============================================================================

def get_neighborhood(cell, occ_map_shape):
    # (Ejercicio 2.1)
    # Devuelve una lista de celdas vecinas (8-conectividad) dentro de los limites del mapa.
    # Recorre los 8 desplazamientos (horizontal, vertical y diagonal) y descarta los que
    # caen fuera del mapa.
    neighbors = []
    cx, cy = int(cell[0]), int(cell[1])
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < occ_map_shape[0] and 0 <= ny < occ_map_shape[1]:
                neighbors.append(np.array([nx, ny]))
    return neighbors

def get_edge_cost(parent, child, occ_map, occ_thresh=0.4, w_occ=10.0):
    # (Ejercicio 2.2 y 2.3)
    # Costo del arco parent -> child.
    # 2.2: si la probabilidad de ocupacion de child supera el umbral, es un obstaculo
    #      y el arco es intransitable (costo infinito). Elegimos 0.4 para ser consistentes
    #      con bresenham_line_of_sight e is_collision_free (mismo espacio libre para todos
    #      los planificadores).
    # 2.3: al costo geometrico (longitud euclidea del arco: 1 en recto, sqrt(2) en diagonal)
    #      le sumamos un termino proporcional a la ocupacion de la celda destino. Asi, ante
    #      caminos de longitud similar, el algoritmo prefiere celdas con baja probabilidad
    #      de ocupacion (mas seguras) sobre celdas con probabilidad alta.
    child_occ = occ_map[int(child[0]), int(child[1])]
    if child_occ >= occ_thresh:
        return np.inf
    parent = np.asarray(parent, dtype=float)
    child = np.asarray(child, dtype=float)
    dist = np.linalg.norm(child - parent)          # longitud geometrica del arco
    edge_cost = dist * (1.0 + w_occ * child_occ)   # penalizacion por ocupacion
    return edge_cost

def run_dijkstra(occ_map, start, goal):
    '''
    Calcula el camino de costo mínimo utilizando el algoritmo de Dijkstra.
    La heurística se fija explícitamente en 0.
    '''
    print("Ejecutando Algoritmo de Dijkstra...")
    plot_map(occ_map, start, goal)

    costs = np.ones(occ_map.shape) * np.inf
    closed_flags = np.zeros(occ_map.shape)
    predecessors = -np.ones(occ_map.shape + (2,), dtype=np.int32)

    costs[start[0], start[1]] = 0
    parent = start

    skipper = 0

    while not np.array_equal(parent, goal):
        # Dijkstra solo evalúa el costo acumulado 'costs' sin sumar heurística
        open_costs = np.where(closed_flags == 1, np.inf, costs)

        x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)
        
        if open_costs[x, y] == np.inf:
            break  # No hay más nodos alcanzables
        
        parent = np.array([x, y])
        closed_flags[x, y] = 1
        
        # (Ejercicio 2.4)
        # Relajacion: para cada vecino no expandido, si llegar a el a traves de 'parent'
        # cuesta menos que su costo actual, actualizamos su costo y su predecesor.
        for neighbor in get_neighborhood(parent, occ_map.shape):
            nx, ny = int(neighbor[0]), int(neighbor[1])
            if closed_flags[nx, ny] == 1:
                continue
            tentative = costs[parent[0], parent[1]] + get_edge_cost(parent, neighbor, occ_map)
            if tentative < costs[nx, ny]:
                costs[nx, ny] = tentative
                predecessors[nx, ny] = parent
        
        skipper += 1
        if skipper % 15 == 0:
          plot_expanded(parent, start, goal)
        else:
          plot_expanded(parent, start, goal, wait=False)
    
    # Reconstrucción del camino
    if np.array_equal(parent, goal):
        reconstruct_and_plot_path(predecessors, costs, start, goal, closed_flags)
    else:
        print("No se encontró un camino válido.")

def get_heuristic(cell, goal):
    # (Ejercicio 3.2)
    # Distancia euclidea hasta la meta. Es admisible (nunca sobreestima el costo real:
    # ninguna trayectoria sobre la grilla puede ser mas corta que la linea recta) y
    # consistente (cumple la desigualdad triangular), por lo que A* es optimo.
    # Usamos euclidea (no octile) porque get_heuristic la comparte run_thetastar, donde
    # el costo real puede ser exactamente la linea recta any-angle; octile sobreestimaria
    # ese costo y volveria Theta* no optimo.
    cell = np.asarray(cell, dtype=float)
    goal = np.asarray(goal, dtype=float)
    heuristic = np.linalg.norm(cell - goal)
    return heuristic

def run_astar(occ_map, start, goal, w=1.0, plot=True):
    '''
    Calcula el camino de costo mínimo utilizando el algoritmo A* (Búsqueda Informada).
    El parámetro w pondera la heurística: f(n) = g(n) + w·h(n) (Weighted A*, ejercicio 3.4).
      - w = 1  -> A* estándar (heurística admisible, camino óptimo).
      - w > 1  -> búsqueda más voraz: expande menos nodos pero el camino puede ser subóptimo.
    plot=False desactiva el dibujo (útil para correr experimentos en lote).
    Devuelve (celdas_expandidas, costo_g, longitud_real) o None si no hay camino.
    '''
    print(f"Ejecutando Algoritmo A* (w={w})...")
    plot_map(occ_map, start, goal)

    costs = np.ones(occ_map.shape) * np.inf
    closed_flags = np.zeros(occ_map.shape)
    predecessors = -np.ones(occ_map.shape + (2,), dtype=np.int32)

    # Precalcular matriz de heurística para optimizar el bucle
    heuristic = np.zeros(occ_map.shape)
    for x in range(occ_map.shape[0]):
        for y in range(occ_map.shape[1]):
            heuristic[x, y] = get_heuristic([x, y], goal)

    costs[start[0], start[1]] = 0
    parent = start

    skipper = 0

    while not np.array_equal(parent, goal):
        # A* selecciona basándose en f(n) = g(n) + w·h(n)  (w=1 -> A* clásico)
        open_costs = np.where(closed_flags == 1, np.inf, costs) + w * heuristic

        x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)

        if open_costs[x, y] == np.inf:
            break

        parent = np.array([x, y])
        closed_flags[x, y] = 1

        # (Ejercicio 3.3)
        # Misma relajacion que Dijkstra: g(child) = g(parent) + costo_arco. La heuristica
        # se suma aparte al elegir el nodo a expandir (linea open_costs = ... + heuristic),
        # por eso aqui solo actualizamos el costo acumulado real g.
        for neighbor in get_neighborhood(parent, occ_map.shape):
            nx, ny = int(neighbor[0]), int(neighbor[1])
            if closed_flags[nx, ny] == 1:
                continue
            tentative = costs[parent[0], parent[1]] + get_edge_cost(parent, neighbor, occ_map)
            if tentative < costs[nx, ny]:
                costs[nx, ny] = tentative
                predecessors[nx, ny] = parent

        skipper += 1
        if plot:
            if skipper % 5 == 0:
                plot_expanded(parent, start, goal)
            else:
                plot_expanded(parent, start, goal, wait=False)

    if np.array_equal(parent, goal):
        path_length, path = reconstruct_and_plot_path(predecessors, costs, start, goal, closed_flags, plot=plot)
        return int(np.count_nonzero(closed_flags)), float(costs[goal[0], goal[1]]), float(path_length), path
    else:
        print("No se encontró un camino válido.")
        return None

def reconstruct_and_plot_path(predecessors, costs, start, goal, closed_flags, plot=True):
    parent = goal
    path_length = 0
    skipper = 0
    path = [np.array(goal, dtype=float)]          # nodos del camino (meta -> ... -> inicio)
    while predecessors[parent[0], parent[1]][0] >= 0:
        skipper += 1
        if plot:
            if skipper % 5 == 0:
              plot_path(parent, goal)
            else:
              plot_path(parent, goal, wait = False)
        predecessor = predecessors[parent[0], parent[1]]
        path_length += np.linalg.norm(parent - predecessor)
        parent = predecessor
        path.append(np.array(parent, dtype=float))
    path.reverse()                                # inicio -> meta

    print("Meta alcanzada  : " + str(goal))
    print("Celdas expandidas: " + str(np.count_nonzero(closed_flags)))
    print("Costo del camino : " + str(costs[goal[0], goal[1]]))
    print("Longitud real    : " + str(path_length))

    if plot:
        plot_path(start, goal)
        print("\n[INFO] Mostrando mapa final. Cierre la ventana gráfica para terminar.")
        plt.show()

    return path_length, path

# =============================================================================
# 4. THETA* (A* CON SALTO DE ÁNGULO)
# =============================================================================

def bresenham_line_of_sight(p1, p2, occ_map):
    '''
    Determina si hay línea de visión directa entre p1 y p2 usando el algoritmo de Bresenham.
    Arguments:
    p1, p2 -- Coordenadas de celda como [y, x] o [x, y] de manera consistente.
    occ_map -- Matriz del mapa de ocupación.
    
    Output:
    True si el camino está libre de obstáculos, False de lo contrario.
    '''
    y1, x1 = int(p1[0]), int(p1[1])
    y2, x2 = int(p2[0]), int(p2[1])
    
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    
    sy = 1 if y1 < y2 else -1
    sx = 1 if x1 < x2 else -1
    
    err = dx - dy
    
    while True:
        # Verificar límites del mapa
        if y1 < 0 or y1 >= occ_map.shape[0] or x1 < 0 or x1 >= occ_map.shape[1]:
            return False
        # Verificar si la celda actual es un obstáculo (umbral de ocupación >= 0.4)
        if occ_map[y1, x1] >= 0.4:
            return False
            
        if y1 == y2 and x1 == x2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
            
    return True

def run_thetastar(occ_map, start, goal):
    print("Ejecutando Theta* (Any-Angle Path Planning)...")
    plot_map(occ_map, start, goal)
    
    costs = np.ones(occ_map.shape) * np.inf
    closed_flags = np.zeros(occ_map.shape)
    predecessors = -np.ones(occ_map.shape + (2,), dtype=np.int32)
    
    # Precalcular matriz de heurística de forma limpia
    heuristic = np.zeros(occ_map.shape)
    for x in range(occ_map.shape[0]):
        for y in range(occ_map.shape[1]):
            heuristic[x, y] = get_heuristic([x, y], goal)
    
    costs[start[0], start[1]] = 0
    predecessors[start[0], start[1]] = start  # El padre de start es start inicialmente
    
    parent = start

    skipper = 0
    
    while not np.array_equal(parent, goal):
        # f(n) = g(n) + h(n)
        open_costs = np.where(closed_flags == 1, np.inf, costs) + heuristic
        
        x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)
        if open_costs[x, y] == np.inf: 
            break
        
        parent = np.array([x, y])
        closed_flags[x, y] = 1
        skipper += 1
        if skipper % 10 == 0:  # Plot expanded nodes every 10 iterations
            plot_expanded(parent, start, goal)
        else:
            plot_expanded(parent, start, goal, wait=False)
        
        # (Ejercicio 4.2)
        # Theta* evalua dos formas de llegar a cada vecino y se queda con la mas barata:
        #   PATH 2 (salto angular): si el ABUELO (padre de 'parent') tiene linea de vision
        #     directa al vecino, conectamos el vecino directamente al abuelo con costo
        #     g(abuelo) + dist(abuelo, vecino). Esto "salta" el quiebre de 45 grados y
        #     genera trayectorias rectas de cualquier angulo.
        #   PATH 1 (A* clasico): si no hay linea de vision, usamos la arista parent->vecino
        #     con costo g(parent) + dist(parent, vecino).
        grandparent = predecessors[parent[0], parent[1]]
        for neighbor in get_neighborhood(parent, occ_map.shape):
            nx, ny = int(neighbor[0]), int(neighbor[1])
            if closed_flags[nx, ny] == 1:
                continue
            if occ_map[nx, ny] >= 0.4:        # vecino ocupado: intransitable
                continue

            if bresenham_line_of_sight(grandparent, neighbor, occ_map):
                # PATH 2: camino directo desde el abuelo
                tentative = (costs[grandparent[0], grandparent[1]]
                             + np.linalg.norm(neighbor - grandparent))
                if tentative < costs[nx, ny]:
                    costs[nx, ny] = tentative
                    predecessors[nx, ny] = grandparent
            else:
                # PATH 1: camino clasico desde el padre
                tentative = (costs[parent[0], parent[1]]
                             + np.linalg.norm(neighbor - parent))
                if tentative < costs[nx, ny]:
                    costs[nx, ny] = tentative
                    predecessors[nx, ny] = parent
                        
    # Reconstrucción del camino Any-Angle
    if np.array_equal(parent, goal):
        print("Theta*: ¡Camino Encontrado!")
        path_length = 0
        path = [np.array(goal, dtype=float)]      # nodos any-angle (meta -> ... -> inicio)
        tmp = predecessors[parent[0], parent[1]]
        plt.plot([tmp[0], goal[0]], [tmp[1], goal[1]],
                 'b-', linewidth=2, zorder=5)
        while not np.array_equal(predecessors[parent[0], parent[1]], parent):
            predecessor = predecessors[parent[0], parent[1]]

            # PASAMOS EL PREDECESOR AQUÍ PARA QUE DIBUJE LA LÍNEA DE VISIÓN RECTA
            plot_path(parent, goal, wait=False, predecessor=predecessor)

            path_length += np.linalg.norm(parent - predecessor)
            parent = predecessor
            path.append(np.array(parent, dtype=float))
        path.reverse()                            # inicio -> meta

        # Forzar el render final en la pantalla
        plt.draw()
        plt.pause(0.001)

        print("Celdas expandidas: " + str(np.count_nonzero(closed_flags)))
        print("Costo del camino : " + str(costs[goal[0], goal[1]]))
        print("Longitud real    : " + str(path_length))

        print("\n[INFO] Mostrando mapa final. Cierre la ventana gráfica para terminar.")
        plt.show()
        return int(np.count_nonzero(closed_flags)), float(costs[goal[0], goal[1]]), float(path_length), path
    else:
        print("No se encontró un camino válido con Theta*.")
        print("\n[INFO] Mostrando mapa final. Cierre la ventana gráfica para terminar.")
        plt.show()
        return None

# =============================================================================
# 5. SECCIÓN: RRT (RAPIDLY-EXPLORING RANDOM TREES)
# =============================================================================

def is_collision_free(p1, p2, occ_map):
    # Verificación rápida por muestreo discreto entre dos puntos continuos
    steps = int(np.linalg.norm(p2 - p1) * 20) + 1
    for t in np.linspace(0, 1, steps):
        p = p1 + t * (p2 - p1)
        ny, nx = int(round(p[0])), int(round(p[1]))
        if ny < 0 or ny >= occ_map.shape[0] or nx < 0 or nx >= occ_map.shape[1]:
            return False
        if occ_map[ny, nx] >= 0.4:
            return False
    return True

def run_rrt(occ_map, start, goal, max_nodes=10000, step_size=2.0):
    plot_map(occ_map, start, goal)
    nodes = [np.array(start)]
    parents = {0: None}
    
    found_goal = False
    for i in range(max_nodes):
        if random.random() < 0.1:
            q_rand = np.array(goal)
        else:
            q_rand = np.array([random.uniform(0, occ_map.shape[0]-1), 
                               random.uniform(0, occ_map.shape[1]-1)])
            
        # (Ejercicio 5.1) [steer]
        # q_rand es un punto aleatorio en el mapa o el objetivo con cierta probabilidad (sesgo hacia la meta)
        # Deberán primero encontrar el nodo más cercano en el árbol a q_rand (q_near) y luego extenderse hacia q_rand desde q_near
        # con una distancia máxima de step_size. El nuevo nodo se llamará q_new.
        # Deberán poblar estas tres variables, siendo q_near el nodo más cercano a q_rand,
        # q_new el nuevo nodo extendido desde q_near hacia q_rand, y idx_nearest el índice de q_near en la lista de nodos.
        # Noten que luego se utiliza la funcion is_collision_free para verificar si la arista entre q_near y q_new
        # es válida antes de agregar q_new al árbol.
        # 1) Nodo mas cercano del arbol a q_rand
        dists = [np.linalg.norm(q_rand - n) for n in nodes]
        idx_nearest = int(np.argmin(dists))
        q_near = nodes[idx_nearest].astype(float)

        # 2) Extension (steer): avanzar desde q_near hacia q_rand como maximo step_size
        direction = q_rand - q_near
        d = np.linalg.norm(direction)
        if d <= step_size:
            q_new = q_rand.astype(float)            # q_rand esta dentro del alcance
        else:
            q_new = q_near + (direction / d) * step_size

        if is_collision_free(q_near, q_new, occ_map):
            nodes.append(q_new)
            new_idx = len(nodes) - 1
            parents[new_idx] = idx_nearest
            
            plt.plot([q_near[0], q_new[0]], [q_near[1], q_new[1]], 'y-', alpha=0.6)
            if i % 10 == 0: plt.pause(1e-5)
            
            if np.linalg.norm(q_new - goal) < step_size:
                if is_collision_free(q_new, goal, occ_map):
                    nodes.append(np.array(goal))
                    parents[len(nodes)-1] = new_idx
                    plt.plot([q_new[0], goal[0]], [q_new[1], goal[1]], 'y-', alpha=0.6)
                    found_goal = True
                    break
        # ----------------------------------

    if not found_goal:
        print("RRT: No se encontró un camino válido.")
    else:  
        print("RRT: ¡Camino Encontrado! Dibujando solución...")
        curr_idx = len(nodes) - 1
        while curr_idx is not None:
            p = nodes[curr_idx]
            if parents[curr_idx] is not None:
                p_next = nodes[parents[curr_idx]]
                # zorder=5 obliga a la línea a renderizarse arriba de todo
                plt.plot([p[0], p_next[0]], [p[1], p_next[1]], 'b-', linewidth=2.5, zorder=5)
            curr_idx = parents[curr_idx]
            
        plt.draw() # Forzar actualización del buffer gráfico
        plt.pause(0.001)
              
    print("\n[INFO] Cierre la ventana gráfica para terminar.")
    plt.show()


def run_rrt_star(occ_map, start, goal, max_nodes=5000, step_size=3.0, search_radius=6.0):
    print("Ejecutando RRT* (Optimal Path Planning)...")
    plot_map(occ_map, start, goal)
    
    nodes = [np.array(start)]
    parents = {0: None}
    
    # Costo acumulado g(n) real desde el nodo de inicio a cada nodo del árbol
    costs = {0: 0.0} 
    
    found_goal = False
    goal_idx = None
    min_goal_cost = np.inf

    for i in range(max_nodes):
        # Muestreo con sesgo hacia la meta
        if random.random() < 0.1:
            q_rand = np.array(goal)
        else:
            q_rand = np.array([random.uniform(0, occ_map.shape[0]-1), 
                               random.uniform(0, occ_map.shape[1]-1)])
            
        # (Ejercicio 6)
        # Al igual que en RRT, deberán encontrar el nodo más cercano q_near y extenderse hacia
        # q_rand para obtener q_new. Sin embargo, en RRT* no basta con agregar q_new al árbol,
        # sino que también deben optimizar su conexión al árbol y re-cablear a los vecinos si es necesario
        # Primero debemos encontrar el nodo más cercano q_near a q_rand y su índice idx_nearest,
        # luego extendernos hacia q_rand para obtener q_new
        # Nodo mas cercano y extension (igual que en RRT)
        dists = [np.linalg.norm(q_rand - n) for n in nodes]
        idx_nearest = int(np.argmin(dists))
        q_near = nodes[idx_nearest].astype(float)
        direction = q_rand - q_near
        d = np.linalg.norm(direction)
        if d <= step_size:
            q_new = q_rand.astype(float)
        else:
            q_new = q_near + (direction / d) * step_size

        if is_collision_free(q_near, q_new, occ_map):
            # "Choose Parent": elegir el mejor padre para q_new entre los nodos cercanos.
            # PASO 1: candidatos = nodos dentro de search_radius con arista libre hacia q_new
            near_indices = [j for j in range(len(nodes))
                            if np.linalg.norm(nodes[j] - q_new) <= search_radius
                            and is_collision_free(nodes[j].astype(float), q_new, occ_map)]

            # PASO 2: entre los candidatos, elegir el que minimiza g(q_new) = g(j) + dist(j, q_new).
            # Inicializamos con q_near (siempre valido por la verificacion de colision previa).
            idx_best = idx_nearest
            cost_best = costs[idx_nearest] + np.linalg.norm(q_new - q_near)
            for j in near_indices:
                c = costs[j] + np.linalg.norm(q_new - nodes[j])
                if c < cost_best:
                    cost_best = c
                    idx_best = j

            # Insertar oficialmente el nuevo nodo optimizado al árbol
            nodes.append(q_new)
            new_idx = len(nodes) - 1
            parents[new_idx] = idx_best
            costs[new_idx] = cost_best

            # PASO 3: "Rewire" - si llegar a un vecino A TRAVES de q_new es mas barato que su
            # costo actual, lo reconectamos como hijo de q_new.
            for idx_near in near_indices:
                c_through_new = costs[new_idx] + np.linalg.norm(nodes[idx_near] - q_new)
                if c_through_new < costs[idx_near]:
                    parents[idx_near] = new_idx
                    costs[idx_near] = c_through_new
            
            # Control interactivo e hitos hacia la meta
            if np.linalg.norm(q_new - goal) < step_size:
                if is_collision_free(q_new, goal, occ_map):
                    cost_to_goal = costs[new_idx] + np.linalg.norm(q_new - goal)
                    if cost_to_goal < min_goal_cost:
                        min_goal_cost = cost_to_goal
                        # Si ya existía un nodo meta anterior, lo sobreescribimos con una mejor arista
                        if goal_idx is not None:
                            parents[goal_idx] = new_idx
                            costs[goal_idx] = min_goal_cost
                        else:
                            nodes.append(np.array(goal))
                            goal_idx = len(nodes) - 1
                            parents[goal_idx] = new_idx
                            costs[goal_idx] = min_goal_cost
                        found_goal = True

            # Dibujo asincrónico del árbol dinámico (limpiando frames para notar el re-cableado)
            if i % 50 == 0:
                plt.clf()
                plot_map(occ_map, start, goal)
                for idx, p_idx in parents.items():
                    if p_idx is not None:
                        plt.plot([nodes[idx][0], nodes[p_idx][0]], [nodes[idx][1], nodes[p_idx][1]], 'y-', alpha=0.4)
                plt.pause(1e-5)

    # Reconstrucción del camino óptimo final
    if found_goal:
        print("RRT*: ¡Camino Óptimo Encontrado!")
        curr_idx = goal_idx
        path_length = 0
        while curr_idx is not None:
            p = nodes[curr_idx]
            if parents[curr_idx] is not None:
                p_next = nodes[parents[curr_idx]]
                plt.plot([p[0], p_next[0]], [p[1], p_next[1]], 'b-', linewidth=2.5, zorder=5)
                path_length += np.linalg.norm(p - p_next)
            curr_idx = parents[curr_idx]
            
        plt.draw()
        print("Nodos totales generados : " + str(len(nodes)))
        print("Longitud real del camino: " + str(path_length))
    else:
        print("RRT*: No se encontró un camino válido.")
        
    print("\n[INFO] Mostrando mapa final. Cierre la ventana gráfica para terminar.")
    plt.show()

# =============================================================================
# FUNCIONES PRINCIPALES Y DE VISUALIZACIÓN ORIGINALES
# =============================================================================

def plot_map(occ_map, start, goal):
    plt.clf()
    plt.imshow(occ_map.T, cmap=plt.cm.gray, interpolation='none', origin='upper')
    plt.plot([start[0]], [start[1]], 'ro', label='Inicio')
    plt.plot([goal[0]], [goal[1]], 'go', label='Meta')
    plt.axis([0, occ_map.shape[0]-1, 0, occ_map.shape[1]-1])
    plt.xlabel('x')
    plt.ylabel('y')

def plot_expanded(expanded, start, goal, wait=True):
    if np.array_equal(expanded, start) or np.array_equal(expanded, goal): return
    plt.plot([expanded[0]], [expanded[1]], 'yo', markersize=3)
    if wait:
        plt.pause(1e-5)

def plot_path(path, goal, wait=True, predecessor=None):
    if np.array_equal(path, goal): 
        return
        
    if predecessor is not None:
        # Dibujamos una línea recta continua entre el predecesor y el nodo actual
        # zorder=5 asegura que se pinte por encima de las celdas expandidas y del mapa
        plt.plot([predecessor[0], path[0]], [predecessor[1], path[1]], 
                 'b-', linewidth=2, zorder=5)
    else:
        # Comportamiento por defecto (puntos para Dijkstra/A* clásico)
        plt.plot([path[0]], [path[1]], 'bo', markersize=4, zorder=5)
        
    if wait:
        plt.pause(1e-5)

def main():
    # Cargar mapa de ocupación falso para pruebas locales
    # En producción usar: occ_map = np.loadtxt('map.txt')
    occ_map = np.loadtxt('map.txt')
    # double the size of the map for better visualization
    occ_map = np.kron(occ_map, np.ones((2, 2)))
    
    start = np.array([44,66])
    goal = np.array([80, 30])

    # Selector de algoritmos para que el alumno evalúe sus implementaciones
    seleccion = int(input("Seleccione Algoritmo: 1:APF | 2:A* | 3:Dijkstra | 4:Theta* | 5:RRT | 6:RRT* :   "))
    
    if seleccion == 1:
        run_potential_fields(occ_map, start, goal)
    elif seleccion == 2:
        run_astar(occ_map, start, goal)
    elif seleccion == 3:
        run_dijkstra(occ_map, start, goal)
    elif seleccion == 4:
        run_thetastar(occ_map, start, goal)
    elif seleccion == 5:
        run_rrt(occ_map, start, goal)
    elif seleccion == 6:
        run_rrt_star(occ_map, start, goal)

if __name__ == "__main__":
    main()