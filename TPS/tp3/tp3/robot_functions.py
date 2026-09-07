import copy
import numpy as np
import random


class particle():

    def __init__(self):
        self.x = (random.random()-0.5)*2  # initial x position
        self.y = (random.random()-0.5)*2 # initial y position
        self.orientation = random.uniform(-np.pi,np.pi) # initial orientation
        self.weight = 1.0

    def set(self, new_x, new_y, new_orientation):
        '''
        set: sets a robot coordinate, including x, y and orientation
        '''
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self, odom, noise):
        '''
        move_odom: Takes in Odometry data and moves the robot based on the odometry data.
        Applies the standard probabilistic odometry motion model (Thrun et al., ch. 5).

        noise = [alpha1, alpha2, alpha3, alpha4]
          alpha1/2 control noise in rotations, alpha3/4 control noise in translation.
        '''
        dist       = odom['t']
        delta_rot1 = odom['r1']
        delta_rot2 = odom['r2']

        alpha1, alpha2, alpha3, alpha4 = noise

        # Sample independent Gaussian noise for each motion component
        std_r1 = np.sqrt(max(alpha1 * delta_rot1**2 + alpha2 * dist**2, 1e-12))
        std_t  = np.sqrt(max(alpha3 * dist**2 + alpha4 * (delta_rot1**2 + delta_rot2**2), 1e-12))
        std_r2 = np.sqrt(max(alpha1 * delta_rot2**2 + alpha2 * dist**2, 1e-12))

        r1_hat = delta_rot1 + np.random.normal(0.0, std_r1)
        t_hat  = dist       + np.random.normal(0.0, std_t)
        r2_hat = delta_rot2 + np.random.normal(0.0, std_r2)

        x_new     = self.x + t_hat * np.cos(self.orientation + r1_hat)
        y_new     = self.y + t_hat * np.sin(self.orientation + r1_hat)
        theta_new = self.orientation + r1_hat + r2_hat

        # Wrap angle to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        self.set(x_new, y_new, theta_new)

    def set_weight(self, weight):
        '''
        set_weights: sets the weight of the particles
        '''
        self.weight = float(weight)


class RobotFunctions:

    def __init__(self, num_particles=0):
        self.particles = []
        if num_particles != 0:
            self.num_particles = num_particles
            self.particles = []
            for _ in range(self.num_particles):
                self.particles.append(particle())

    def get_weights(self,):
        if self.num_particles != 0:
            weights = np.array([p.weight for p in self.particles])
            weights /= np.sum(weights)  # Normalize weights
            return weights
        else:
            return np.array([])

    def get_particle_states(self,):
        if self.num_particles == 0:
            return np.array([])

        samples = np.array([[p.x, p.y, p.orientation] for p in self.particles])
        return samples

    def move_particles(self, deltas):
        for part in self.particles:
            part.move_odom(deltas, [0.2, 0.2, 0.002, 0.002])

    def get_selected_state(self,):
        '''
        Devuelve el estado estimado del robot como la media ponderada de las partículas.
        Usa la media circular para el ángulo para evitar discontinuidades.
        '''
        weights = self.get_weights()

        xs           = np.array([p.x           for p in self.particles])
        ys           = np.array([p.y           for p in self.particles])
        orientations = np.array([p.orientation for p in self.particles])

        x_mean = float(np.sum(weights * xs))
        y_mean = float(np.sum(weights * ys))

        # Circular mean for angle
        cos_mean  = np.sum(weights * np.cos(orientations))
        sin_mean  = np.sum(weights * np.sin(orientations))
        theta_mean = float(np.arctan2(sin_mean, cos_mean))

        return [x_mean, y_mean, theta_mean]


    def update_particles(self, data, map_data, grid):
        '''
        Actualiza los pesos de las partículas usando el modelo de campo de verosimilitud
        (slide 40 del tutorial) y luego realiza el resampleo sistemático.

        Modelo: q = prod_k [ z_hit * p(dist_k, sigma) + z_random / z_max ]
        El campo de verosimilitud ya codifica la Gaussiana de distancia, por lo que
        grid[y,x] / 100 ≈ exp(-dist²/(2σ²)).

        data:     LaserScan – lectura del LIDAR
        map_data: OccupancyGrid – mensaje crudo del mapa de verosimilitud
        grid:     np.ndarray – grilla [y, x], (0,0) = esquina inferior izquierda del mapa
        '''
        resolution = float(map_data.info.resolution)
        origin_x   = map_data.info.origin.position.x
        origin_y   = map_data.info.origin.position.y
        height, width = grid.shape

        # Parámetros del modelo de sensor (z_hit + z_random = 1)
        z_hit    = 0.9
        z_random = 0.1
        z_max    = float(data.range_max)

        # Subsample LIDAR rays for speed (~36 rays out of 360)
        n_rays     = len(data.ranges)
        step       = max(1, n_rays // 36)
        sub_ranges = list(data.ranges)[::step]
        sub_incr   = data.angle_increment * step

        # ---- Compute log-weights for each particle ----
        log_weights = []
        for p in self.particles:
            points = self.scan_refererence(
                sub_ranges,
                data.range_min, data.range_max,
                data.angle_min, data.angle_max,
                sub_incr,
                [p.x, p.y, p.orientation]
            )

            if points.shape[1] == 0:
                log_weights.append(-1e6)
                continue

            # Convert world coordinates to grid indices
            gx = ((points[0] - origin_x) / resolution).astype(int)
            gy = ((points[1] - origin_y) / resolution).astype(int)

            in_bounds = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
            gx = gx[in_bounds]
            gy = gy[in_bounds]

            if len(gx) == 0:
                log_weights.append(-1e6)
                continue

            # Mezcla: z_hit * p_campo + z_random/z_max  (algoritmo slide 40)
            # grid[y,x] ∈ [0,100] → normalizado a [0,1] representa la Gaussiana
            p_campo   = grid[gy, gx].astype(np.float64) / 100.0
            p_per_ray = z_hit * p_campo + z_random / z_max  # piso mínimo garantizado

            # Media del log para estabilidad numérica (equivale al producto normalizado)
            log_weights.append(float(np.mean(np.log(p_per_ray))))

        # ---- Numerically stable conversion to weights ----
        log_arr = np.array(log_weights)
        log_arr -= log_arr.max()          # shift to avoid underflow
        w_arr = np.exp(log_arr)

        for i, p in enumerate(self.particles):
            p.set_weight(float(w_arr[i]))

        # ---- Systematic resampling ----
        weights = self.get_weights()      # normalised
        N = self.num_particles
        positions = (np.arange(N) + np.random.random()) / N
        cumsum    = np.cumsum(weights)
        indices   = np.searchsorted(cumsum, positions)
        indices   = np.clip(indices, 0, N - 1)

        # deepcopy so each particle is an independent object
        self.particles = [copy.deepcopy(self.particles[i]) for i in indices]

        # Resetear pesos a 1.0 post-resampleo: entre actualizaciones de sensor
        # todas las partículas son hipótesis igualmente válidas → promedio simple
        for p in self.particles:
            p.set_weight(1.0)


    def scan_refererence(self, ranges, range_min, range_max, angle_min, angle_max, angle_increment, last_odom):
        '''
        Scan Reference recibe:
            - ranges: lista rangos del escáner láser
            - range_min: rango mínimo del escáner
            - range_max: rango máximo del escáner
            - angle_min: ángulo mínimo del escáner
            - angle_max: ángulo máximo del escáner
            - angle_increment: incremento de ángulo del escáner
            - last_odom: última odometría [tx, ty, theta]
        Devuelve puntos en el mapa transformados a coordenadas globales donde
            - points_map[0]: coordenadas x
            - points_map[1]: coordenadas y
        '''
        tx, ty, theta = last_odom

        n = len(ranges)
        angles    = np.array([angle_min + i * angle_increment for i in range(n)])
        ranges_arr = np.array(ranges, dtype=np.float64)

        # Excluir lecturas inválidas: inf, nan, fuera de rango, y z_max exacto
        # (algoritmo slide 40: "if z_k != z_max")
        valid = (
            np.isfinite(ranges_arr) &
            (ranges_arr >= range_min) &
            (ranges_arr < range_max)
        )
        r = ranges_arr[valid]
        a = angles[valid]

        # Local (robot-frame) Cartesian coordinates
        x_local = r * np.cos(a)
        y_local = r * np.sin(a)

        # Rotate and translate to global (map) frame
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_global = tx + x_local * cos_t - y_local * sin_t
        y_global = ty + x_local * sin_t + y_local * cos_t

        return np.array([x_global, y_global])
