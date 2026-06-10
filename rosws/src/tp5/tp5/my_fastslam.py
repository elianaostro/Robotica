#!/usr/bin/env python3
"""FastSLAM 1.0 basado en landmarks (TP5).

Cada partícula mantiene su propia hipótesis de la trayectoria del robot y un
mapa EKF independiente por landmark (filtro de Kalman extendido 2x2 sobre la
posición [x, y] de cada landmark). El problema de asociación de datos está
resuelto: el índice de cada pose dentro de /observed_landmarks es el ID del
landmark.

Entradas:
  /delta              custom_msgs/DeltaOdom  -> dr1, dt, dr2 (modelo de odometría)
  /observed_landmarks geometry_msgs/PoseArray-> por landmark: x=rango, z=ángulo
                                                (todo en cero => no observado)

Salidas:
  /fastslam_pose      geometry_msgs/PoseStamped  pose estimada (mejor partícula)
  /fastslam_path      nav_msgs/Path              trayectoria estimada
  /fastslam_landmarks visualization_msgs/MarkerArray  landmarks + covarianzas
"""

import copy
import math

import numpy as np
import rclpy
from rclpy.node import Node

from custom_msgs.msg import DeltaOdom
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray


def normalize_angle(angle):
    """Lleva un ángulo al rango (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_from_yaw(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class Particle:
    """Una hipótesis: pose del robot + mapa EKF de landmarks."""

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0
        # id_landmark -> [mu (2,), sigma (2x2)]
        self.landmarks = {}

    def clone(self):
        """Copia profunda (los mapas no deben compartir objetos entre partículas)."""
        p = Particle(self.x, self.y, self.theta)
        p.weight = self.weight
        p.landmarks = {
            lm_id: [mu.copy(), sigma.copy()]
            for lm_id, (mu, sigma) in self.landmarks.items()
        }
        return p


class FastSlamNode(Node):
    def __init__(self):
        super().__init__("fastslam")

        # --- Parámetros ---
        self.num_particles = self.declare_parameter("num_particles", 100).value

        # Ruido de medición (rango, ángulo) -> desvío 0.05 por consigna
        sigma_r = 0.05
        sigma_phi = 0.05
        self.Qt = np.diag([sigma_r ** 2, sigma_phi ** 2])

        # Ruido del modelo de movimiento (odometría). Cada partícula muestrea
        # deltas perturbados con estos desvíos para que el filtro diverja.
        self.alpha1 = 0.01   # rot por rot
        self.alpha2 = 0.01   # rot por trans
        self.alpha3 = 0.02   # trans por trans
        self.alpha4 = 0.01   # trans por rot

        # --- Partículas ---
        self.particles = [Particle() for _ in range(self.num_particles)]

        # --- Pub / Sub ---
        self.create_subscription(DeltaOdom, "/delta", self.delta_callback, 10)
        self.create_subscription(
            PoseArray, "/observed_landmarks", self.obs_callback, 10
        )

        self.pose_pub = self.create_publisher(PoseStamped, "/fastslam_pose", 10)
        self.path_pub = self.create_publisher(Path, "/fastslam_path", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/fastslam_landmarks", 10
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.get_logger().info(
            f"FastSLAM iniciado con {self.num_particles} partículas"
        )

    # ------------------------------------------------------------------ #
    # Paso de predicción: muestreo del modelo de movimiento por partícula  #
    # ------------------------------------------------------------------ #
    def delta_callback(self, msg: DeltaOdom):
        dr1 = msg.dr1
        dt = msg.dt
        dr2 = msg.dr2

        # Desvíos del ruido de muestreo (modelo de odometría de Thrun)
        std_r1 = math.sqrt(self.alpha1 * dr1 ** 2 + self.alpha2 * dt ** 2)
        std_t = math.sqrt(self.alpha3 * dt ** 2 + self.alpha4 * (dr1 ** 2 + dr2 ** 2))
        std_r2 = math.sqrt(self.alpha1 * dr2 ** 2 + self.alpha2 * dt ** 2)

        for p in self.particles:
            # Muestra deltas perturbados de forma independiente por partícula
            sr1 = dr1 - np.random.normal(0.0, std_r1) if std_r1 > 0 else dr1
            st = dt - np.random.normal(0.0, std_t) if std_t > 0 else dt
            sr2 = dr2 - np.random.normal(0.0, std_r2) if std_r2 > 0 else dr2

            p.x += st * math.cos(p.theta + sr1)
            p.y += st * math.sin(p.theta + sr1)
            p.theta = normalize_angle(p.theta + sr1 + sr2)

    # ------------------------------------------------------------------ #
    # Paso de corrección: actualización EKF de landmarks + peso            #
    # ------------------------------------------------------------------ #
    def obs_callback(self, msg: PoseArray):
        any_observation = False

        for p in self.particles:
            for lm_id, pose in enumerate(msg.poses):
                r = pose.position.x
                phi = pose.position.z

                # Landmark no observado en este instante
                if r == 0.0 and phi == 0.0:
                    continue
                any_observation = True

                if lm_id not in p.landmarks:
                    self._init_landmark(p, lm_id, r, phi)
                else:
                    self._update_landmark(p, lm_id, r, phi)

        if not any_observation:
            return

        self._normalize_weights()
        self._resample()
        self._publish(msg.header.stamp)

    def _init_landmark(self, particle, lm_id, r, phi):
        """Inicializa un nuevo landmark con el modelo inverso de medición."""
        bearing = particle.theta + phi
        mu = np.array([
            particle.x + r * math.cos(bearing),
            particle.y + r * math.sin(bearing),
        ])

        # Jacobiano del modelo de medición respecto del landmark, evaluado
        # en mu. Sigma inicial = H^-1 Q H^-T (propagación de la incertidumbre
        # de la medición al espacio del landmark).
        H = self._measurement_jacobian(particle, mu)
        H_inv = np.linalg.inv(H)
        sigma = H_inv @ self.Qt @ H_inv.T

        particle.landmarks[lm_id] = [mu, sigma]
        # El peso no se modifica en la primera observación.

    def _update_landmark(self, particle, lm_id, r, phi):
        """Actualización EKF del landmark y factor de importancia (peso)."""
        mu, sigma = particle.landmarks[lm_id]

        dx = mu[0] - particle.x
        dy = mu[1] - particle.y
        q = dx ** 2 + dy ** 2
        r_exp = math.sqrt(q)
        if r_exp < 1e-6:
            return
        phi_exp = normalize_angle(math.atan2(dy, dx) - particle.theta)

        # Jacobiano 2x2 respecto del landmark
        H = np.array([
            [dx / r_exp, dy / r_exp],
            [-dy / q,    dx / q],
        ])

        # Covarianza de la innovación
        S = H @ sigma @ H.T + self.Qt
        S_inv = np.linalg.inv(S)

        # Ganancia de Kalman (2x2)
        K = sigma @ H.T @ S_inv

        # Innovación (con normalización del ángulo)
        innovation = np.array([
            r - r_exp,
            normalize_angle(phi - phi_exp),
        ])

        # Actualización de media y covarianza del landmark
        mu_new = mu + K @ innovation
        sigma_new = (np.eye(2) - K @ H) @ sigma

        particle.landmarks[lm_id] = [mu_new, sigma_new]

        # Factor de importancia: verosimilitud gaussiana de la medición
        det = np.linalg.det(2.0 * math.pi * S)
        if det <= 0:
            return
        likelihood = (
            1.0 / math.sqrt(det)
            * math.exp(-0.5 * float(innovation @ S_inv @ innovation))
        )
        particle.weight *= max(likelihood, 1e-300)

    def _measurement_jacobian(self, particle, mu):
        dx = mu[0] - particle.x
        dy = mu[1] - particle.y
        q = dx ** 2 + dy ** 2
        r_exp = math.sqrt(q)
        return np.array([
            [dx / r_exp, dy / r_exp],
            [-dy / q,    dx / q],
        ])

    # ------------------------------------------------------------------ #
    # Normalización y remuestreo                                          #
    # ------------------------------------------------------------------ #
    def _normalize_weights(self):
        total = sum(p.weight for p in self.particles)
        if total <= 0:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles
            return
        for p in self.particles:
            p.weight /= total

    def _effective_sample_size(self):
        return 1.0 / sum(p.weight ** 2 for p in self.particles)

    def _resample(self):
        # Remuestreo solo si el filtro degeneró (ESS < N/2)
        if self._effective_sample_size() >= self.num_particles / 2.0:
            return

        # Remuestreo de baja varianza (systematic resampling)
        N = self.num_particles
        weights = np.array([p.weight for p in self.particles])
        positions = (np.arange(N) + np.random.uniform()) / N
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0  # robustez numérica

        new_particles = []
        i = 0
        for pos in positions:
            while pos > cumulative[i]:
                i += 1
            new_particles.append(self.particles[i].clone())

        self.particles = new_particles
        for p in self.particles:
            p.weight = 1.0 / N

    # ------------------------------------------------------------------ #
    # Publicación de la mejor partícula                                   #
    # ------------------------------------------------------------------ #
    def _best_particle(self):
        return max(self.particles, key=lambda p: p.weight)

    def _publish(self, stamp):
        best = self._best_particle()

        # --- Pose estimada ---
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = float(best.x)
        pose_msg.pose.position.y = float(best.y)
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = quaternion_from_yaw(best.theta)
        self.pose_pub.publish(pose_msg)

        # --- Trayectoria ---
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

        # --- Landmarks de la mejor partícula ---
        ma = MarkerArray()
        for lm_id, (lm_mu, lm_sigma) in best.landmarks.items():
            ma.markers.append(
                self.make_landmark_marker(lm_id * 2, lm_mu[0], lm_mu[1])
            )
            ma.markers.append(
                self.make_covariance_marker(
                    lm_id * 2 + 1, lm_mu[0], lm_mu[1], lm_sigma
                )
            )
        self.marker_pub.publish(ma)

    # ------------------------------------------------------------------ #
    # Markers (provistos por la consigna)                                 #
    # ------------------------------------------------------------------ #
    def make_landmark_marker(self, idx, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = int(idx)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m

    def make_covariance_marker(self, idx, x, y, cov):
        # Parámetros de la elipse a partir de los autovalores de la covarianza
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = math.atan2(vecs[1, 0], vecs[0, 0])
        scale_x = 30 * 2 * math.sqrt(max(vals[0], 0.0))
        scale_y = 30 * 2 * math.sqrt(max(vals[1], 0.0))

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = int(idx)
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation = quaternion_from_yaw(angle)
        m.scale.x = max(scale_x, 1e-3)
        m.scale.y = max(scale_y, 1e-3)
        m.scale.z = 0.01
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 0.3
        return m


def main(args=None):
    rclpy.init(args=args)
    node = FastSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nCerrando FastSLAM...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
