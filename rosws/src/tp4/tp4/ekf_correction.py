#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from custom_msgs.msg import Belief


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class EKFCorrection(Node):
    def __init__(self):
        super().__init__('ekf_correction')

        self.mu = None
        self.covariance = None
        self.landmarks = None

        # Qt_i: measurement noise covariance (range, angle)
        sigma_r = 0.05
        sigma_phi = 0.05
        self.Qt_i = np.diag([sigma_r**2, sigma_phi**2])

        self.belief_sub = self.create_subscription(
            Belief, '/belief', self.belief_callback, 10)
        self.belief_pub = self.create_publisher(Belief, '/belief', 10)
        self.landmarks_sub = self.create_subscription(
            PoseArray, '/landmarks', self.landmarks_callback, 10)
        self.obs_sub = self.create_subscription(
            PoseArray, '/observed_landmarks', self.obs_callback, 10)

    def belief_callback(self, msg):
        self.mu = np.array([msg.mu.x, msg.mu.y, msg.mu.theta])
        self.covariance = np.array(msg.covariance).reshape(3, 3)

    def landmarks_callback(self, msg):
        self.landmarks = msg.poses

    def obs_callback(self, msg):
        if self.mu is None or self.landmarks is None:
            return
        if len(msg.poses) != len(self.landmarks):
            return

        mu = self.mu.copy()
        sigma = self.covariance.copy()
        corrected = False

        for i, obs_pose in enumerate(msg.poses):
            r_obs = obs_pose.position.x
            phi_obs = obs_pose.position.z

            # Landmark not detected when all values are zero
            if r_obs == 0.0 and phi_obs == 0.0:
                continue

            lx = self.landmarks[i].position.x
            ly = self.landmarks[i].position.y

            dx = lx - mu[0]
            dy = ly - mu[1]
            q = dx**2 + dy**2
            r_exp = np.sqrt(q)

            if r_exp < 1e-6:
                continue

            phi_exp = normalize_angle(np.arctan2(dy, dx) - mu[2])

            # Jacobian Ht of measurement model h w.r.t. state (2x3)
            Ht = np.array([
                [-dx / r_exp, -dy / r_exp,  0.0],
                [ dy / q,     -dx / q,     -1.0]
            ])

            # Innovation covariance (2x2)
            St = Ht @ sigma @ Ht.T + self.Qt_i

            # Kalman gain (3x2)
            Kt = sigma @ Ht.T @ np.linalg.inv(St)

            # Innovation with angle normalization
            innovation = np.array([
                r_obs - r_exp,
                normalize_angle(phi_obs - phi_exp)
            ])

            # Innovation gating: reject outliers using Mahalanobis distance
            # chi2(2 DOF) threshold at 99.9% = 13.8; anything beyond this is an outlier
            mahal_sq = float(innovation @ np.linalg.inv(St) @ innovation)
            if mahal_sq > 13.8:
                self.get_logger().debug(
                    f'Landmark {i} gated out: mahal={mahal_sq:.1f}, '
                    f'dr={innovation[0]:.2f} m, dphi={innovation[1]:.2f} rad')
                continue

            mu = mu + Kt @ innovation
            mu[2] = normalize_angle(mu[2])

            # Joseph form: numerically stable, guarantees positive-definite covariance
            I_KH = np.eye(3) - Kt @ Ht
            sigma = I_KH @ sigma @ I_KH.T + Kt @ self.Qt_i @ Kt.T
            corrected = True

        if not corrected:
            return

        self.mu = mu
        self.covariance = sigma

        out = Belief()
        out.mu.x = float(mu[0])
        out.mu.y = float(mu[1])
        out.mu.theta = float(mu[2])
        out.covariance = sigma.flatten().tolist()
        self.belief_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EKFCorrection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
