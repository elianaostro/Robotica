#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from custom_msgs.msg import Belief, DeltaOdom


class EKFPrediction(Node):
    def __init__(self):
        super().__init__('ekf_prediction')

        self.mu = None
        self.covariance = None

        # Qt: motion model noise covariance
        self.Qt = np.diag([0.02, 0.02, 0.02])

        self.belief_sub = self.create_subscription(
            Belief, '/belief', self.belief_callback, 10)
        self.belief_pub = self.create_publisher(Belief, '/belief', 10)
        self.delta_sub = self.create_subscription(
            DeltaOdom, '/delta', self.delta_callback, 10)

    def belief_callback(self, msg):
        self.mu = np.array([msg.mu.x, msg.mu.y, msg.mu.theta])
        self.covariance = np.array(msg.covariance).reshape(3, 3)

    def delta_callback(self, msg):
        if self.mu is None:
            return

        dr1 = msg.dr1
        dt = msg.dt
        dr2 = msg.dr2

        x, y, theta = self.mu
        theta_dr1 = theta + dr1

        # Predicted state via motion model g(mu, u)
        mu_bar = np.array([
            x + dt * np.cos(theta_dr1),
            y + dt * np.sin(theta_dr1),
            theta + dr1 + dr2
        ])
        mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))

        # Jacobian Gt of g w.r.t. state (without noise)
        Gt = np.array([
            [1, 0, -dt * np.sin(theta_dr1)],
            [0, 1,  dt * np.cos(theta_dr1)],
            [0, 0,  1]
        ])

        # Predicted covariance
        sigma_bar = Gt @ self.covariance @ Gt.T + self.Qt

        self.mu = mu_bar
        self.covariance = sigma_bar

        out = Belief()
        out.mu.x = float(mu_bar[0])
        out.mu.y = float(mu_bar[1])
        out.mu.theta = float(mu_bar[2])
        out.covariance = sigma_bar.flatten().tolist()
        self.belief_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EKFPrediction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
