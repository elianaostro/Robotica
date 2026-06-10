#!/usr/bin/env python3
"""Conversor de odometría a deltas (modelo de odometría).

La consigna asume que la simulación publica /delta directamente. En este
workspace la simulación publica /calc_odom (odometría con ruido), así que este
nodo auxiliar la convierte a custom_msgs/DeltaOdom (dr1, dt, dr2) para poder
probar el FastSLAM localmente. Si el evaluador corre una simulación que ya
publica /delta, este nodo no es necesario.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node

from custom_msgs.msg import DeltaOdom
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from scipy.spatial.transform import Rotation as R


class OdomToDelta(Node):
    def __init__(self):
        super().__init__("odom_to_delta")

        self.create_subscription(Odometry, "/calc_odom", self.odom_callback, 10)
        self.create_subscription(Odometry, "/odom", self.real_odom_callback, 10)

        self.delta_pub = self.create_publisher(DeltaOdom, "/delta", 10)

        # Trayectorias para comparar en RViz
        self.real_path_pub = self.create_publisher(Path, "/real_robot_path", 10)
        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"

        self.last_odom = (0.0, 0.0, 0.0)
        self.read_odom = False

    def real_odom_callback(self, data: Odometry):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = data.pose.pose
        self.real_path_msg.poses.append(ps)
        self.real_path_msg.header.stamp = ps.header.stamp
        self.real_path_pub.publish(self.real_path_msg)

    def odom_callback(self, data: Odometry):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        q = data.pose.pose.orientation
        theta = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=False)[2]

        if self.read_odom:
            dx = x - self.last_odom[0]
            dy = y - self.last_odom[1]
            delta_t = math.sqrt(dx ** 2 + dy ** 2)

            if delta_t > 1e-6:
                delta_rot1 = math.atan2(dy, dx) - self.last_odom[2]
                delta_rot2 = theta - self.last_odom[2] - delta_rot1
            else:
                delta_rot1 = 0.0
                delta_rot2 = theta - self.last_odom[2]

            delta_rot1 = math.atan2(math.sin(delta_rot1), math.cos(delta_rot1))
            delta_rot2 = math.atan2(math.sin(delta_rot2), math.cos(delta_rot2))

            # Solo publicar cuando hubo movimiento real
            if delta_t > 1e-6 or abs(delta_rot2) > 1e-6:
                msg = DeltaOdom()
                msg.dr1 = float(delta_rot1)
                msg.dr2 = float(delta_rot2)
                msg.dt = float(delta_t)
                self.delta_pub.publish(msg)

        self.last_odom = (x, y, theta)
        self.read_odom = True


def main(args=None):
    rclpy.init(args=args)
    node = OdomToDelta()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
