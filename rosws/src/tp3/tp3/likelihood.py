import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np


class LikelihoodMapPublisher(Node):
    def __init__(self):
        super().__init__('likelihood_map_publisher')
        qos = rclpy.qos.QoSProfile(depth=1)
        qos.durability = rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.pub = self.create_publisher(OccupancyGrid, '/likelihood_map', qos)
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos
        )
    
    def map_callback(self, msg):
        prob_msg = OccupancyGrid()
        prob_msg.header = msg.header
        prob_msg.info = msg.info


        # TODO: Completar el codigo para que el mensaje contenga la data del mapa de probabilidades
        # prob_msg.data = mapa de probabilidades
        # ROS espera que la data sea un array unidimensional de enteros de 8 bits con valores entre 0 y 100
        # msg.data contiene la data del mapa de ocupación como un array de enteros de 8 bits con 0 para celdas libres
        # 100 para celdas ocupadas y -1 para celdas desconocidas
        w = int(msg.info.width)
        h = int(msg.info.height)

        if w <= 0 or h <= 0 or len(msg.data) != w * h:
            self.get_logger().warning("Invalid map dimensions; publishing zeros.")
            prob_msg.data = np.zeros(len(msg.data), dtype=np.int8).tolist()
            self.pub.publish(prob_msg)
            return

        occ = np.array(msg.data, dtype=np.int16).reshape((h, w))
        occupied = (occ == 100)

        # Approx. Euclidean distance transform (chamfer 3-4 mask) in meters.
        if not np.any(occupied):
            prob = np.zeros((h, w), dtype=np.int16)
        else:
            res = float(msg.info.resolution) if msg.info.resolution > 0.0 else 1.0
            dist = np.full((h, w), np.inf, dtype=np.float32)
            dist[occupied] = 0.0

            # Forward pass
            diag = np.float32(np.sqrt(2.0))
            for y in range(h):
                for x in range(w):
                    d = dist[y, x]
                    if y > 0:
                        d = min(d, dist[y - 1, x] + 1.0)
                        if x > 0:
                            d = min(d, dist[y - 1, x - 1] + diag)
                        if x + 1 < w:
                            d = min(d, dist[y - 1, x + 1] + diag)
                    if x > 0:
                        d = min(d, dist[y, x - 1] + 1.0)
                    dist[y, x] = d

            # Backward pass
            for y in range(h - 1, -1, -1):
                for x in range(w - 1, -1, -1):
                    d = dist[y, x]
                    if y + 1 < h:
                        d = min(d, dist[y + 1, x] + 1.0)
                        if x > 0:
                            d = min(d, dist[y + 1, x - 1] + diag)
                        if x + 1 < w:
                            d = min(d, dist[y + 1, x + 1] + diag)
                    if x + 1 < w:
                        d = min(d, dist[y, x + 1] + 1.0)
                    dist[y, x] = d

            dist_m = dist * res

            # Likelihood-field model: p(z|x,m) ~ exp(-d^2 / (2*sigma^2))
            sigma = 0.20  # meters
            prob_f = 100.0 * np.exp(-(dist_m ** 2) / (2.0 * sigma ** 2))
            prob_f[occupied] = 100.0  # keep obstacles fully bright
            prob = np.clip(np.rint(prob_f), 0, 100).astype(np.int16)

        prob_msg.data = prob.astype(np.int8).reshape(-1).tolist()




        self.pub.publish(prob_msg)
        self.get_logger().info("Published likelihood map")

def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
