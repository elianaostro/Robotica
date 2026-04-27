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
        prob_msg.data = np.zeros(len(msg.data), dtype=np.int8).tolist()




        self.pub.publish(prob_msg)
        self.get_logger().info("Published likelihood map")

def main(args=None):
    rclpy.init(args=args)
    node = LikelihoodMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
