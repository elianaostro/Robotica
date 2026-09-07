#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ListenerNode(Node):
   def __init__(self):
       super().__init__('listener')
       self.subscription = self.create_subscription(
           String,
           'chatter',
           self.callback,
           10  # QoS profile: queue size
       )
       self.subscription  # Prevent unused variable warning

   def callback(self, msg):
       self.get_logger().info(f"I heard: {msg.data}")

def main():
   rclpy.init()
   node = ListenerNode()
   try:
       rclpy.spin(node)
   except KeyboardInterrupt:
       pass
   finally:
       node.destroy_node()
       rclpy.shutdown()

if __name__ == '__main__':
   main()
