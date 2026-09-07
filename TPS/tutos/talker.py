#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TalkerNode(Node):
   def __init__(self):
       super().__init__('talker')
       self.publisher = self.create_publisher(String, 'chatter', 10)
       self.timer = self.create_timer(1, self.timer_callback)  # 1s
       self.get_logger().info("I will publish to the topic 'chatter'")

   def timer_callback(self):
       msg = String()
       msg.data = f'hello world {self.get_clock().now().to_msg().sec}'
       self.get_logger().info(msg.data)
       self.publisher.publish(msg)

def main():
   rclpy.init()
   node = TalkerNode()
   try:
       rclpy.spin(node)
   except KeyboardInterrupt:
       pass
   finally:
       node.destroy_node()
       rclpy.shutdown()

if __name__ == '__main__':
   main()
