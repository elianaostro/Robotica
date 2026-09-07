import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class TurtlebotFSM(Node):
    def __init__(self):
        super().__init__('turtlebot_fsm_node')
        
        self.declare_parameter('robot_type', 'turtlebot3')
        self.robot_type = self.get_parameter('robot_type').get_parameter_value().string_value

        self.STATE_FORWARD = 0
        self.STATE_STOP = 1
        self.STATE_ROTATE = 2
        self.state = self.STATE_FORWARD

        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.initial_rotation_yaw = 0.0
        self.obstacle_detected = False

        self.linear_speed = 0.5
        self.angular_speed = 1.0
        self.rotation_angle = math.radians(110.0)

        self.setup_robot_config()

        self.cmd_vel_pub = self.create_publisher(Twist, f'{self.ns_prefix}/cmd_vel', 10)
        
        self.scan_sub = self.create_subscription(LaserScan, f'{self.ns_prefix}/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, f'{self.ns_prefix}/{self.odom_topic}', self.odom_callback, self.odom_qos)

        self.create_timer(0.1, self.main_loop)

    def setup_robot_config(self):
        if self.robot_type == 'turtlebot4':
            self.ns_prefix = '/tb4_0'
            self.odom_topic = 'odom'
            self.odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.inertia_compensation = math.radians(20.5)
        else:
            self.ns_prefix = ''
            self.odom_topic = 'calc_odom'
            self.odom_qos = QoSProfile(depth=10)
            self.inertia_compensation = 0.0

    def odom_callback(self, msg):
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        ranges = msg.ranges
        intensities = msg.intensities if len(msg.intensities) > 0 else None
        min_distance = float('inf')

        for i, r in enumerate(ranges):
            if math.isnan(r) or math.isinf(r) or r < msg.range_min:
                continue
            if self.robot_type == 'turtlebot4' and intensities is not None and intensities[i] == 0.0:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            if self.robot_type == 'turtlebot4':
                angle += math.pi / 2
            angle = math.atan2(math.sin(angle), math.cos(angle))

            if abs(angle) < math.radians(30.0):
                if r < min_distance:
                    min_distance = r

        self.obstacle_detected = min_distance <= 0.5

    def move_forward(self):
        twist = Twist()
        twist.linear.x = self.linear_speed
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def rotate_robot(self):
        twist = Twist()
        twist.angular.z = self.angular_speed
        self.cmd_vel_pub.publish(twist)

    def main_loop(self):
        if self.state == self.STATE_FORWARD:
            if self.obstacle_detected:
                self.initial_rotation_yaw = self.current_yaw
                adjusted_rotation = self.rotation_angle - self.inertia_compensation
                self.target_yaw = self.initial_rotation_yaw + adjusted_rotation
                self.target_yaw = math.atan2(math.sin(self.target_yaw), math.cos(self.target_yaw))
                self.state = self.STATE_STOP

        elif self.state == self.STATE_STOP:
            self.state = self.STATE_ROTATE

        elif self.state == self.STATE_ROTATE:
            error = self.target_yaw - self.current_yaw
            error = math.atan2(math.sin(error), math.cos(error))
            if error <= 0:
                self.state = self.STATE_FORWARD

        if self.state == self.STATE_FORWARD:
            self.move_forward()
        elif self.state == self.STATE_STOP:
            self.stop_robot()
        elif self.state == self.STATE_ROTATE:
            self.rotate_robot()

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()