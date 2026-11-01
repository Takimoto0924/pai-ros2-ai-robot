
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


TOPIC = "/model/vehicle_blue/cmd_vel"


class RobotController(Node):
    def __init__(self):
        super().__init__("robot_controller")
        self.publisher = self.create_publisher(
            Twist, TOPIC, 10
        )

    def wait_for_bridge(self):
        for _ in range(50):
            if self.publisher.get_subscription_count() > 0:
                return True
            time.sleep(0.2)
        return False

    def move(self, linear, angular, duration):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)

        end_time = time.monotonic() + duration

        while time.monotonic() < end_time:
            self.publisher.publish(msg)
            time.sleep(0.1)

    def stop(self):
        self.move(0.0, 0.0, 1.0)


def main():
    rclpy.init()
    robot = RobotController()

    try:
        if not robot.wait_for_bridge():
            raise RuntimeError("Bridgeに接続できません")

        print("前進")
        robot.move(0.3, 0.0, 3.0)
        robot.stop()

        print("左旋回")
        robot.move(0.0, 0.6, 2.0)
        robot.stop()

        print("前進")
        robot.move(0.3, 0.0, 3.0)
        robot.stop()

        print("右旋回")
        robot.move(0.0, -0.6, 2.0)
        robot.stop()

        print("走行完了")

    except KeyboardInterrupt:
        print("中断しました")

    finally:
        robot.stop()
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
