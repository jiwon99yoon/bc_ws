import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys


class ImageSubscriber(Node):
    def __init__(self, camera_name: str):
        super().__init__('image_subscriber')

        self.camera_name = camera_name
        topic_name = f'mujoco_sensor/{self.camera_name}'

        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.callback,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info(f"Subscribed to {topic_name}")

    def callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow(f'Image from {self.camera_name}', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Usage: ros2 run <your_package> <your_executable> <camera_name>")
        return

    camera_name = sys.argv[1]
    node = ImageSubscriber(camera_name)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
