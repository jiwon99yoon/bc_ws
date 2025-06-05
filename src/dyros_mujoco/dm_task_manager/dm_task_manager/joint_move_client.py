from dm_msgs.srv import JointMove
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
import numpy as np

class JointMoveClient(Node):
    def __init__(self):
        super().__init__('joint_move_client')

        self.cli = self.create_client(JointMove, '/joint_move_srv')  # 서비스 이름
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_joint(self, joint, t):
        req = JointMove.Request()
        req.joint = joint    
        req.duration = t
        self.future = self.cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    
    print("Enter joint angle (degree):")
    pos = list(map(float, input().split()))
    pos = np.array(pos)*np.pi/180 # deg2rad

    print("Enter duration:")
    t = float(input())  # 리스트가 아닌 하나의 값으로 입력받음

    print("Complete to define the goal pose")

    joint = JointState()
    joint.position = pos.tolist()

    client = JointMoveClient()
    client.send_joint(joint, t)
    
    rclpy.spin_until_future_complete(client, client.future)

    if client.future.result() is not None:
        if client.future.result().is_received:
            client.get_logger().info('Pose was successfully received by server!')
        else:
            client.get_logger().warning('Pose was not received correctly.')
    else:
        client.get_logger().error('Service call failed')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()