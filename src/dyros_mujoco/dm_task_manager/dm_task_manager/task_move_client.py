from dm_msgs.srv import TaskMove
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node


class TaskMoveClient(Node):
    def __init__(self):
        super().__init__('task_move_client')

        self.cli = self.create_client(TaskMove, '/task_move_srv')  # 서비스 이름
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_pose(self, pose, t):
        req = TaskMove.Request()
        req.pose = pose                 
        req.duration = t
        self.future = self.cli.call_async(req)
        

def main(args=None):
    rclpy.init(args=args)
    
    print("Enter position (x y z):")
    pos = list(map(float, input().split()))
    
    print("Enter orientation (x y z w):")
    ori = list(map(float, input().split()))
    
    print("Enter duration:")
    t = float(input())  # 리스트가 아닌 하나의 값으로 입력받음

    print("Complete to define the goal pose")

    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    
    pose.orientation.x = ori[0]
    pose.orientation.y = ori[1]
    pose.orientation.z = ori[2]
    pose.orientation.w = ori[3]

    client = TaskMoveClient()
    client.send_pose(pose, t)
    
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