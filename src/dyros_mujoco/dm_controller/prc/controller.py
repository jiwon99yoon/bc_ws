from dm_controller_wrapper_cpp import Controller as Fr3Controllercpp
from dm_msgs.srv import TaskMove, JointMove
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

class JointMoveServer(Node):
    def __init__(self):
        super().__init__("joint_move_server")
        self.srv = self.create_service(JointMove, '/joint_move_srv', self.goal_callback)

        self.goal_received = False

        self.joint = None
        self.duration = None

    def goal_callback(self, request, response):
        print("Joint Move Goal has been arrived")

        goal = request.joint

        self.x = np.array(goal.position)

        self.duration = request.duration

        response.is_received = True

        self.goal_received = True
        return response

    def goal_state(self):
        return self.goal_received
    
    def goal_reset(self):
        self.goal_received = False
        self.x = None
        self.duration = None

class TaskMoveServer(Node):
    def __init__(self):
        super().__init__("task_move_server")
        self.srv = self.create_service(TaskMove, '/task_move_srv', self.goal_callback)

        self.goal_received = False

        self.x = None
        self.rot = None # 3x3 rotation matrix
        self.duration = None

    def goal_callback(self, request, response):
        print("Task Move Goal has been arrived")

        goal = Pose()
        goal = request.pose

        self.x = np.array([goal.position.x, goal.position.y, goal.position.z])

        quat = [goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w]
        self.rot = R.from_quat(quat).as_matrix()

        self.duration = request.duration

        response.is_received = True

        self.goal_received = True
        return response

    def goal_state(self):
        return self.goal_received
    
    def goal_reset(self):
        self.goal_received = False
        self.x = None
        self.rot = None # 3x3 rotation matrix
        self.duration = None

class Fr3Controller(Node):
    def __init__(self, urdf_path):
        super().__init__("robot_control_node")

        self.controller = Fr3Controllercpp(urdf_path) # rc = robot controller
        self.dof = 7 + 2 # include gripper state
        self.ready_duration = 2.0
        self.t = 0.0
        self.t_stamp = 0.0
        self.hz = 1000 

        self.is_ready = False

        self.tm = TaskMoveServer()
        self.jm = JointMoveServer()

    def updateModel(self, data, time):

        q = data.qpos[:self.dof]
        qd = data.qvel[:self.dof]      
        # print(data.sensordata)

        n_joint = 7
        tau = data.sensordata[2:n_joint*3:3] # torque = 3-axes data

        ft = data.sensordata[-6:]

        self.t = time / self.hz        
        self.controller.updateModel(q, qd, tau, ft, time)
        
    def setIdleConfig(self, time):
        tau_d = self.controller.setIdleConfig(time)
        return tau_d
    
    def gripperOpen(self, target_width, time):
        gf = self.controller.gripperOpen(target_width, time)
        return np.array([gf[0]])
    
    def quat2rot(self, quat):
        rotation = R.from_quat(quat)  # [x, y, z, w]
        rot_matrix = rotation.as_matrix()  # Convert to 3x3 rotation matrix
        return rot_matrix
        
    def setReady(self):
        if self.t >= self.ready_duration:            
            self.is_ready = True
            # self.setTimeStamp()
            self.controller.updateInitialValues()
        else:
            self.is_ready = False

    def setTimeStamp(self):
        self.t_stamp = self.t

    def compute(self):

        # if not self.th.is_running(self.t >= self.ready_duration):
        if not self.is_ready:
            time = np.array((0.0, self.ready_duration))
            tau_d = self.setIdleConfig(time)
            gf = self.gripperOpen(0.04, np.array([0.0, 2.0], dtype=np.float64))
            ctrl = np.concatenate((tau_d, gf), axis=0)

            self.setReady()

        else:        

            if self.tm.goal_state():
                time = np.array((self.t_stamp, self.t_stamp+self.tm.duration))
                tau_d = self.controller.taskMove(self.tm.x, self.tm.rot, time)
                gf = self.gripperOpen(0.04, np.array([0.0, 2.0], dtype=np.float64))

                ctrl = np.concatenate((tau_d, gf), axis=0)

                if self.t - self.t_stamp >= self.tm.duration:
                    self.controller.updateInitialValues()
                    self.tm.goal_reset()

            elif self.jm.goal_state():
                time = np.array((self.t_stamp, self.t_stamp+self.jm.duration))
                tau_d = self.controller.jointMove(self.jm.x, time)
                gf = self.gripperOpen(0.04, np.array([0.0, 2.0], dtype=np.float64))

                ctrl = np.concatenate((tau_d, gf), axis=0)

                if self.t - self.t_stamp >= self.jm.duration:
                    self.controller.updateInitialValues()
                    self.jm.goal_reset()

            else:
                ctrl = self.controller.initState()
                self.setTimeStamp()
            
        return ctrl
    
