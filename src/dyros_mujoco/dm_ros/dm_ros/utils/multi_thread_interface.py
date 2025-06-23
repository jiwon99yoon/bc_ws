import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer as mj_view
import threading
import time
from .scene_monitor import SceneMonitor
from .image_publisher import MujocoCameraBridge
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState          # 추가!

class MujocoROSBridge(Node):
    def __init__(self, robot_info, camera_info, robot_controller):
        super().__init__('mujoco_ros_bridge')

        # robot_info = [xml, urdf, hz]
        self.xml_path = robot_info[0]
        self.urdf_path = robot_info[1]
        self.ctrl_freq = robot_info[2]

        # camera_info = [name, width, height, fps]
        self.camera_name = camera_info[0]
        self.width = camera_info[1]
        self.height = camera_info[2]
        self.fps = camera_info[3]
          
        self.rc = robot_controller

        # Mujoco 모델 로드
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = 1 / self.ctrl_freq
        self.model.opt.timestep = self.dt
       
        self.sm = SceneMonitor(self.model, self.data)
        self.hand_eye = MujocoCameraBridge(self.model, camera_info)
      
        self.ctrl_dof = 8 # 7 + 1 
        self.ctrl_step = 0
    
        self.running = True
        self.lock = threading.Lock()
        self.robot_thread = threading.Thread(target=self.robot_control, daemon=True)
        self.hand_eye_thread = threading.Thread(target=self.hand_eye_control, daemon=True)
        self.ros_thread = threading.Thread(target=self.ros_control, daemon=True)

        # <----------------- 아래 부분 추가 ------------------------>
        # ROS2 Publisher 설정 - MoveIt으로 MuJoCo qpos 보내기 
        self.joint_state_to_moveit = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_timer = self.create_timer(self.dt, self.joint_publish_callback)   

        # MoveIt의 topic subscriber에 맞도록 MuJoCo qpos 이름 mapping하기 위한 변수
        self.moveit_joint_names = []
        self.moveit_joint_names.append("panda_finger_joint1")
        for i in range(1, 8):
            self.moveit_joint_names.append(f"panda_joint{i}")

        # ROS2 Subscriber 설정 - MoveIt이 만든 Trajectory 받기
        self.latest_joint_set = None
        self.joint_set_mutex = threading.Lock()
        self.joint_set_sub = self.create_subscription(JointState, '/panda/joint_set', self.joint_set_callback, 10)
        # <----------------- 윗 부분 추가 ------------------------>

    # <----------------- 아래 부분 추가 ------------------------>
    # ROS2 Publisher Callback함수
    def joint_publish_callback(self):
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = self.moveit_joint_names.copy()    #MoveIt이 받는 이름 맞게끔 설정
        js_msg.position = [0.0] + [float(self.data.qpos[i]) for i in range(self.ctrl_dof-1)]
        self.joint_state_to_moveit.publish(js_msg)

    # ROS2 Subscriber Callback함수 - "/panda/joint_set" 토픽에서 moveit이 보낸 JointState 메시지를 받는 콜백 함수
    def joint_set_callback(self, msg):
        # moveit에서 보낸 joint_set 메세지를 그대로 저장 - 일단 저장해둔 뒤, robot control에서 덮어씌우기
        with self.joint_set_mutex: self.latest_joint_set = msg
    
    # <----------------- 윗 부분 추가 ------------------------>

    # visualize thread = main thread
    def run(self):        
        scene_update_freq = 30
        try:     
            with mj_view.launch_passive(self.model, self.data) as viewer:            
                # self.sm.getAllObject()        
                # self.sm.getTargetObject()       
                # self.sm.getSensor() 
                self.robot_thread.start()    
                self.hand_eye_thread.start()
                self.ros_thread.start()


                while self.running and viewer.is_running():   
                    start_time = time.perf_counter()       

                    with self.lock:                        
                        viewer.sync()  # 화면 업데이트          

                    self.time_sync(1/scene_update_freq, start_time, False)
                   
        except KeyboardInterrupt:
            print("\nSimulation interrupted. Closing viewer...")
            self.running = False
            self.robot_thread.join()
            self.hand_eye_thread.join()
            self.ros_thread.join()
            self.sm.destroy_node()

    def robot_control(self):
        self.ctrl_step = 0

        try:
            while rclpy.ok() and self.running:            
                with self.lock:
                    start_time = time.perf_counter()                        

                    # <----------------- 아래 부분 추가 ------------------------>
                    # moveit이 보낸 /panda/joint_set 메세지 존재 여부 검사 - 있으면 그 값을 qpos로 덮어쓰기
                    target_js = None
                    with self.joint_set_mutex:
                        if self.latest_joint_set is not None:                        
                            target_js = self.latest_joint_set

                    if target_js is not None:
                        # moveit이 보낸 궤적 가지고 target_js.position에 들어있는 값을 qpos로 덮어쓰기
                        for i in range(self.ctrl_dof):
                            try:
                                self.data.qpos[i] = float(target_js.position[i])
                            except:
                                pass
                        self.data.qpos[7] = 0.04
                        self.data.qpos[8] = 0.04
                        # mujoco 시뮬레이션 한 스텝
                        mujoco.mj_step(self.model, self.data)  # 시뮬레이션 실행
                    
                    # <----------------- 윗 부분 추가 ------------------------>
                    else:   # moveit이 보낸 궤적이 없으면, 로봇 컨트롤러에서 계산한 값을 qpos로 덮어쓰기
                        mujoco.mj_step(self.model, self.data)  # 시뮬레이션 실행
                        self.rc.updateModel(self.data, self.ctrl_step)  #시뮬레이터 내부 상태를 DMController에 업데이트                 
                        self.data.ctrl[:self.ctrl_dof] = self.rc.compute() #DMController에서 계산한 제어값을 qpos로 덮어쓰기

                    self.ctrl_step += 1
                    
                self.time_sync(self.dt, start_time, False)
            
        except KeyboardInterrupt:
            self.get_logger().into("\nSimulation interrupted. Closing robot controller ...")
            self.rc.destroy_node()

    def hand_eye_control(self):
        renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        hand_eye_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

        while rclpy.ok() and self.running:            
            with self.lock:
                start_time = time.perf_counter()  
                renderer.update_scene(self.data, camera=hand_eye_id)
                self.hand_eye.getImage(renderer.render(), self.ctrl_step)     

            self.time_sync(1/self.fps, start_time, False)
        self.hand_eye.destroy_node()

    def time_sync(self, target_dt, t_0, verbose=False):
        elapsed_time = time.perf_counter() - t_0
        sleep_time = target_dt - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if verbose:
            print(f'Time {elapsed_time*1000:.4f} + {sleep_time*1000:.4f} = {(elapsed_time + sleep_time)*1000} ms')
    
    def ros_control(self):
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(self.rc.tm)
        executor.add_node(self.rc.jm) 
        executor.add_node(self.hand_eye)  
        executor.add_node(self)        # MujocoROSBridge 자신도 spin대상에 포함 - joint qpos publish, subscribe 위해서
        executor.spin()
        executor.shutdown()

        self.rc.tm.destroy_node()
        self.rc.jm.destroy_node()