import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer as mj_view
import threading
import time
from .scene_monitor import SceneMonitor
from .image_publisher import MujocoCameraBridge
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState #jointstate()추가

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
      
        self.ctrl_dof = 8 # 7 + 1 <- mujoco urdf엔 9개나, controller에서 8개로 계산하기 때문
        self.ctrl_step = 0
        
        '''
        #   각각의 qpos index에 해당하는 joint 이름을 출력해 봅시다.
        print(f"[Bridge] MuJoCo model.nq = {self.model.nq}")
        for i in range(self.model.nq):
            # mj_id2name(model, object_type, object_id) → joint 이름 반환
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"[Bridge] qpos index {i} -> joint name = {name}")
        # ↑↑↑ 여기까지 삽입 ↑↑↑
        # ──────────────────────────────────────────────────────────────────────────
        '''
        ''' 일단 publish 관련 코드 주석처리
        # 퍼블리시 주기 제어용 변수
        self.publish_rate_hz = 100  # 예: 100Hz로 퍼블리시
        self.publish_skip_count = int(self.ctrl_freq / self.publish_rate_hz)
        self._step_counter = 0
        '''
        '''
        # JoinstState publisher 생성 -> moveit으로 joint 정보 퍼블리시 
        self.joint_state_to_moveit = self.create_publisher(JointState, 'joint_states', 10)   
        # “초기 퍼블리시가 이미 이루어졌는지” 확인하기 위한 플래그 -> 초기 한번만 퍼블리시 - 렉 안걸리게 하려고
        self._has_published_initial = False

        # MuJoCo qpos 순서와 1:1 매핑되는 moveit_joint 이름들
        self.moveit_joint_names = []
        for i in range(1, 8):
            self.moveit_joint_names.append(f"panda_joint{i}")
        self.moveit_joint_names.append("panda_finger_joint1")
        '''

        # moveit -> /panda/joint_set을 받기 위한 subscription 변수
        # moveit_sim_interface에서 moveit이 보낸 joint_set 메세지: /panda/joint_set :  현재 panda로 되어있음
        self.latest_joint_set = None
        self.joint_set_mutex = threading.Lock()
        self.joint_set_sub = self.create_subscription(
            JointState,
            '/panda/joint_set',
            self.jointSetCallback,
            10
        )

        # MuJoCo qpos 순서와 1:1 매핑되는 moveit_joint 이름들
        # 1에서 7까지로 되어있음. 
        self.mujoco_joint_names = []
        for i in range(1, 8):
            self.mujoco_joint_names.append(f"fr3_joint{i}")
        # panda는 joint 정보 7개만 보냄
        # self.mujoco_joint_names.append("finger_joint1")
        # finger_joint1으로 설정 - franka_hand_urdf.xml #finger_joint2도 있긴함

        self.running = True
        self.lock = threading.Lock()
        self.robot_thread = threading.Thread(target=self.robot_control, daemon=True)
        self.hand_eye_thread = threading.Thread(target=self.hand_eye_control, daemon=True)
        self.ros_thread = threading.Thread(target=self.ros_control, daemon=True)

    #"/panda/joint_set" 토픽에서 moveit이 보낸 JointState 메시지를 받는 콜백 함수
    def jointSetCallback(self, msg):
        # moveit에서 보낸 joint_set 메세지를 그대로 저장
        with self.joint_set_mutex:
            self.latest_joint_set = msg
        #로깅함수
        self.get_logger().info(f"Received joint_set: {self.latest_joint_set.position}")
        
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

                    '''기존 실행 
                    mujoco.mj_step(self.model, self.data)  # 시뮬레이션 실행
                    self.rc.updateModel(self.data, self.ctrl_step)                    
                    self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   
                    '''
                    # moveit이 보낸 /panda/joint_set 메세지 존재 여부 검사 - 있으면 그 값을 qpos로 덮어쓰기
                    
                    target_js = None
                    with self.joint_set_mutex:
                        if self.latest_joint_set is not None:
                        
                            target_js = self.latest_joint_set
                            target_js.name = self.mujoco_joint_names.copy()  # moveit_joint_names를 mujoco_joint_names로 변경
                    
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
                        #self.rc.updateModel(self.data, self.ctrl_step)  #시뮬레이터 내부 상태를 DMController에 업데이트 
                    
                    else:
                        # moveit이 보낸 궤적이 없으면, 로봇 컨트롤러에서 계산한 값을 qpos로 덮어쓰기
                        mujoco.mj_step(self.model, self.data)  # 시뮬레이션 실행
                        self.rc.updateModel(self.data, self.ctrl_step)  #시뮬레이터 내부 상태를 DMController에 업데이트                 
                        self.data.ctrl[:self.ctrl_dof] = self.rc.compute() #DMController에서 계산한 제어값을 qpos로 덮어쓰기

                    '''
                    #moveit으로 joint 정보 한번 퍼블리시
                    if not self._has_published_initial:
                        js_msg = JointState()
                        js_msg.header.stamp = self.get_clock().now().to_msg()
                        js_msg.name = self.moveit_joint_names.copy()
                        # qpos를 position으로
                        js_msg.position = [float(self.data.qpos[i]) for i in range(self.ctrl_dof)]
                        # velocity 정보도 함께 포함할 수 있다면
                        # js_msg.velocity = [float(self.data.qvel[i]) for i in range(self.ctrl_dof)]
                        # effort 정보가 필요하면, data.sensordata나 data.qfrc_applied 등에서 가져와 채울 수 있다
                        # js_msg.effort = [...]
                        self.joint_state_to_moveit.publish(js_msg)
                        self._has_published_initial = True
                    
                    '''
                    '''
                    # 3) JointState 퍼블리시 스로틀링 (매 10스텝마다 1회)
                    self._step_counter += 1
                    if self._step_counter >= self.publish_skip_count:
                        self._step_counter = 0
                        js_msg = JointState()
                        js_msg.header.stamp = self.get_clock().now().to_msg()
                        js_msg.name = self.joint_names.copy()
                        # qpos를 position으로
                        js_msg.position = [float(self.data.qpos[i]) for i in range(self.ctrl_dof)]
                        # velocity 정보도 함께 포함할 수 있다면
                        # js_msg.velocity = [float(self.data.qvel[i]) for i in range(self.ctrl_dof)]
                        # effort 정보가 필요하면, data.sensordata나 data.qfrc_applied 등에서 가져와 채울 수 있다
                        # js_msg.effort = [...]
                        self.joint_state_pub.publish(js_msg)
                    '''

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
        executor.add_node(self)        # MujocoROSBridge 자신도 spin대상에 포함 
        executor.spin()
        executor.shutdown()

        self.rc.tm.destroy_node()
        self.rc.jm.destroy_node()