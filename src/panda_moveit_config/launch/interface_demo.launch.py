# demo.launch.py wrapper해서 remap 
# -> /joint_states대신 mujoco가 /fr3/joint_set으로 qpos publish하면 moveit에서 그것만 불러오도록
# 기존 demo.launch.py로 실행했을때, 실행되는 헬버함수 generate_demo_launch가 실행하는
# joint_state_broadcaster가 publish하는 /joint_states와의 중복을 막기위해!
# 다만, 기존 dyros_mujoco의 initial qpos를 불러온다면 (0, -75, 0, -135, 0, 90, 45) 이 값은 moveit에서 self-collision에 해당하기에
# moveit이 불러오는 initial qpos와는 중복가능
# 따라서 dyros_mujoco/fr3_free의 initial qpos를 바꿔야함 - 이러면 초기 mujoco의 qpos 불러오면 moveit의 initial qpos와 중복 X
# mujoco의 qpos는 dm_controller/src/controller.cpp의 q_idle_으로 수정
# moveit의 qpos는 franka_description/robots/panda_arm.ros2_control.xacro에서 수정 가능
# 그 이유는 현재 panda_moveit_config/config/panda.urdf.xacro가 franka_description내의 panda_arm_hand.urdf.xacro를 불러옴과 동시에
# panda.ros2_control.xacro의 경로를 못찾아 ros2_control은 franka_description/robots/panda_arm.ros2_control.xacro로 진행되기 때문

# 만일 mujoco의 qpos publish naming을 /joint_states로 한다면
# 기존 demo.launch.py로만 실행해도 됨
# 하지만 이 경우 moveit 내부 joint_state_broadcaster가 publish하는 /joint_states와 중복됨

# import os
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import GroupAction, IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch_ros.actions import SetRemap

# def generate_launch_description():
#     pkg_share = get_package_share_directory('panda_moveit_config')
#     demo = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(pkg_share, 'launch', 'demo.launch.py')
#         )
#     )

#     # 같은 스코프 안에 있는 모든 노드(/joint_states 구독자)에 remap을 적용
#     wrapper = GroupAction(
#         actions=[
#             SetRemap(src='/joint_states', dst='/fr3/joint_set'),
#             demo
#         ]
#     )

#     return LaunchDescription([wrapper])

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('panda_moveit_config')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'rsp.launch.py'))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'move_group.launch.py'))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'moveit_rviz.launch.py'))
        )
    ])
