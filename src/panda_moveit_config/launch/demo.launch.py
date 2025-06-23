# from moveit_configs_utils import MoveItConfigsBuilder
# from moveit_configs_utils.launches import generate_demo_launch

# def generate_launch_description():
#     moveit_config = MoveItConfigsBuilder("panda", package_name="panda_moveit_config").to_moveit_configs()
#     return generate_demo_launch(moveit_config)


# import os
# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory
# 
# def generate_launch_description():
    # pkg = get_package_share_directory('panda_moveit_config')
# 
    # return LaunchDescription([
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'rsp.launch.py'))
        # ),
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'move_group.launch.py'))
        # ),
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'moveit_rviz.launch.py'))
        # )
    # ])
# 

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
