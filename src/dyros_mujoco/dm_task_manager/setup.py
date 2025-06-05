from setuptools import find_packages, setup

package_name = 'dm_task_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hs_dyros',
    maintainer_email='lhs4138@snu.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        "test": ["pytest"]},    entry_points={
        'console_scripts': [
            'task_move_cli = dm_task_manager.task_move_client:main',
            'joint_move_cli = dm_task_manager.joint_move_client:main'  
        ],
    },
)
