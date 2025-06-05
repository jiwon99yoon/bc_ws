from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'dm_ros'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/dm_ros']),  # 패키지 마커 파일
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob('launch/*.py')),
    ('share/' + package_name + '/config', glob('config/*.yaml')),
]

robots_path = 'robots'
for root, dirs, files in os.walk(robots_path):
    for file in files:
        relative_path = os.path.relpath(root, robots_path)
        install_path = os.path.join('share', package_name, 'robots', relative_path)
        data_files.append((install_path, [os.path.join(root, file)]))

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=[package_name],
    data_files= data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Haeseong Lee',
    maintainer_email='lhs4138@snu.ac.kr',
    description='dyros mujoco with ros2',
    license='TODO: License declaration',
    extras_require={
        "test": ["pytest"]},
    entry_points={
        'console_scripts': [
            'fr3_test = dm_ros.fr3_test:main',
            'fr3_free = dm_ros.fr3_free:main',
        ],
    },
)
