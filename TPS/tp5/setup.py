from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'tp5'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='udesa',
    maintainer_email='tadeo.casiraghi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "features = tp5.features:main",
            "feature_finder = tp5.feature_finder:main",
            "fastslam = tp5.my_fastslam:main",
            "odom_to_delta = tp5.odom_to_delta:main",
        ],
    },
)
