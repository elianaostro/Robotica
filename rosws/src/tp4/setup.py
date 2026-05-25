from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tp4'

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
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "ekf = tp4.my_ekf:main",
            "features = tp4.features:main",
            "feature_finder = tp4.feature_finder:main",
            "ekf_prediction = tp4.ekf_prediction:main",
            "ekf_correction = tp4.ekf_correction:main",
        ],
    },
)
