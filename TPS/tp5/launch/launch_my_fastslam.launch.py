#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('tp5'), 'rviz', 'fastslam.rviz'
    )

    # --- Nodos ---
    # features: extrae landmarks (posiciones) del /map -> /landmarks
    features = Node(
        package='tp5',
        executable='features',
        name='features',
        output='screen',
    )

    # feature_finder: observa los landmarks desde /scan -> /observed_landmarks
    # (rango y ángulo relativos en el frame del robot)
    feature_finder = Node(
        package='tp5',
        executable='feature_finder',
        name='feature_finder',
        output='screen',
    )

    # odom_to_delta: convierte /calc_odom -> /delta (modelo de odometría).
    # La consigna asume que la simulación publica /delta; en este workspace lo
    # genera este nodo auxiliar. Si la simulación ya publica /delta, quitarlo.
    odom_to_delta = Node(
        package='tp5',
        executable='odom_to_delta',
        name='odom_to_delta',
        output='screen',
    )

    # fastslam: nodo principal. Consume /delta y /observed_landmarks y publica
    # la pose estimada y los landmarks (con covarianza) de la mejor partícula.
    fastslam = Node(
        package='tp5',
        executable='fastslam',
        name='fastslam',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    # --- Orden de arranque ---
    start_finder_after_features = RegisterEventHandler(
        OnProcessStart(target_action=features, on_start=[feature_finder])
    )
    start_fastslam_after_finder = RegisterEventHandler(
        OnProcessStart(target_action=feature_finder, on_start=[fastslam])
    )

    ld = LaunchDescription()
    ld.add_action(features)
    ld.add_action(start_finder_after_features)
    ld.add_action(start_fastslam_after_finder)
    ld.add_action(odom_to_delta)
    ld.add_action(rviz)
    return ld
