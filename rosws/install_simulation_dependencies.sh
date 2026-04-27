#!/usr/bin/env bash
# Instala dependencias del e-manual TurtleBot3 (Humble) + Nav2 para custom_room.launch.py
# Uso: bash install_simulation_dependencies.sh
# Requiere: sudo sin contraseña o introducirla cuando la pida.

set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  ros-humble-turtlebot3 \
  ros-humble-turtlebot3-msgs \
  ros-humble-turtlebot3-simulations \
  ros-humble-turtlebot3-teleop \
  ros-humble-navigation2 \
  ros-humble-gazebo-ros-pkgs

echo
echo "Listo. En cada terminal nueva:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $(dirname "$(readlink -f "$0")")/install/setup.bash"
echo
echo "Opcional (modelo del robot, coherente con tu launch que usa burger):"
echo "  export TURTLEBOT3_MODEL=burger"
