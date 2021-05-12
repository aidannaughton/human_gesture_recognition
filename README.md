# README
## Project Structure
The standalone python version is found in the standalone folder. 
- Run it by calling `python3 pose_estimation.py`

The ROS integration is found in the ros folder.
- A copy of the package itself is in `ROS/pose_estimation`
- Additionally there are two catkin workspaces
  - `catkin_ws` is the complete project.
  - `cv_bridge_ws` is the workspace used to build cv_bridge for python 3.6

This package was greated with rospy in ros melodic (Ubuntu 18.04)
The pose_estimation package requires the packages cv_camera and cv_bridge.

## Required Packages
cv_camera:
- ros wiki page: http://wiki.ros.org/cv_camera
- github page: https://github.com/OTL/cv_camera

cv_bridge:
- ros wiki page: http://wiki.ros.org/cv_bridge
- github page: https://github.com/ros-perception/vision_opencv

The cv_bridge package must be built with python 3.6, as it is natively built in python 2.7. To do so I followed the following instructions
    (credit to https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3)

### Building cv_bridge:
Install dependencies
```
sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge
```
- `python-catkin-tools` is needed for catkin tool
- `python3-dev` and `python3-catkin-pkg-modules` is needed to build cv_bridge
- `python3-numpy` and `python3-yaml` is cv_bridge dependencies
- `ros-kinetic-cv-bridge` is needed to install a lot of cv_bridge deps. Probaply you already have it installed.

Create catkin workspace
```
mkdir cv_bridge_ws
cd cv_bridge_ws
catkin init
```

Instruct catkin to set cmake variables
```
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

Instruct catkin to install built packages into install place. It is $CATKIN_WORKSPACE/install folder
```
catkin config --install
```

Clone cv_bridge src
```
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
```

Find version of cv_bridge in your repository
```
apt-cache show ros-melodic-cv-bridge | grep Version
```

Checkout right version in git repo. In our case it is 1.13.0
```
cd src/vision_opencv/
git checkout 1.13.0
cd ../../
```

Build
```
catkin build cv_bridge
```

Navigate to your standard catkin workspace, make and source the setup
```
cd [Path to your ws]
catkin_make
source devel/setup.bash
```

Extend environment with the python 3.6 cv_bridge_ws package
```
source [Path to cv_bridge_ws]/install/setup.bash --extend
```

You can test to see if it worked properly by running a python shell
```python3
>>> from cv_bridge.boost.cv_bridge_boost import getCvType
```

## Running the project
To run the project navigate to ROS/catkin_ws and run the following commands
```
source devel/setup.bash
source ../cv_bridge_ws/install/setup.bash --extend
roslaunch pose_estimation pose_estimation.launch
```

## To use the project itself:
- option 1 will run the pose estimation and pointing recognition
- option 2 will assist in the creation of a ground truth variable (NOTE: there is no ground truth at the start of the program)
- option 3 will save the ground truth if it exists to the specified file (NOTE: these files are stored in /home/{your username}/.ros)
- option 4 will load the ground truth from a specified file if it exists
