This project is based on cv2 and the mediapipe packages for python. Using the mediapipe holistic predictor, we can get a pose for the human skeleton as well as hands. (Face as well, but that isn't used here.) We can further extract keypoints on this skeleton which have x,y,z coordinates, where z is the depth in the image. Currently, we only use x and y coordinates, however z coordinates could be used to construct a 3D pointing system sensitive to objects at different depths from the camera. 


https://www.youtube.com/watch?v=pG4sUNDOZFg&ab_channel=NicholasRenotte

https://github.com/nicknochnack/Full-Body-Estimation-using-Media-Pipe-Holistic/blob/main/Media%20Pipe%20Holistic%20Tutorial.ipynb
