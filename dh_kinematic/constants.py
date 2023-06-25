import numpy as np

# link lengths in meters
# source: http://wiki.icub.org/wiki/Arm_Control
# TO DO: hand_length
shoulder_distance = 0.05
upper_arm_length = 0.22
forearm_length = 0.16
head_distance = 0.02
forearm_thickness = 0.01

"""
arm angle limits
shoulder yaw:   0° to -120°
shouler pitch:  80° to 40°
shoulder roll:  90° to 0°
elbow:          -10° to -90°
"""
theta_limits = np.array([[0, 120], [40, 80], [0, 90], [-190, -10]])
theta_limits_rad = np.radians(theta_limits)

# in 2D, shoulder yaw and shoulder pitch are constant, both 90°
theta_limits_2d = np.array([[90, 90], [90, 90], [0, 90], [-160, -10]])
theta_limits_2d_rad = np.radians(theta_limits_2d)
