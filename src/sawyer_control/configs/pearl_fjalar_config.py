from sawyer_control.configs.base_config import *
import numpy as np
from gym.spaces import Box


# First dimension is forward/back, second dimension is left/right, third dimension is up/down

# Reaching info
POSITION_SAFETY_BOX_LOWS = np.array([0.46, -0.17, 0.28])
POSITION_SAFETY_BOX_HIGHS = np.array([0.7, 0.17, 0.42])

POSITION_SAFETY_BOX = Box(POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS, dtype=np.float32)

POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=np.sqrt(2)/2.0, y=-np.sqrt(2)/2.0, z=0.0, w=0.0
)

# TODO - should this be different from position?

## Center
# x: 0.590826253496
# y: 0.188318873384
# z: 0.214806981327

# TORQUE_EE_SAFETY_BOX_LOWS = np.array([0.5, -0.1, 0.25])
# TORQUE_EE_SAFETY_BOX_HIGHS = np.array([0.6, 0.1, 0.35])

TORQUE_EE_SAFETY_BOX_LOWS = np.array([0.50, 0.0, 0.15])
TORQUE_EE_SAFETY_BOX_HIGHS = np.array([0.80, 0.30, 0.45])
TORQUE_EE_SAFETY_BOX = Box(TORQUE_EE_SAFETY_BOX_LOWS, TORQUE_EE_SAFETY_BOX_HIGHS, dtype=np.float32)

TORQUE_SAFETY_BOX_LOWS = np.array([-0.13, -0.4, 0.1])
TORQUE_SAFETY_BOX_HIGHS = np.array([0.7, 0.4, 0.8])
TORQUE_SAFETY_BOX = Box(TORQUE_SAFETY_BOX_LOWS, TORQUE_SAFETY_BOX_HIGHS, dtype=np.float32)

# Max vals: https://github.com/RethinkRobotics/sawyer_moveit/blob/master/sawyer_moveit_config/config/joint_limits.yaml
# MAX_TORQUES = 0.75 * np.array([3.5, 5.0, 5.0, 5.0, 3.0, 2.0, 3.0])  # torque limits for each joint
# MAX_TORQUES = 0.5 * np.array([4.0, 5.0, 3.0, 4.0, 2.0, 2.0, 3.0])  # torque limits for each joint
# MAX_TORQUES = 0.5 * np.array([8.0, 12.0, 6.0, 5.0, 4.0, 3.0, 6.0])  # Aurick's torque
MAX_TORQUES = 0.5 * np.array([8.0, 9.0, 6.0, 5.0, 4.0, 3.0, 6.0])  # Lowered effect of j1

JOINT_TORQUE_HIGH = MAX_TORQUES
JOINT_TORQUE_LOW = -1*MAX_TORQUES
GRAVITY_COMP_ADJUSTMENT = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

MAX_VELOCITIES = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])

# Reset info
RESET_SAFETY_BOX_LOWS = TORQUE_SAFETY_BOX_LOWS
RESET_SAFETY_BOX_HIGHS = TORQUE_SAFETY_BOX_HIGHS
RESET_SAFETY_BOX = Box(RESET_SAFETY_BOX_LOWS, RESET_SAFETY_BOX_HIGHS, dtype=np.float32)

POSITION_RESET_POS = np.mean(np.array([POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS]), axis=0)
# 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6'

# RESET_ANGLES = np.array([0.394435546875, -0.5772666015625, -1.0237197265625, 1.80161328125, 1.0277392578125,
#                          0.98661328125, -2.47254296875])


## Aurick's
# RESET_ANGLES = np.array([ 0.28625879, -0.06652148, -1.45274023,  1.87639258,  1.47005664,
#         1.47084668, -0.87656152])

## Mine
# RESET_ANGLES = np.array([0.1938154296875, -0.824548828125, -0.23578515625, 1.6639423828125, 0.245529296875,
#                          0.8700537109375, -2.9615185546875])

## Calibration Pos
RESET_ANGLES = np.array([0.0405908203125, -1.141642578125, -0.10630859375, 2.0363115234375, 0.0765771484375,
                         0.6658544921875, 0.0361298828125]
)

## Max wrist rotation
WRIST_ANGLE_LOW = np.array([-3])
WRIST_ANGLE_HIGH = np.array([3])

JOINT_NAMES = ['right_j0',
               'right_j1',
               'right_j2',
               'right_j3',
               'right_j4',
               'right_j5',
               'right_j6'
               ]
