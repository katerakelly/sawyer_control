from sawyer_control.configs.base_config import *
import numpy as np
from gym.spaces import Box


# First dimension is forward/back, second dimension is left/right, third dimension is up/down

# Reaching info
POSITION_SAFETY_BOX_LOWS = np.array([0.46, -0.17, 0.28])
POSITION_SAFETY_BOX_HIGHS = np.array([0.8, 0.17, 0.52])

POSITION_SAFETY_BOX = Box(POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS, dtype=np.float32)

POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=1.0, y=0.0, z=0.0, w=0.0
)

# TODO - should this be different from position?
TORQUE_SAFETY_BOX_LOWS = np.array([0.5, -0.1, 0.25])
TORQUE_SAFETY_BOX_HIGHS = np.array([0.6, 0.1, 0.35])
TORQUE_SAFETY_BOX = Box(TORQUE_SAFETY_BOX_LOWS, TORQUE_SAFETY_BOX_HIGHS, dtype=np.float32)

# Reset info
RESET_SAFETY_BOX_LOWS = TORQUE_SAFETY_BOX_LOWS
RESET_SAFETY_BOX_HIGHS = TORQUE_SAFETY_BOX_HIGHS
RESET_SAFETY_BOX = Box(RESET_SAFETY_BOX_LOWS, RESET_SAFETY_BOX_HIGHS, dtype=np.float32)

POSITION_RESET_POS = np.mean(np.array([POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS]), axis=0)
# 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6'

RESET_ANGLES = np.array([0.394435546875, -0.5772666015625, -1.0237197265625, 1.80161328125, 1.0277392578125,
                         0.98661328125, -2.47254296875])

# RESET_ANGLES = np.array([ 0.28625879, -0.06652148, -1.45274023,  1.87639258,  1.47005664,
#         1.47084668, -0.87656152])

JOINT_NAMES = ['right_j0',
               'right_j1',
               'right_j2',
               'right_j3',
               'right_j4',
               'right_j5',
               'right_j6'
               ]



