from sawyer_control.configs.base_config import *
import numpy as np
from gym.spaces import Box


# First dimension is forward/back, second dimension is left/right, third dimension is up/down

# Reaching info
POSITION_SAFETY_BOX_LOWS = np.array([0.46, -0.17, 0.18])
POSITION_SAFETY_BOX_HIGHS = np.array([0.8, 0.17, 0.52])

POSITION_SAFETY_BOX = Box(POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS, dtype=np.float32)

POSITION_CONTROL_EE_ORIENTATION=Quaternion(
    x=1.0, y=0.0, z=0.0, w=0.0
)

# Reset info
RESET_SAFETY_BOX_LOWS = POSITION_SAFETY_BOX_LOWS
RESET_SAFETY_BOX_HIGHS = POSITION_SAFETY_BOX_HIGHS
RESET_SAFETY_BOX = Box(RESET_SAFETY_BOX_LOWS, RESET_SAFETY_BOX_HIGHS, dtype=np.float32)

POSITION_RESET_POS = np.mean(np.array([POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS]), axis=0)
# 'right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6'
RESET_ANGLES = np.array(
# [-0.5391240234375, -1.3373681640625, 0.2619453125, 2.274025390625, -0.0606650390625, 0.6475078125, 4.7030986328125]
   [-0.5811299085617065, -1.3128759860992432, 0.31348729133605957, 2.7572646141052246, 0.3204902410507202, -0.14360937476158142, 4.205990314483643]
)
JOINT_NAMES = ['right_j0',
               'right_j1',
               'right_j2',
               'right_j3',
               'right_j4',
               'right_j5',
               'right_j6'
               ]



