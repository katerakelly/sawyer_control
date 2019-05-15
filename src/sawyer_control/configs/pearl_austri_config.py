from sawyer_control.configs.base_config import *
import numpy as np
from gym.spaces import Box


# First dimension is forward/back, second dimension is left/right, third dimension is up/down
POSITION_SAFETY_BOX_LOWS = np.array([.4, -.3, .1])
POSITION_SAFETY_BOX_HIGHS = np.array([.7, .3, .5])

POSITION_SAFETY_BOX = Box(POSITION_SAFETY_BOX_LOWS, POSITION_SAFETY_BOX_HIGHS, dtype=np.float32)
