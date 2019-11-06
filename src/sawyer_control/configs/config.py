import sawyer_control.configs.austri_config as austri_config
import sawyer_control.configs.base_config as base_config
import sawyer_control.configs.pearl_austri_config as pearl_austri_config
import sawyer_control.configs.pearl_fjalar_config as pearl_fjalar_config
import sawyer_control.configs.ros_config as ros_config
config_dict = dict(
    ros_config=ros_config,
    base_config=base_config,
    austri_config=austri_config,
    pearl_austri_config=pearl_austri_config,
    pearl_fjalar_config=pearl_fjalar_config,
)