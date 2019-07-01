import sawyer_control.configs.austri_config as austri_config
import sawyer_control.configs.base_config as base_config
import sawyer_control.configs.ros_config as ros_config
import sawyer_control.configs.laudri_reaching as laudri_reaching_config
import sawyer_control.configs.laudri_peg as laudri_peg_config


config_dict = dict(
    ros_config=ros_config,
    base_config=base_config,
    austri_config=austri_config,
    laudri_reaching_config=laudri_reaching_config,
    laudri_peg_config=laudri_peg_config
)
