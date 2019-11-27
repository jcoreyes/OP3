########
# This file is to keep track of the saved *.pkl model files. Each entry should have the following:
# 1. A parameter dictionary
# 2. The name of the file to load

params_to_info = dict(
    cloth_reg = dict(  # OP3 Model
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",
            decoder_model_type = "reg",
            dynamics_model_type = "reg_ac32",
            sto_repsize = 64,
            det_repsize = 64,
            extra_args = dict(
                beta = 0.01,
                deterministic_sampling = False
            )
        ),
        model_file = "cloth_reg_params.pkl",
    ),
    cloth_sequence = dict(  # Sequence IODINE Model (mentioned in appendix of IODINE paper)
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",
            decoder_model_type = "reg",
            dynamics_model_type = "reg_ac32",
            sto_repsize = 128,
            det_repsize = 0,
            extra_args = dict(
                beta = 0.01,
                deterministic_sampling = False
            )
        ),
        model_file = "cloth_sequence_params.pkl",
    ),
    cloth_static = dict(  # Static IODINE Model
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",
            decoder_model_type = "reg",
            dynamics_model_type = "reg_ac32",
            sto_repsize = 128,
            det_repsize = 0,
            extra_args = dict(
                beta = 0.01,
                deterministic_sampling = False
            ),
            action_dim=0,
        ),
        model_file = "cloth_static_params.pkl",
    ),
)