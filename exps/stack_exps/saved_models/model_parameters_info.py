
########
# This file is to keep track of the saved *.pkl model files. Each entry should have the following:
# 1. A parameter dictionary
# 2. The name of the file to load

params_to_info = dict(
    s64d64_v1_params = dict(
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",
            decoder_model_type = "reg",
            dynamics_model_type = "reg_ac32",
            sto_repsize = 64,
            det_repsize = 64,
            extra_args = dict(
                beta = 1e-2,
                deterministic_sampling = False
            )
        ),
        model_file = "s64d64_v1_params.pkl"
    ),
)