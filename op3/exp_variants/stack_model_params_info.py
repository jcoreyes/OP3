
########
# This file is to keep track of the *_params.pkl files. Each file should have the following:
# 1. Parameter dictionary
# 2. The folder in op3_exps or s3 where the parameters came from

params_to_info = dict(
    s64d64_v1_params = dict(
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",
            decoder_model_type = "reg",
            dynamics_model_type = "reg_ac32",
            sto_repsize = 64,
            det_repsize = 64,
            extra_args = dict(
                beta = 1,
                deterministic_sampling = False
            )
        ),
        folder = "/nfs/kun1/users/rishiv/Research/op3_exps/08-24-stack-o2p2-60k-single-step-physics-v2/08-24-stack_o2p2_60k-single_step_physics-v2_2019_08_24_15_50_23_0000--s-67083",
    ),
    unfactorized_v1_params = dict(
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",
            decoder_model_type = "reg",
            dynamics_model_type = "reg_ac32",
            sto_repsize = 64*7,
            det_repsize = 64*7,
            extra_args = dict(
                beta = 1,
                deterministic_sampling = False
            )
        ),
        folder = "/nfs/kun1/users/rishiv/Research/op3_exps/09-14-stack-o2p2-60k-single-step-physics-v2-unfactorized/09-14-stack_o2p2_60k-single_step_physics-v2-unfactorized_2019_09_15_06_36_07_0000--s-51801",
    ),
)

