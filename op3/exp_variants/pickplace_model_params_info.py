
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
                beta = 1e-2,
                deterministic_sampling = False
            )
        ),
        folder = "/nfs/kun1/users/rishiv/Research/op3_exps/09-14-pickplace-o12-v2-10k-single-step-physics-v2-reg-smallbeta/09-14-pickplace_o12_v2_10k-single_step_physics-v2-reg-smallbeta_2019_09_15_06_41_46_0000--s-96623",
    ),
    curriculum_aws_params=dict(
        op3_args=dict(
            refinement_model_type="size_dependent_conv",
            decoder_model_type="reg",
            dynamics_model_type="reg_ac32",
            sto_repsize=64,
            det_repsize=64,
            extra_args=dict(
                beta=1e-2,
                deterministic_sampling=False
            )
        ),
        folder="/nfs/kun1/users/rishiv/Research/op3_exps/09-20-pickplace-multienv-10k-curriculum-v2-reg/09-20-pickplace_multienv_10k-curriculum-v2-reg_2019_09_20_07_24_06_0000--s-20061",
    ),
    random_aws_params=dict(
        op3_args=dict(
            refinement_model_type="size_dependent_conv",
            decoder_model_type="reg",
            dynamics_model_type="reg_ac32",
            sto_repsize=64,
            det_repsize=64,
            extra_args=dict(
                beta=1e-2,
                deterministic_sampling=False
            )
        ),
        folder="/nfs/kun1/users/rishiv/Research/op3_exps/09-20-pickplace-multienv-10k-random-alternating-v2-reg/09-20-pickplace_multienv_10k-random_alternating-v2-reg_2019_09_20_07_31_14_0000--s-13521",
    ),
    reg_9_23_params=dict(
        op3_args=dict(
            refinement_model_type="size_dependent_conv",
            decoder_model_type="reg",
            dynamics_model_type="reg_ac32",
            sto_repsize=64,
            det_repsize=64,
            extra_args=dict(
                beta=1e-2,
                deterministic_sampling=False
            )
        ),
        folder="/nfs/kun1/users/rishiv/Research/op3_exps/09-23-pickplace-o12-noise-10k-curriculum-v2-reg/09-23-pickplace_o12_noise_10k-curriculum-v2-reg_2019_09_24_04_45_57_0000--s-98070",
    ),
    sto_9_23_params=dict(
        op3_args=dict(
            refinement_model_type="size_dependent_conv",
            decoder_model_type="reg",
            dynamics_model_type="reg_ac32",
            sto_repsize=128,
            det_repsize=0,
            extra_args=dict(
                beta=1e-2,
                deterministic_sampling=False
            )
        ),
        folder="/nfs/kun1/users/rishiv/Research/op3_exps/09-23-pickplace-o12-noise-10k-curriculum-v2-sto/09-23-pickplace_o12_noise_10k-curriculum-v2-sto_2019_09_24_04_48_25_0000--s-40338",
    ),
)

