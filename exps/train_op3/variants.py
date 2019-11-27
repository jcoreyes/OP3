## This file contains the needed dictionaries to train models

# Training on stack dataset
stack_variant = dict(
    op3_args=dict(
        refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
        decoder_model_type="reg",  # reg, reg_no_share
        dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
        sto_repsize=64,
        det_repsize=64,
        extra_args=dict(
            beta=1e-2,
            deterministic_sampling=False
        ),
        K=7
    ),
    schedule_args=dict(  # Arguments for TrainingScheduler
        seed_steps=4,
        T=5,  # Max number of steps into the future we want to go or max length of a schedule
        schedule_type='single_step_physics',
    ),
    training_args=dict(  # Arguments for OP3Trainer
        batch_size=80,  # Change to appropriate constant based off dataset size and GPU's
        lr=3e-4,
    ),
    num_epochs=150,
    save_period=1,
    dataparallel=True,
    dataset='stack_o2p2_60k',
    debug=False,
)

# Training on pickplace dataset
pickplace_variant = dict(
    op3_args=dict(
        refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
        decoder_model_type="reg",  # reg, reg_no_share
        dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
        sto_repsize=64,
        det_repsize=64,
        extra_args=dict(
            beta=1e-2,
            deterministic_sampling=False
        ),
        K=4
    ),
    schedule_args=dict(  # Arguments for TrainingScheduler
        seed_steps=4,
        T=5,  # Max number of steps into the future we want to go or max length of a schedule
        schedule_type='curriculum',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
    ),
    training_args=dict(  # Arguments for OP3Trainer
        batch_size=80,  # Change to appropriate constant based off dataset size and GPU's
        lr=3e-4,
    ),
    num_epochs=300,
    save_period=1,
    dataparallel=True,
    dataset='pickplace_multienv_10k',
    debug=False,
)

# Training on cloth dataset
cloth_variant = dict(
    op3_args=dict(
        refinement_model_type="size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
        decoder_model_type="reg",  # reg, reg_no_share
        dynamics_model_type="reg_ac32",  # reg_ac32, reg_ac32_no_share
        sto_repsize=64,
        det_repsize=64,
        extra_args=dict(
            beta=1e-2,
            deterministic_sampling=False
        ),
        K=4
    ),
    schedule_args=dict(  # Arguments for TrainingScheduler
        seed_steps=4,
        T=5,  # Max number of steps into the future we want to go or max length of a schedule
        schedule_type='curriculum',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
    ),
    training_args=dict(  # Arguments for OP3Trainer
        batch_size=80,  # Change to appropriate constant based off dataset size
        lr=3e-4,
    ),
    num_epochs=300,
    save_period=1,
    dataparallel=True,
    dataset='cloth',
    debug=False,
)