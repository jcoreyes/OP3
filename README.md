# OP3
We present object-centric perception, prediction, and planning (OP3), an entity-centric dynamic latent variable framework for model-based reinforcement learning that acquires entity representations from raw visual observations without supervision and uses them to predict and plan.

CoRL 2019 Conference Paper: ["Entity Abstraction in Visual Model-Based Reinforcement Learning"](https://arxiv.org/abs/1910.12827). 

More information is available at the [project website](https://sites.google.com/view/op3website/).

## Installation
1. Copy `conf.py` to `conf_private.py`:
```
cp op3/launchers/conf.py op3/launchers/conf_private.py
```
2. Install and use the included Ananconda environment
```
$ conda env create -f environment/linux-gpu-env.yml
$ source activate op3
```
These Anaconda environments use MuJoCo 1.5 and gym 0.10.5 which are not needed for training the model but are needed for generating the datasets and running MPC.
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

3. Download this version of [doodad](https://github.com/jcoreyes/doodad) and add the repo to your pythonpath. Docker and doodad dependencies will only be needed if running on AWS or GCP.

## Running Experiments
### Downloading and Generating Datasets
Download datasets from this [google drive folder](https://drive.google.com/drive/folders/18wbAHeyvBblabn9pz3MEajox0-1mXrn2?usp=sharing) to `op3/data/datasets/`.
#### Single-Step Block Stacking
<br>
<p float="center">
  <img src="https://drive.google.com/uc?export=view&id=1iJDYIdx99qdcwTIsMM1Q7n0I3_j0DiXY" width="19%">
  <img src="https://drive.google.com/uc?export=view&id=1ojpswJna-mO0jJsbUAVkm5ZfBhDZVyXI" width="19%">
  <img src="https://drive.google.com/uc?export=view&id=1vIGQwLQuXOLLNUv8GyiCmv9edAiqWNxb" width="19%">
  <img src="https://drive.google.com/uc?export=view&id=1jOLp9agc-WF1oi9AiiU16VvyPf4A9Ipw" width="19%">
  <img src="https://drive.google.com/uc?export=view&id=1TLUQo3ekl9Err_Mi_7hK77Q2bAUwlTwG" width="19%">
</p>

The stack_o2p2_60.h5 dataset contains 60,000 before and after images of blocks being dropped. This is the same dataset used in [O2P2](https://people.eecs.berkeley.edu/~janner/o2p2/) "Reasoning About Physical Interactions with Object-Oriented Prediction and Planning."

This dataset can also be generated using the Mujoco environment.

```
cd op3/envs/blocks/mujoco
python stacking_generating_singlestep.py --num_images NUM_IMAGES 
```
which will by default output the dataset to op3/envs/blocks/rendered/blocks.h5. See the args in the file for more options such as controlling the number of objects in the scene.

#### Multi-Step Block Stacking
The pickplace_multienv_10k.h5 dataset contains 10,000 trajectories where each trajectory contains five frames of randomly picking and placing blocks.


This dataset can also be generated using the Mujoco environment. Run
```
cd op3/envs/blocks/mujoco
python block_pick_and_place.py -f OUTPUT_FILENAME --num_sums NUM_SIMS 
```
See the args in the file for more options such as controlling the number of objects in the scene and biasing pick locations to pick up objects.


### Training OP3
To train OP3 run
```
python exps/train_op3/train_op3.py --variant [stack, pickplace, cloth] --debug 0
```
where the variants are `stack` for single step block stacking, `pickplace` for multistep block stacking, and `cloth` for the real world evaluation on the robotic pushing dataset from [here](https://sites.google.com/berkeley.edu/robotic-interaction-datasets). These loads in parameters from `op3/exp_variants/variants.py` which can also be modified or extended. The preprocessed cloth dataset can be downloaded from [here](https://drive.google.com/file/d/1_eZE0BH5-NVkusQg5FZsFTP6v1kvfIk5/view?usp=sharing)


### Running MPC
To run visual mpc with a trained op3 model, for single-step block stacking run
```
python exps/stack_exps/mpc_stack.py -m <stack_model_params_file>
```
and for multi-step block stacking run,
```
python exps/pickplace_exps/mpc_pickplace.py -m <pickplace_model_params_file>
```

where the -m argument if the name of the model file trained previously. e in `op3/data/saved_models`. Pretrained models are provided in the appropiate saved_models directory such as OP3/exps/stack_exps/saved_models.

## Using a GPU
You can use a GPU by calling
```
import op3.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)
```
before launching the scripts.

If you are using `doodad` (see below), simply use the `use_gpu` flag:
```
run_experiment(..., use_gpu=True)
```

## Visualizing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `op3.launchers.conf.LOCAL_LOG_DIR`. Default name is `op3/data/logs`.
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.


You can visualize results with [viskit](https://github.com/vitchyr/viskit).
```bash
python viskit/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```
This `viskit` repo also has a few extra nice features, like plotting multiple Y-axis values at once, figure-splitting on multiple keys, and being able to filter hyperparametrs out.

## Launching jobs with `doodad`
The `run_experiment` function makes it easy to run Python code on Amazon Web
Services (AWS) or Google Cloud Platform (GCP) by using
[doodad](https://github.com/justinjfu/doodad/).

It's as easy as:
```
from op3.launchers.launcher_util import run_experiment

def function_to_run(variant):
    learning_rate = variant['learning_rate']
    ...

run_experiment(
    function_to_run,
    exp_prefix="my-experiment-name",
    mode='ec2',  # or 'gcp'
    variant={'learning_rate': 1e-3},
)
```
You will need to set up parameters in conf_private.py (see step one of Installation).
This requires some knowledge of AWS and/or GCP, which is beyond the scope of
this README.
To learn more, more about `doodad`, [go to the repository](https://github.com/justinjfu/doodad/).

## Credits
A lot of the coding infrastructure is based on [rlkit](https://github.com/vitchyr/rlkit).

The Dockerfile is based on the [OpenAI mujoco-py Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile).
