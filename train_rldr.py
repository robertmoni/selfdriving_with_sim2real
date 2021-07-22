import logging
logger = logging.getLogger()
logger.setLevel(logging.WARN)

import os
from datetime import datetime
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import CSVLogger, TBXLogger
from env import launch_and_wrap_env
from env import launch_and_wrap_env
from utils import seed
from rllib_callbacks import on_episode_start, on_episode_step, on_episode_end, on_train_result
from rllib_loggers import TensorboardImageLogger


seed = 42

environment_config = {
    # Run mode allows to load different settings in different run modes. Available options: 'train', 'inference', 'debug'
    "mode": 'debug',
    # Length of an episode (if not terminated due to a failure)
    "episode_max_steps": 500,
    # The input image will be scaled to (height, width)
    "resized_input_shape" : '(84, 84)',
    # Crop the top part of the image
    "crop_image_top": 'true',  # The top 1/top_crop_divider part of the image will be cropped off. (E.g. 3 crops the top third of the image)
    "top_crop_divider": 3,
    # Convert the image to grayscale
    "grayscale_image": 'false',
    # Stack multiple frames as input
    "frame_stacking": 'true',
    # Number of frames to stack if frame stacking is enabled
    "frame_stacking_depth": 3,
    # Apply motion blur to the images during training
    "motion_blur": 'false',
    # Map the action space to a certain type. Available options are:
    # 'leftright', leftright_clipped, 'leftright_braking', steering_braking, 'discrete'
    # 'heading', 'heading_smooth', 'heading_trapz', 'heading_sine', 'heading_limited',
    "action_type": 'heading',
    # Overwrite the default reward function of Gym Duckietown ('default' leaves the default reward of the gym)
    # Available options: 'default', 'default_clipped', 'posangle', 'lane_distance'
    "reward_function": 'posangle',
    # Set Gym Duckietown's distortion parameter to generate fisheye distorted images
    "distortion": 'true',
    # How large ange deviation should be accepted when the robot is placed into the simulator
    "accepted_start_angle_deg": 4,
    "simulation_framerate": 20,
    # Skip frames in the agent-environment loop and only step the environment using the last action
    "frame_skip": 1,
    # Computed actions come into effect this much later in the time period of a step.
    # Allowed values: floats in the (0., 1.) interval or 'random' to get random values in each instance of the env
    "action_delay_ratio": 0.0,
    # Map(s) used during training. Individual map names could be specified or 'multimap1' to train on a custom map set
    "training_map": 'udem1',
    # Use Gym Duckietown's domain randomization
    "domain_rand": 'false',
    "dynamics_rand": 'false',
    "camera_rand": 'false',
    # If >0.0 a new observation/frame will be the same as the previous one, with a probability of frame_repeating
    "frame_repeating" : 0.0,
    # Spawn obstacles (duckies, duckiebots, etc.) to random drivable positions on a map (with or without fixed obstacles)
    # To spawn other duckiebots this option is not recommended, use spawn_forward_obstacle instead
    # Type and amount of spawned obstacles is determined by the dictionary under the 'obstacles' key
    "spawn_obstacles": 'false',
    "obstacles" :{
      # Keys at this level specify the type of object to be spawned.
      # Supported options: 'duckie', 'duckiebot', 'cone', 'barrier'
      "duckie" :{
          # Density: Amount of ducks per drivable tile. Can't be larger than one currently
          "density": 0.5,
          # Non-static objects move according to their default behaviour implemented in the gym Duckietown
          # Duckies "walk" back and forth (like crossing the road), duckiebots perform lane following.
          "static": 'true'},
      "duckiebot" :{
          "density": 0,
          "static": 'false'}},
    # Spawn a duckiebot in front of the controlled robot in every episode
    # Parameters of the robot are randomised, but settings are hardcoded in ForwardObstacleSpawnnigWrapper)
    "spawn_forward_obstacle": 'false',
    # Evaluators of the AI driving olympics use small steps to generate many frames which are overlayed to get blured images
    # The aido wrapper implements this blur, and also implements the same dynamics simulation
    #Warning: Using AIDOWrapper slows down the environment simulation (and therefore the training)!
    # Computing an observation can take 10-20 times longer as without it!
    "aido_wrapper": 'false',
    # RLlib only allows unknown keys in the env config -> important "global" keys are kept/copied here
    "wandb":{
      "project": 'duckietown-rllib'},
    "experiment_name": 'experiment',
    "seed": "0000",}

# %% [markdown]
# ## 1.2. RLlib config

# %%
ray_config = {
    "timesteps_total": 1.e+4,
    
    "ray_init_config" :{
#         "address": 127.0.0.1,
        "num_cpus": 17,
#         "webui_host": 127.0.0.1
    }
,
    # To load a trained model to continue training DO NOT USE THIS OPTION FOR PRETRAINING
    # -1 means no trained models are restored, training starts from random weights
    "restore_seed": -1,
    # If multiple trainings/experiments ran with the same seed
    "restore_experiment_idx": 0,
    # Every training saves two checkpoints the best (reward) and the final
    # If the final is the best, it's the only one
    # Otherwise the best checkpoint is saved earlier than the final, thus pretrained_checkpoint_idx: 0 --> best, 1 --> final
    # For trainings with rllib sweeps, grid_searches, the checkpoints from all trials are loaded from the same 'list'
    #   pretrained_checkpoint_idx: 0 --> 1st trial best checkpoint
    #   pretrained_checkpoint_idx: 1 --> 1st trial final checkpoint
    #   pretrained_checkpoint_idx: 2 --> 2st trial best checkpoint
    #   pretrained_checkpoint_idx: 4 --> 2st trial final checkpoint
    #   etc.
    # Pay attention to trials with only one checkpoint (last = best)
    "restore_checkpoint_idx": 0,



    # For debugging on systems with less ram and vram
    "debug_hparams":{
      "rllib_config":{
        "num_workers": 1,
        "num_gpus": 0},
    #    train_batch_size: 64
    #    sgd_minibatch_size: 32
    #    eager: true
    #    log_level: 'DEBUG'
    #    num_sgd_iter: 2
      "ray_init_config":{
        "num_cpus": 1,
        "memory": 2097152000, # 2000 * 1024 * 1024
        "object_store_memory": 209715200, # 200 * 1024 * 1024
        "redis_max_memory": 209715200, # 200 * 1024 * 1024
        "local_mode": True}},


    "inference_hparams": {
      "rllib_config": {
        "explore": 'false',
        "num_workers": 0,
        "num_gpus": 0,
        "callbacks": {}},
      "ray_init_config":{
        "num_cpus": 1,
        "memory": 2097152000, # 2000 * 1024 * 1024
        "object_store_memory": 209715200, # 200 * 1024 * 1024
        "redis_max_memory": 209715200, # 200 * 1024 * 1024
        "local_mode": True}}

}

# %% [markdown]
# ## 1.3 PPO config

# %%

ppo_config = {
    
  'env': 'Duckietown',
  'callbacks': {'on_episode_start': on_episode_start,
                'on_episode_step': on_episode_step,
                'on_episode_end': on_episode_end,
                'on_train_result': on_train_result},
  "env_config": environment_config,  
  # === RLlib common congfig ================================================
  # https://ray.readthedocs.io/en/latest/rllib-training.html#common-parameters
  # Number of rollout worker actors to create for parallel sampling. Setting
  # this to 0 will force rollouts to be done in the trainer actor.
  "num_workers": 8,
  # Default sample batch size (unroll length). Batches of this size are
  # collected from rollout workers until train_batch_size is met.
  # "sample_batch_size": 265,
  # Number of GPUs to allocate to the trainer process. This can be fractional
  # (e.g., 0.3 GPUs).
  "num_gpus": 0,
  # Training batch size, if applicable. Should be >= sample_batch_size.
  # Samples batches will be concatenated together to a batch of this size,
  # which is then passed to SGD.
  "train_batch_size": 4096,
  # Discount factor of the MDP.
  "gamma": 0.99,
  # The default learning rate.
  # Note, that scientific notation is only interpreted as a number if the . and the sign are included!
  "lr": 5.e-5,
  # Whether to write episode stats and videos to the agent log dir. This is
  # typically located in ~/ray_results.
  "monitor": 'false',
  # Evaluate with every `evaluation_interval` training iterations.
  # The evaluation stats will be reported under the "evaluation" metric key.
  # Note that evaluation is currently not parallelized, and that for Ape-X
  # metrics are already only reported for the lowest epsilon workers.
  "evaluation_interval": None, # 25
  # Number of episodes to run per evaluation period. If using multiple
  # evaluation workers, we will run at least this many episodes total.
  "evaluation_num_episodes": 2,
  # Typical usage is to pass extra args to evaluation env creator
  # and to disable exploration by computing deterministic actions.
  # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
  # policy, even if this is a stochastic one. Setting "explore=False" here
  # will result in the evaluation workers not using this optimal policy!
  "evaluation_config": {"monitor": 'true'},
  # This argument, in conjunction with worker_index, sets the random seed of
  # each worker, so that identically configured trials will have identical
  # results. This makes experiments reproducible.
  "seed": 1234,
  # === PPO-specific config =================================================
  # https://ray.readthedocs.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
  # The GAE(lambda) parameter.
  "lambda": 0.95,
  # Total SGD batch size across all devices for SGD. This defines the
  # minibatch size within each epoch.
  "sgd_minibatch_size": 128,
  # Coefficient of the value function loss. IMPORTANT: you must tune this if
  # you set vf_share_layers: True. (it's False by default).
  "vf_loss_coeff": 0.5,
   # Coefficient of the entropy regularizer.
  "entropy_coeff": 0.0,
  # PPO clip parameter.
  "clip_param": 0.2,
  # Clip param for the value function. Note that this is sensitive to the
  # scale of the rewards. If your expected V is large, increase this.
  "vf_clip_param": 0.2,
  # If specified, clip the global norm of gradients by this amount.
  "grad_clip": 0.5
}

# %% [markdown]
# # 2. Training setup
# %% [markdown]
# - Set logger
# %% [markdown]
# - Initialize Ray for training

# %%
ray.init(num_cpus = ray_config["ray_init_config"]["num_cpus"])


# %%
register_env('Duckietown', launch_and_wrap_env)


# %%
tune.run(PPOTrainer,
         stop={'timesteps_total': ray_config["timesteps_total"]},
         config=ppo_config,
         local_dir="./artifacts",
         checkpoint_at_end=True,
         trial_name_creator=lambda trial: trial.trainable_name,  # for PPO this will make experiment dirs start with PPO_
         name="ppo_DA",
         keep_checkpoints_num=1,
         checkpoint_score_attr="episode_reward_mean",
         checkpoint_freq=1,
         loggers=[CSVLogger, TBXLogger, TensorboardImageLogger],
#          verbose = 1,
         )