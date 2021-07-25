import os

# "Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &"
os.environ['DISPLAY'] = ':0'
import gym
import numpy as np
from PIL import Image
import logging
from rllib_utils.env import launch_and_wrap_env

from tqdm import tqdm
from args import get_training_args, print_args, dump_args_to_json
import pyvirtualdisplay
import subprocess


if __name__ == "__main__":
    # _display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
    #                                 size=(1400, 900))
    # # display is active
    # _ = _display.start()
    # print("DISPLAY == ", os.environ.get("DISPLAY"))
    
    ###########################################################
    # Argparse
    args = get_training_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.data_dir+'/train'):
        os.makedirs(args.data_dir+'/train')
    if not os.path.exists(args.data_dir+'/test'):
        os.makedirs(args.data_dir+'/test')


    ###########################################################
    # Print args + cuda availability

    print_args(args)
    logging.basicConfig()
    logger = logging.getLogger('gym-duckietown')
    logger.setLevel(logging.WARNING)
    # "LF-norm-zigzag", "LF-norm-loop",
    # MAPSETS = {'multimap':[ "LF-norm-small_loop", "LF-norm-techtrack.yaml"]}
    MAPSETS = {'multimap':[ "LF-norm-zigzag", "LF-norm-loop","LF-norm-small_loop", "LF-norm-techtrack"]}
    environment_config = {
        "mode": 'debug',
        "episode_max_steps": 500,
        "resized_input_shape" : '(640, 480)',
        "crop_image_top": True,  
        "top_crop_divider": 3,
        "grayscale_image": False,
        "frame_stacking": True,
        "frame_stacking_depth": 3,
        "motion_blur": False,
        "action_type": 'heading',
        "reward_function": 'posangle',
        "distortion": True,
        "accepted_start_angle_deg": 4,
        "simulation_framerate": 20,
        "frame_skip": 1,
        "action_delay_ratio": 0.0,
        "training_map": 'udem1',
        "domain_rand": False,
        "dynamics_rand": False,
        "camera_rand": False,
        "frame_repeating" : 0.0,
        "spawn_obstacles": False,
        "obstacles" :{
            "duckie" :{
                    "density": 0.5,
                    "static": True},
            "duckiebot" :{
                    "density": 0,
                    "static": False}},
        "spawn_forward_obstacle": False,
        "aido_wrapper": False,
        "wandb":{
            "project": 'duckietown-rllib'},
        "experiment_name": 'experiment',
        "seed": "0000",}

    with tqdm(total=len(MAPSETS['multimap'])*args.rollouts*args.seq_len) as pbar1:
        for env_id in enumerate(MAPSETS['multimap']):
            print("environment: {}".format(env_id[1]))
            environment_config["training_map"]=env_id[1]
            print("env_config :", environment_config)
            #env = gym.make(ENV_IDS[env_id[0]], domain_rand=False)
            
            env = launch_and_wrap_env(environment_config)


            # print("env_id is {} meaning {}".format(env_id[0], ENV_IDS[env_id[0]]))

            for i in range(args.rollouts):
                env.reset()
                # policy is not configured yet, using random policy now
                action = env.action_space.sample()
                # print("actions", a_rollout[:10])
                # print("shape", len(a_rollout))


                t = 0
                while True:

                    obs, _, _, _ = env.step(action)
                    rollout_cnt = args.rollouts * env_id[0] + i
                    np.save(os.path.join(args.data_dir+'/train', 'rollout_{}_{}'.format(rollout_cnt, t)),
                            np.array(obs, dtype=np.float32))
                    pbar1.update(1)
                    t += 1
                    if t == args.seq_len:
                        break
    
    print()
    # _ = _display.stop()
 