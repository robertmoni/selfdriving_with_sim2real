import os
import sys
sys.path.append('/selfdriving_with_sim2real')
# "Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &"
os.environ['DISPLAY'] = ':0'

import gym
import numpy as np
from PIL import Image
import logging
 
from rllib_utils.env import launch_and_wrap_env
from tqdm import tqdm
from rllib_utils.args import get_training_args, print_args, dump_args_to_json
import pyvirtualdisplay
import subprocess
from gym_duckietown.exceptions import NotInLane



if __name__ == "__main__":
    # _display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
    #                                 size=(1400, 900))
    # # display is active
    # _ = _display.start()
    # print("DISPLAY == ", os.environ.get("DISPLAY"))
    
    ###########################################################
    # Argparse
    args = get_training_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
 
    ###########################################################
    # Print args + cuda availability

    print_args(args)
    logging.basicConfig()
    logger = logging.getLogger('gym-duckietown')
    logger.setLevel(logging.WARNING)
    # "LF-norm-zigzag", "LF-norm-loop",
    # MAPSETS = {'multimap':[ "LF-norm-small_loop", "LF-norm-techtrack.yaml"]}
    MAPSETS = {'multimap':["huge_loop2", "map3", "map2","map1"]}
    environment_config = {
        "mode": 'debug',
        "episode_max_steps": 500,
        "resized_input_shape" : '(64, 64)',
        "crop_image_top": False,  
        "top_crop_divider": 1,
        "grayscale_image": False,
        "frame_stacking": False,
        "frame_stacking_depth": 0,
        "motion_blur": False,
        "action_type": 'heading',
        "reward_function": 'posangle',
        "distortion": True,
        "accepted_start_angle_deg": 4,
        "simulation_framerate": 20,
        "frame_skip": 3,
        "action_delay_ratio": 0.0,
        "training_map": 'multimap',
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
                # action = env.action_space.sample()
                # print("actions", a_rollout[:10])
                # print("shape", len(a_rollout))


                t = 0
                while True:
                    
                    # get parameters 
                    try:
                        lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
                    except NotInLane:
                        break
                    
                    distance_to_road_center = lane_pose.dist
                    angle_from_straight_in_rads = lane_pose.angle_rad
                
                    # PD controller parameters
                    k_p = 0.5
                    k_d = 5

                    # velocity = 0.5
                    action = (k_p * distance_to_road_center + k_d * angle_from_straight_in_rads)

                    obs, _, _, _ = env.step(action)
                    rollout_cnt = args.rollouts * env_id[0] + i
                    obs = Image.fromarray(np.uint8(obs*255))
                    obs.save(os.path.join(args.save_path, 'real_{}_{}.png'.format(rollout_cnt, t)))
                    pbar1.update(1)
                    t += 1
                    if t == args.seq_len:
                        break
    
    print()
    # _ = _display.stop()
 