import os
#from os.path import join, exists
import gym
import numpy as np
from PIL import Image
import logging
from env import launch_and_wrap_env

from tqdm import tqdm
from args import get_training_args, print_args, dump_args_to_json

if __name__ == "__main__":

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

    MAPSETS = {'multimap':["LF-norm-zigzag", "LF-norm-loop", "multitrack", "huge_loop"]}
    
    environment_config = {
    "seed": "0000",
    "training_map": 'multimap',
    "episode_max_steps": 500,
    "domain_rand": 'true',
    "dynamics_rand": 'true',
    "camera_rand": 'true',
    "accepted_start_angle_deg": 4,
    "distortion": 'true',
    "simulation_framerate": 30,
    "frame_skip": 1,
    }
    

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