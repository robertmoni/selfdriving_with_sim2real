import argparse
import json
from termcolor import colored


def update_from_json(args, json_path):
    args_dict = vars(args)
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
        args_dict.update(json_dict)
        return args_dict


def dump_args_to_json(args, json_path):
    with open(json_path, "w") as json_file:
        json.dump(vars(args), json_file, indent=2)


def print_args(args):
    print("==============================================")
    print("Training config")
    for key, val in args.__dict__.items():
        print(" ", key, "=", colored(val, 'green'))
    print("==============================================")



def get_common_args(parser:argparse.ArgumentParser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--experiment_name", default="debug", type=str,
                        help='Experiment name used for Hyperdash and Tensorboard logging')
    parser.add_argument("--experiment_run_num", default=0, type=int,
                        help='Run number for the current experiment used for Tensorboard logging')
    parser.add_argument('--algo', dest='algo', choices=['td3', 'sac', 'a2c', 'ppo'], default='sac')
    parser.add_argument('--vae', dest='vae', action='store_true',
                        help='Use a VAE encoder to reduce the dimension of the observation space')
    parser.add_argument('--vae_cnn_extractor', dest='vae_cnn_extractor', action='store_true',
                        help='Use the pretrained weights of a VAE as the Stable Baselines CNN feature extractor')
    parser.add_argument('--vae_weight_file', dest='vae_weight_file', default='vae/models/vae-16-v0.1.pkl')
    parser.add_argument('--crop_image_top', dest='crop_image_top', action='store_true',
                        help='Crop the top part of the image, the amount of crop is hardcoded for now')
    parser.add_argument('--frame_stacking', dest='frame_stacking', action='store_true')
    parser.add_argument('--frame_stacking_depth', type=int, default=3,
                        help='Number of frames to stack if frame stacking is enabled')
    parser.add_argument('--action_type', dest='action_type', default='heading',
                        choices=['leftright', 'heading', 'heading_smooth', 'heading_trapz', 'leftright_braking', 'discrete'])
    parser.add_argument('--reward_function', dest='reward_function',
                        choices=['default', 'Almasip', 'Carracing', 'Startd', 'Posangle', 'Experimental'], default='Posangle',
                        help='Possible reward functions')
    parser.add_argument('--clip_reward', dest='clip_reward', action='store_true')
    parser.add_argument('--lane_penalty', dest='lane_penalty', action='store_true',
                        help='Penalize if agent moves to the left lane')
    parser.add_argument('--distortion', type=bool, default=True,
                        help='Set Duckietown gym\'s distortion parameter to generate fisheye distorted images')
    parser.add_argument('--accepted_start_angle_deg', type=float, default=4,
                        help="How large ange deviation should be accepted when the robot is placed into the simulator") # Default 4 before in earlier experiments this value was used
    parser.add_argument('--simulation_framerate', type=int, default=30)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--env_ecp_dir', type=str, help="Where to place rollouts", default="env_exp")
    parser.add_argument('--resized_input_shape', dest='resized_input_shape', default="(120, 160)", type=str,
                        help='The input image will be scaled to (height, widht) ')


    return parser


def get_training_args(parser:argparse.ArgumentParser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = get_common_args(parser)
    parser.add_argument('--rollouts', type=int, help="Number of rollouts", default=2000)
    parser.add_argument('--data_dir', type=str, help="Where to place rollouts", default="dataset")
    parser.add_argument('--seq_len', type=int, help="Length of a sequence", default=10)
    parser.add_argument("--start_timesteps", default=300, type=int,
                        help='For how many time steps a purely random policy is run for')
    parser.add_argument("--eval_freq", default=5000, type=int, help='How often (time steps) we evaluate')
    parser.add_argument("--validation_freq", default=25000, type=int,
                        help='How often (time steps) we validate the model on the ETHZ_autolab_technical_track')
    parser.add_argument("--max_timesteps", default=5e5, type=int, help='Max time steps to run environment for')
    parser.add_argument("--lr", dest='learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--expl_noise", default=0.2, type=float, help='Std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=64, type=int, help='Batch size for both actor and critic')
    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor')
    parser.add_argument("--tau", default=0.005, type=float, help='Target network update rate')
    parser.add_argument("--policy_noise", default=0.2, type=float,
                        help='Noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.5, type=float, help='Range to clip target policy noise')
    parser.add_argument("--policy_delay", default=2, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--env_timesteps", default=500, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int,
                        help='Maximum number of steps to keep in the replay buffer')
    parser.add_argument('--load_policy', default="", type=str,
                        help='Load a previously trained model when starting training')
    parser.add_argument('--exploration_noise_decay', dest='exploration_noise_decay', action='store_true')
    parser.add_argument('--entropy_coeff', dest='entropy_coeff', default='auto_0.1', help='Entropy coefficient for SAC')
    parser.add_argument('--training_map', dest='training_map', default='huge_loop', help='Map used during training.')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel workers used wit algorithms relying on them such as A2C or PPO')
    parser.add_argument('--domain_rand', type=bool, default=False, choices=[0, 1],
                        help='Use Duckietown gym\'s built in domain randomization (0: false, 1: true)')
    return parser.parse_args()


def get_test_args(parser:argparse.ArgumentParser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = get_common_args(parser)
    return parser.parse_args()


def get_vae_args(parser:argparse.ArgumentParser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--folder', help='Path to a folder containing images for training', type=str,
                        default='dataset/')
    parser.add_argument('-s', '--image_shape', dest='image_shape', default="(120, 160)", type=str,
                        help='The image will be scaled to (height, width)')
    parser.add_argument("--experiment_name", default="VAE-debug", type=str,
                        help='Experiment name used for Weights and Biases and Tensorboard logging')
    return parser

def check_image_shape_str(shape_str):
    image_shape = eval(shape_str)
    assert len(image_shape) == 2, 'Invalid input shape format. Use (height, width) format, including the ().'
    print("[config.args.py] - Images will be saved with shape: {}".format(image_shape))
    return image_shape