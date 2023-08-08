import argparse
import gym
import numpy as np
import os
import pickle
import algos
from logger import Logger, setup_logger
from env_util import make_env
from replaybuffer import ReplayBuffer
import d4rl
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

env_names = [
    'antmaze-large-diverse-v2', 'antmaze-medium-diverse-v2', 'antmaze-umaze-diverse-v2',
    'FetchReach-v1', 'FetchPush-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1',
    'kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General parameters
    parser.add_argument('--log_dir', default='./results/', type=str, help='Log directory')
    parser.add_argument("--env_name", default='antmaze-large-diverse-v2', type=str, choices=env_names)
    parser.add_argument("--dataset", default='./offline_data/antmaze/large', type=str, help='Directory of offline dataset')
    parser.add_argument("--seed", default=1, type=int, help='Set gym, pytorch and numpy seeds')
    parser.add_argument('--pretrain_latent_sg', default=True, type=str2bool, help='Pretrain(True)/load(False) latent subgoal model')
    parser.add_argument('--train_highlevel', default=True, type=str2bool, help='Train(Train)/load(False) high-level policy')
    parser.add_argument("--train_lowlevel", default=True, type=str2bool, help='Train(True)/Load(False) low-level policy')
    parser.add_argument('--latentmodel_path', default=None, type=str)
    parser.add_argument('--prior_path', default=None, type=str)
    parser.add_argument('--highpolicy_path', default=None, type=str)
    
    # Latent subgoal model pre-training
    parser.add_argument("--subgoal_period", default=50, type=int, help='Generating a subgoal every c-step')
    parser.add_argument("--latent_dim", default=2, type=int, help='The dimension of latent vector z')
    parser.add_argument('--beta', default=0.1, type=float, help='Weighting parameter')
    parser.add_argument('--pre_batch_size', default=100, type=int) 
    parser.add_argument('--pre_total_step', default=int(5e5), type=int)
    parser.add_argument('--pre_log_period', default=10000, type=int)
    parser.add_argument('--pre_save_period', default=int(5e5), type=int)

    # High-level subgoal generation training
    parser.add_argument('--high_batch_size', default=256, type=int) 
    parser.add_argument('--high_total_step', default=int(2e5), type=int)
    parser.add_argument('--high_log_period', default=10000, type=int)
    parser.add_argument('--high_save_period', default=int(2e5), type=int)
    parser.add_argument("--relabel_ratio", default=0.8, type=float, help='Relabeling goals with the ratio, otherwise keep original goal')
    parser.add_argument('--use_automatic_alpha_tuning', default=True, type=str2bool, help='Automatically tune alpha for KL loss')
    parser.add_argument('--target_divergence', default=1.0, type=float, help='Target divergence for automatic alpha tuning')
    parser.add_argument('--alpha', default=1.0, type=float, help='Constant alpha')
    parser.add_argument('--with_lagrange', default=True, type=str2bool, help='CQL parameter')
    parser.add_argument('--lagrange_thresh', default=5, type=float, help='CQL parameter')
    parser.add_argument('--alpha_prime', default=1.0, type=float, help='CQL [parameter')
    parser.add_argument('--tau', default=0.005, type=float, help='Soft update parameter for critic networks')
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--high_actor_lr', default=1e-4, type=float)
    parser.add_argument('--high_critic_lr', default=1e-4, type=float)

    ## Low-level goal reaching policy training
    parser.add_argument('--low_batch_size', default=512, type=int) 
    parser.add_argument('--low_total_step', default=int(5e5), type=int)
    parser.add_argument('--low_log_period', default=10000, type=int)
    parser.add_argument('--low_save_period', default=int(5e5), type=int)
    parser.add_argument('--low_relabel_strategy', default=True, type=str2bool, help='Relabel t<i<=t+c if True, else t<i<=T')
    parser.add_argument('--use_tanh', action='store_true', help='squashing action with tanh')
    parser.add_argument('--low_actor_lr', default=1e-4, type=float)
    parser.add_argument('--low_critic_lr', default=1e-4, type=float)

    args = parser.parse_args()

    # Setup Logging
    file_name = f'seed_{args.seed}'
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    variant = vars(args)
    variant.update(node=os.uname()[1])
    
    # Setup Environment
    env, env_info = make_env(args.env_name, args.seed)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Dataset
    dataset_file = os.path.join(args.dataset, 'buffer.pkl')
    dataset = pickle.load(open(dataset_file,'rb'))
    print("Loaded data from " + dataset_file)

    replay_buffer = ReplayBuffer(args.env_name, **env_info)        
    replay_buffer.load(dataset)

    # Pre-train or Load Latent Subgoal Model
    latent_sg = algos.LatentSubgoalModel(replay_buffer, **env_info, **vars(args))

    if args.pretrain_latent_sg:
        print(time.ctime(), 'Pre-training Latent Subgoal Model...')
        pretrain_logger = Logger()
        pretrain_folder_name = os.path.join(folder_name, 'latent_sg')
        setup_logger(logger=pretrain_logger, exp_prefix=os.path.basename(pretrain_folder_name), variant=variant, log_dir=pretrain_folder_name)
        latent_sg.train(folder_name, pretrain_logger, **vars(args))
    else:
        latent_sg.load(args.latentmodel_path, args.prior_path)
        print('Loaded Latent Subgoal Model from:', args.latentmodel_path)

    # Train or Load High-level Policy
    highmodel = algos.HighlevelModel(env, replay_buffer, latent_sg.latentmodel, latent_sg.prior, **env_info, **vars(args))

    if args.train_highlevel:
        print(time.ctime(), 'Training High-level Policy...')
        high_logger = Logger()
        high_folder_name = os.path.join(folder_name, 'highlevel')
        setup_logger(logger=high_logger, exp_prefix=os.path.basename(high_folder_name), variant=variant, log_dir=high_folder_name)
        highmodel.train(folder_name, high_logger, **vars(args))
    else:
        highmodel.load(args.highpolicy_path)
        print('Loaded High-level Policy from:', args.highpolicy_path)

    # Train Low-level Policy
    lowmodel = algos.LowlevelModel(env, replay_buffer, highmodel, **env_info, **vars(args))
    
    if args.train_lowlevel:
        low_logger = Logger()
        low_folder_name = os.path.join(folder_name, 'lowlevel')
        setup_logger(logger=low_logger, exp_prefix=os.path.basename(low_folder_name), variant=variant, log_dir=low_folder_name)
        print(time.ctime(), 'Training Low-level Policy...')
        lowmodel.train(folder_name, low_logger, **vars(args))
        