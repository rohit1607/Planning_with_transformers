"""
Script to 
    - load a trained model and 
    - test it on some arbitrary initial starting point within the traj envelope

Notes:
    1. Starting in the middle of the envelope doesnt work well for any intial timestep
    2. Starting at / near the original start location 
        - works well if starting at the original start time
        - progressivly gets worse on starting at other time
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F

import gym
import gym_examples

from model_min_dt import DecisionTransformer
sys.path.insert(0, '/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/')
from utils.utils import read_cfg_file, log_and_viz_params
from src_utils import get_data_split, cgw_trajec_dataset, plot_attention_weights, visualize_output, evaluate_on_env
from src_utils import viz_op_traj_with_attention
import pickle
import pprint
import seaborn as sns
import wandb
from datetime import datetime
import imageio.v2 as imageio

wandb.login()


def visualize(model_path, cfg_name, params2_name):
    model_name = model_path.split('/')[-1]
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")

    cfg = read_cfg_file(cfg_name=cfg_name)
    params2 = read_cfg_file(cfg_name=params2_name)
    dataset_name = "DG3"
    wandb_exp_name = "viz_my-dt-" + dataset_name + "__" + model_name
    wandb.init(project="visualize_attention",
        name = wandb_exp_name,
        config=cfg
        )
    cfg=wandb.config
    pprint.pprint(cfg)

    rtg_target = cfg.rtg_target
    env_name = cfg.env_name
    state_dim = cfg.state_dim
    split_tr_tst_val = cfg.split_tr_tst_val
    split_ran_seed = cfg.split_ran_seed

    max_eval_ep_len = cfg.max_eval_ep_len  # max len of one episode
    num_eval_ep = cfg.num_eval_ep       # num of evaluation episodes

    rtg_scale = cfg.rtg_scale
    batch_size = cfg.batch_size           # training batch size
    lr = cfg.lr                            # learning rate
    wt_decay = cfg.wt_decay               # weight decay
    warmup_steps = cfg.warmup_steps       # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = cfg.max_train_iters
    num_updates_per_iter = cfg.num_updates_per_iter
    comp_val_loss = cfg.comp_val_loss

    context_len = cfg.context_len     # K in decision transformer
    n_blocks = cfg.n_blocks          # num of transformer blocks
    embed_dim = cfg.embed_dim          # embedding (hidden) dim of transformer
    n_heads = cfg.n_heads            # num of transformer heads
    dropout_p = cfg.dropout_p         # dropout probability

    # load data from this file
    dataset_path = cfg.dataset_path
    dataset_name = cfg.dataset_name
    # saves model and csv in this directory
    log_dir = cfg.log_dir
    if not os.path.exists('log_dir'):
        os.makedirs('log_dir')

    # training and evaluation device
    device = torch.device(cfg.device)


    model = torch.load(model_path)
    model.eval()


    print(model.parameters())


    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)

    # Split dataset
    idx_split, set_split = get_data_split(traj_dataset, split_tr_tst_val, split_ran_seed)
    train_traj_set, test_traj_set, val_traj_set = set_split
    test_idx, train_idx, val_idx = idx_split
    train_traj_dataset = cgw_trajec_dataset(train_traj_set, context_len, rtg_scale)
    train_traj_stats = (train_traj_dataset.state_mean, train_traj_dataset.state_std)
    print(f"train_stats = {train_traj_stats}")

    env = gym.make(env_name)
    env.setup(cfg, params2)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"act_dim = {act_dim}")



    results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                env, rtg_target, rtg_scale,
                                                val_idx, val_traj_set,
                                                # num_eval_ep,
                                                max_eval_ep_len, 
                                                state_mean = train_traj_stats[0], 
                                                state_std = train_traj_stats[1],
                                                comp_val_loss = comp_val_loss)

    movie_name = model_name + "_traj_with_attention_" 
    movie_path = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/attention_traj_videos/"
    aa_movie_sname = movie_path + movie_name + "aa.mp4" 
    as_movie_sname = movie_path + movie_name + "as.mp4"
    aa_writer = imageio.get_writer(aa_movie_sname, fps=1)
    as_writer = imageio.get_writer(as_movie_sname, fps=1)

    for t in range(1, 50,2):
        aa_fname = viz_op_traj_with_attention(op_traj_dict_list, 
                                    mode='a_a_attention', 
                                    stats=train_traj_stats, 
                                    env=env, 
                                    log_wandb=False,
                                    at_time=t,
                                    plot_flow=True,
                                    )
        aa_writer.append_data(imageio.imread(aa_fname))

        as_fname = viz_op_traj_with_attention(op_traj_dict_list, 
                                    mode='a_s_attention', 
                                    stats=train_traj_stats, 
                                    env=env, 
                                    log_wandb=False,
                                    at_time=t,
                                    plot_flow=True,

                                    )
        as_writer.append_data(imageio.imread(as_fname))
   
    aa_writer.close()
    as_writer.close()
    wandb.log({"aa_attention_traj": wandb.Video(aa_movie_sname, format='mp4')})
    wandb.log({"as_attention_traj": wandb.Video(as_movie_sname, format='mp4')})



cfg_name = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/cfg/contGrid_v6.yaml"
params2_name ="/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/data/DG3/params.yml"
model_path = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/log/my_dt_DG3_model_09-21-10-51.p"
visualize(model_path,cfg_name,params2_name)
