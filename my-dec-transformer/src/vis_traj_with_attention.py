"""
Script to 
    - visualise attention weights on trajectories
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
from root_path import ROOT
from os.path import join

from model_min_dt import DecisionTransformer
sys.path.insert(0, ROOT)
from utils.utils import read_cfg_file, log_and_viz_params
from src_utils import get_data_split, cgw_trajec_dataset, plot_attention_weights, visualize_output, evaluate_on_env
from src_utils import viz_op_traj_with_attention, cgw_trajec_test_dataset, visualize_input
import pickle
import pprint
import seaborn as sns
import wandb
from datetime import datetime
import imageio.v2 as imageio

wandb.login()

def evaluate_model(cfg, model_path):

    params2 = read_cfg_file(cfg_name=join(ROOT,cfg["params2_name"]))

    pprint.pprint(cfg)
    rtg_target = cfg["rtg_target"]
    env_name = cfg["env_name"]
    state_dim = cfg["state_dim"]
    split_tr_tst_val = cfg["split_tr_tst_val"]
    split_ran_seed = cfg["split_ran_seed"]

    max_eval_ep_len = cfg["max_eval_ep_len"]  # max len of one episode
    num_eval_ep = cfg["num_eval_ep"]       # num of evaluation episodes

    rtg_scale = cfg["rtg_scale"]
    batch_size = cfg["batch_size" ]          # training batch size
    lr = cfg["lr"]                           # learning rate
    wt_decay = cfg["wt_decay"]               # weight decay
    warmup_steps = cfg["warmup_steps"]       # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = cfg["max_train_iters"]
    num_updates_per_iter = cfg["num_updates_per_iter"]
    comp_val_loss = cfg["comp_val_loss"]

    context_len = cfg["context_len"]     # K in decision transformer
    n_blocks = cfg["n_blocks"]          # num of transformer blocks
    embed_dim = cfg["embed_dim"]          # embedding (hidden) dim of transformer
    n_heads = cfg["n_heads"]            # num of transformer heads
    dropout_p = cfg["dropout_p"]         # dropout probability

    # load data from this file
    dataset_path = join(ROOT,cfg["dataset_path"])
    dataset_name = cfg["dataset_name"]
    # saves model and csv in this directory
    log_dir = join(ROOT,cfg["log_dir"])

    # training and evaluation device
    device = torch.device(cfg["device"])
    

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

    val_traj_dataset = cgw_trajec_test_dataset(val_traj_set, context_len, rtg_scale, train_traj_stats)
    # visualize_input(val_traj_dataset, stats=train_traj_stats,
    #                  env=env, log_wandb=True, info_str='val_set',
    #                 wandb_fname="Input_validation_trajs")


    results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                env, rtg_target, rtg_scale,
                                                val_idx, val_traj_set,
                                                num_eval_ep = num_eval_ep,
                                                max_test_ep_len = max_eval_ep_len, 
                                                state_mean = train_traj_stats[0], 
                                                state_std = train_traj_stats[1],
                                                comp_val_loss = comp_val_loss)

    return env, train_traj_dataset, train_traj_stats, val_traj_dataset, op_traj_dict_list


def visualize(cfgs_n_model_paths, paper_plot_info=None):
    (cfg_name, model_path), (txy_cfg_name,txy_model_path) = cfgs_n_model_paths

    model_name = model_path.split('/')[-1]
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")

    cfg = read_cfg_file(cfg_name=cfg_name)
    dataset_name = cfg['dataset_name']
    wandb_exp_name = "viz_my-dt-" + dataset_name + "__" + model_name
    wandb.init(project="visualize_attention",
        name = wandb_exp_name,
        config=cfg
        )
    cfg=wandb.config
    env,train_traj_dataset, train_traj_stats, val_traj_dataset, op_traj_dict_list = evaluate_model(cfg,model_path)
    
    txy_cfg = read_cfg_file(cfg_name=txy_cfg_name)
    env2, train_traj_dataset2, train_traj_stats2, val_traj_dataset2, op_traj_dict_list2 = evaluate_model(txy_cfg,txy_model_path)

    from paper_plots import paper_plots
    if paper_plot_info != None:
        model_name = model_path.split('/')[-1]
        print(f"model_name = {model_name}")
        save_dir = "paper_plots/"  + model_name
        save_dir = join(ROOT,save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        pp = paper_plots(env, op_traj_dict_list, train_traj_stats,
                         paper_plot_info=paper_plot_info,
                         save_dir=save_dir)
        # pp.plot_vel_mmoc()
        # pp.plot_loss_returns()
        pp.plot_traj_by_arr(train_traj_dataset)
        pp.plot_val_ip_op3_op5(train_traj_stats,val_traj_dataset,
                                train_traj_stats2, op_traj_dict_list2)
        # pp.plot_traj_by_arr(val_traj_dataset, set_str="_val")
        # pp.plot_att_heatmap(100)
        # pp.plot_traj_by_att("a_a_attention")
        # pp.plot_traj_by_att("a_s_attention")


    sys.exit()
    movie_name = model_name + "_traj_with_attention_" 
    movie_path = join(ROOT,"tmp/attention_traj_videos/")
    aa_movie_sname = movie_path + movie_name + "aa.mp4" 
    as_movie_sname = movie_path + movie_name + "as.mp4"
    aa_writer = imageio.get_writer(aa_movie_sname, fps=1)
    as_writer = imageio.get_writer(as_movie_sname, fps=1)
   
    visualize_output(op_traj_dict_list, stats=train_traj_stats, env=env, log_wandb=True,
                                        plot_flow=True,
                                        color_by_time=True,
                                        )


    for t in range(1, 50,40):
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


# TODO: save cfg files with models.
cfg_name = join(ROOT,"archive/contGrid_v6_HW.yaml")
model_path = join(ROOT,"log/my_dt_HW_v2_tmp_model_10-18-00-54.p")
txy_cfg_name = join(ROOT,"archive/contGrid_v5_HW.yaml")
txy_model_path = join(ROOT,"log/my_dt_HW_v2_tmp_model_10-18-01-08.p")
cfgs_n_model_paths = [(cfg_name, model_path), (txy_cfg_name,txy_model_path)]
paper_plot_info = {"trajs_by_arr": {"fname":"T_arr"},
                    "trajs_by_att": {"ts":[17,31,46],"fname":"att"},
                    "att_heatmap":{"fname":"heatmap"},
                    "trajs_ip_op3_op5":{"fname":"val_ip_op3_op5"},
                    "loss_avg_returns":{"fname":"loss"},
                    "vel_field":{"fname":"vel_field", "ts":[1,60,119]}
                    }
visualize(cfgs_n_model_paths, paper_plot_info=paper_plot_info)
