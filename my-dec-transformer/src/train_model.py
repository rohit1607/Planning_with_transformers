import argparse
# from asyncore import read
from logging import raiseExceptions
import os
import sys
import random
import csv
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gym
import gym_examples

from root_path import ROOT
from model_min_dt import DecisionTransformer
sys.path.insert(0, ROOT)
from utils.utils import read_cfg_file, log_and_viz_params
from src_utils import compute_val_loss, get_data_split, cgw_trajec_dataset, cgw_trajec_test_dataset, viz_op_traj_with_attention
from src_utils import evaluate_on_env, visualize_output, visualize_input, plot_attention_weights
import pickle
import wandb
import pprint
import imageio.v2 as imageio
wandb.login()





def train(args, cfg_name ):


    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")

    cfg = read_cfg_file(cfg_name=cfg_name)

    dataset_name = "DG3"
    wandb_exp_name = "my-dt-" + dataset_name + "__" + start_time_str
    wandb.init(project="my_decision_transformer",
        name = wandb_exp_name,
        config=cfg
        )
    cfg=wandb.config
    params2 = read_cfg_file(cfg_name=join(ROOT,cfg.params2_name))

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
    dataset_path = join(ROOT,cfg.dataset_path)
    dataset_name = cfg.dataset_name
    # saves model and csv in this directory
    log_dir = join(ROOT, cfg.log_dir)

    # training and evaluation device
    device = torch.device(cfg.device)
    
    at_pl_t = 50
    if args.quick_run:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        max_train_iters = 2
        num_updates_per_iter = 10
        num_eval_ep = 10
        at_pl_t = 4


    prefix = "my_dt_" + dataset_name

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"
    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    # IMP: preprocess (scales and divides into batches) data
    # load dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)
    
    # Split dataset
    idx_split, set_split = get_data_split(traj_dataset, split_tr_tst_val, split_ran_seed)
    train_traj_set, test_traj_set, val_traj_set = set_split
    test_idx, train_idx, val_idx = idx_split
    print(f"len(val_traj_set) = {len(val_traj_set)}")
    train_traj_dataset = cgw_trajec_dataset(train_traj_set, context_len, rtg_scale, state_dim=state_dim)
    train_traj_stats = (train_traj_dataset.state_mean, train_traj_dataset.state_std)
    print(f"train_stats = {train_traj_stats}")

    val_traj_dataset = cgw_trajec_test_dataset(val_traj_set, context_len, rtg_scale, train_traj_stats)

    train_traj_data_loader = DataLoader(
                            train_traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )    
    train_data_iter = iter(train_traj_data_loader)


    env = gym.make(env_name)
    env.setup(cfg, params2)

    assert(env.observation_space.shape[0] == state_dim), "Error: state_dim mismatch!"
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # print(f"act_dim = {act_dim}")


    # # visualise input
    # visualize_input(val_traj_dataset, stats=train_traj_stats, env=env, log_wandb=True)


    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            ).to(device)

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #                         optimizer,
    #                         # lambda steps: min((steps+1)/warmup_steps, 1)
    #                         lambda steps: min(1, 1)

    #                     )

    total_updates = 0
    action_loss=None
    i_train_iter=None
    eval_max_reward = float('-inf')

    # wandb.watch(model, log_freq=1, log="all", log_graph=True)
    # p_log = log_and_viz_params(model)

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()
        wandb.log({"loss": action_loss })
        
        for itr in range(num_updates_per_iter):
            #TODO: Find meaning of try/except here
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(train_data_iter)

            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim
            actions = actions.to(device)        # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)

            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                        )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            # Get the training_attention_weights for the first batch_size no. of trajectories
            if itr == 0:
                attention_weights_tr_list = []
                for i_bl in range(n_blocks):
                    attention_weights_tr_list.append(model.blocks[i_bl].attention.attention_weights)
            
            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            # scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())
            # p_log.log_params(device=device)
        
        # evaluate action accuracy
        # TODO: implement or edit 'evaluate_on_env' and 'get_d4rl_normalized_score'
        results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                    env, rtg_target, rtg_scale,
                                                    val_idx, val_traj_set,
                                                    num_eval_ep=num_eval_ep,
                                                    max_test_ep_len=max_eval_ep_len, 
                                                    state_mean = train_traj_stats[0], 
                                                    state_std = train_traj_stats[1],
                                                    comp_val_loss = comp_val_loss)

        # visualize output
        if i_train_iter%4 == 1:                        
            visualize_output(op_traj_dict_list, i_train_iter, stats=train_traj_stats, env=env, log_wandb=True)
            fname = join(ROOT,'tmp/attention_heatmaps/')
            fname += save_model_name[:-3] + '_' + 'trainId_' + '.png'
            norm_fname = join(ROOT, 'tmp/normalized_att_heatmaps/')
            norm_fname += save_model_name[:-3] + '_' + 'trainId_' + '.png'
            for b_id, attention_weights_tr in enumerate(attention_weights_tr_list):
                normalized_weights_tr= F.softmax(attention_weights_tr, dim=-1)
                # plot_attention_weights(attention_weights_tr, set_idx=0, scale_each_row=True, cmap='Blues',
                #                         log_wandb=True,fname=fname, info_string='_pre_sfmax_')
                info_string = f"train_itr-{i_train_iter}_BId-{b_id}_"
                plot_attention_weights(normalized_weights_tr, set_idx=0, scale_each_row=True, cmap='Reds',
                                        log_wandb=True,fname=norm_fname, info_string=info_string, wandb_fname='attention map(training)')
            # print(f"actions; \n {op_traj_dict_list[0]['actions']}")

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_avg_val_loss = results['avg_val_loss']
        success_ratio = results['eval/success_ratio']
        eval_avg_returns_per_success = results['eval/avg_returns_per_success']
        # eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100
        wandb.log({"avg_returns": eval_avg_reward,
                    "avg_episode_length": eval_avg_ep_len,
                    "avg_val_loss": eval_avg_val_loss,
                    "success_ratio": success_ratio,
                    "avg_returns_per_success": eval_avg_returns_per_success})

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        # TODO: write logs once code runs
        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n'  
               + "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' )
        #         "eval d4rl score: " + format(eval_d4rl_score, ".5f")
        #     )
        print(log_str)

        # TODO: write logs once code runs
        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    # eval_d4rl_score
                    ]

        csv_writer.writerow(log_data)

        # save model
        # TODO: save model and best metrics after completing evaluation
        if eval_avg_reward > eval_max_reward:
            eval_max_reward = eval_avg_reward
            print("saving current model at: " + save_model_path)

            best_avg_returns = eval_avg_reward
            best_avg_episode_length = eval_avg_ep_len
            best_success_ratio = success_ratio
            best_avg_returns_per_success = eval_avg_returns_per_success
            best_epoch = i_train_iter
            # "avg_val_loss"= eval_avg_val_loss

            torch.save(model.state_dict(), save_model_path)
            tmp_path = save_model_path[:-1]
            torch.save(model, tmp_path)


    wandb.run.summary["best_avg_returns"] = best_avg_returns
    wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    #  wandb.run.summary[avg_val_loss= eval_avg_val_loss,
    wandb.run.summary["best_success_ratio"] = best_success_ratio
    wandb.run.summary["best_avg_returns_per_success"] = best_avg_returns_per_success
    wandb.run.summary["best_epoch"] = best_epoch
    

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)
    
    # wandb.finish()
    # Load best model and visualize attention and trajs
    model = torch.load(tmp_path)
    results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                env, rtg_target, rtg_scale,
                                                val_idx, val_traj_set,
                                                # num_eval_ep,
                                                max_eval_ep_len, 
                                                state_mean = train_traj_stats[0], 
                                                state_std = train_traj_stats[1],
                                                comp_val_loss = comp_val_loss)
    movie_name = save_model_name + "_traj_with_attention_" 
    movie_path = join(ROOT,"tmp/attention_traj_videos/")
    aa_movie_sname = movie_path + movie_name + "aa.mp4" 
    as_movie_sname = movie_path + movie_name + "as.mp4"
    aa_writer = imageio.get_writer(aa_movie_sname, fps=1)
    as_writer = imageio.get_writer(as_movie_sname, fps=1)

     
    for t in range(1,at_pl_t,2):
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

    for i,op_traj_dict in enumerate(op_traj_dict_list):
        if i%100 == 0:
            attention_weights = op_traj_dict['attention_weights']
            normalized_weights = F.softmax(attention_weights, dim=-1)

            norm_fname = join(ROOT, 'tmp/normalized_att_heatmaps/')
            norm_fname += save_model_name[:-3] + '_' + 'valId_' + str(i) + ".png"

            info_string = f"Val_on_best_model-traj-{i}_BId-{-1}_"

            plot_attention_weights(normalized_weights, set_idx=0, scale_each_row=True, cmap='Reds',
                                        log_wandb=True,fname=norm_fname, info_string=info_string, wandb_fname='attention map(validation on trained model)')
"""

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------


"""






def sweep_train():

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")
    defaults = dict(dropout=0.2,)
   
    dataset_name = "DG3"
    wandb_exp_name = "my-dt-" + dataset_name + "__" + start_time_str
    wandb.init(project="my_decision_transformer",
        name = wandb_exp_name,
        config=defaults
        )
    cfg=wandb.config                # cfg is a dictionary without nested 'value' keys
    params2 = read_cfg_file(cfg_name=join(ROOT,cfg.params2_name))
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
    dataset_path = join(ROOT,cfg.dataset_path)
    dataset_name = cfg.dataset_name
    # saves model and csv in this directory
    log_dir = join(ROOT,cfg.log_dir)
    print(dataset_path, log_dir)

    # training and evaluation device
    device = torch.device(cfg.device)
    
    at_pl_t = 50
    if ARGS_QR:
        print("\n ---------- Modifying cfg params for quick run --------------- \n")
        max_train_iters = 2
        num_updates_per_iter = 10
        num_eval_ep = 10
        at_pl_t = 4


    prefix = "my_dt_" + dataset_name

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"
    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    # IMP: preprocess (scales and divides into batches) data
    # load dataset
    with open(dataset_path, 'rb') as f:
        traj_dataset = pickle.load(f)
    
    # Split dataset
    idx_split, set_split = get_data_split(traj_dataset, split_tr_tst_val, split_ran_seed)
    train_traj_set, test_traj_set, val_traj_set = set_split
    test_idx, train_idx, val_idx = idx_split
    print(f"len(val_traj_set) = {len(val_traj_set)}")
    train_traj_dataset = cgw_trajec_dataset(train_traj_set, context_len, rtg_scale, state_dim=state_dim)
    train_traj_stats = (train_traj_dataset.state_mean, train_traj_dataset.state_std)
    print(f"train_stats = {train_traj_stats}")
    # test_traj_dataset = cgw_trajec_test_dataset(test_traj_set, context_len, 
    #                                             rtg_scale, train_traj_stats)
    # val_traj_dataset = cgw_trajec_test_dataset(val_traj_set, context_len, 
    #                                             rtg_scale, train_traj_stats)                      

    train_traj_data_loader = DataLoader(
                            train_traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )    
    train_data_iter = iter(train_traj_data_loader)


    env = gym.make(env_name)
    env.setup(cfg, params2)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # print(f"act_dim = {act_dim}")

    # # visualise input - TODO: debug extra point in the middle
    # visualize_input(train_traj_dataset, stats=train_traj_stats, env=env, log_wandb=True)

    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            ).to(device)

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #                         optimizer,
    #                         # lambda steps: min((steps+1)/warmup_steps, 1)
    #                         lambda steps: min(1, 1)

    #                     )

    total_updates = 0
    action_loss=None
    i_train_iter=None
    eval_max_reward = float('-inf')
    # wandb.watch(model, log_freq=1, log="all", log_graph=True)
    # p_log = log_and_viz_params(model)

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()
        wandb.log({"loss": action_loss })
        
        for itr in range(num_updates_per_iter):
            #TODO: Find meaning of try/except here
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(train_data_iter)


            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim
            actions = actions.to(device)        # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)

            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                        )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            # Get the training_attention_weights for the first batch_size no. of trajectories
            if itr == 0:
                attention_weights_tr_list = []
                for i_bl in range(n_blocks):
                    attention_weights_tr_list.append(model.blocks[i_bl].attention.attention_weights)

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            # scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())
            # p_log.log_params(device=device)


        # evaluate action accuracy
        # TODO: implement 'get_d4rl_normalized_score'
        results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                    env, rtg_target, rtg_scale,
                                                    val_idx, val_traj_set,
                                                    num_eval_ep=num_eval_ep,
                                                    max_test_ep_len=max_eval_ep_len, 
                                                    state_mean = train_traj_stats[0], 
                                                    state_std = train_traj_stats[1],
                                                    comp_val_loss = comp_val_loss)

        # visualize output
        if i_train_iter%4 == 1:                        
            visualize_output(op_traj_dict_list, i_train_iter, stats=train_traj_stats, env=env, log_wandb=True)
            fname = join(ROOT,'tmp/attention_heatmaps/')
            fname += save_model_name[:-3] + '_' + 'trainId_' + '.png'
            norm_fname = join(ROOT, 'tmp/normalized_att_heatmaps/')
            norm_fname += save_model_name[:-3] + '_' + 'trainId_' + '.png'
            for b_id, attention_weights_tr in enumerate(attention_weights_tr_list):
                normalized_weights_tr= F.softmax(attention_weights_tr, dim=-1)
                # plot_attention_weights(attention_weights_tr, set_idx=0, scale_each_row=True, cmap='Blues',
                #                         log_wandb=True,fname=fname, info_string='_pre_sfmax_')
                info_string = f"train_itr-{i_train_iter}_BId-{b_id}_"
                plot_attention_weights(normalized_weights_tr, set_idx=0, scale_each_row=True, cmap='Reds',
                                        log_wandb=True,fname=norm_fname, info_string=info_string, wandb_fname='attention map(training)')
            # print(f"actions; \n {op_traj_dict_list[0]['actions']}")


        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_avg_val_loss = results['avg_val_loss']
        success_ratio = results['eval/success_ratio']
        eval_avg_returns_per_success = results['eval/avg_returns_per_success']
        # eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100
        wandb.log({"avg_returns": eval_avg_reward,
                    "avg_episode_length": eval_avg_ep_len,
                    "avg_val_loss": eval_avg_val_loss,
                    "success_ratio": success_ratio,
                    "avg_returns_per_success": eval_avg_returns_per_success})

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        # TODO: write logs once code runs
        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n'  
               + "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' )
        #         "eval d4rl score: " + format(eval_d4rl_score, ".5f")
        #     )
        print(log_str)

        # TODO: write logs once code runs
        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    # eval_d4rl_score
                    ]

        csv_writer.writerow(log_data)

        # save model
        # TODO: save model and best metrics after completing evaluation
        if eval_avg_reward > eval_max_reward:
            eval_max_reward = eval_avg_reward
            print("saving current model at: " + save_model_path)

            best_avg_returns = eval_avg_reward
            best_avg_episode_length = eval_avg_ep_len
            best_success_ratio = success_ratio
            best_avg_returns_per_success = eval_avg_returns_per_success
            best_epoch = i_train_iter
            # "avg_val_loss"= eval_avg_val_loss

            torch.save(model.state_dict(), save_model_path)
            tmp_path = save_model_path[:-1]
            torch.save(model, tmp_path)

    wandb.log({"best_avg_returns": best_avg_returns})
    wandb.run.summary["best_avg_returns"] = best_avg_returns
    wandb.run.summary["best_avg_episode_length"] = best_avg_episode_length
    #  wandb.run.summary[avg_val_loss= eval_avg_val_loss,
    wandb.run.summary["best_success_ratio"] = best_success_ratio
    wandb.run.summary["best_avg_returns_per_success"] = best_avg_returns_per_success
    wandb.run.summary["best_epoch"] = best_epoch
    

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)
    
  
    # wandb.finish()
    # Load best model and visualize attention and trajs
    model = torch.load(tmp_path)
    results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                env, rtg_target, rtg_scale,
                                                val_idx, val_traj_set,
                                                # num_eval_ep,
                                                max_eval_ep_len, 
                                                state_mean = train_traj_stats[0], 
                                                state_std = train_traj_stats[1],
                                                comp_val_loss = comp_val_loss)
    movie_name = save_model_name + "_traj_with_attention_" 
    movie_path = join(ROOT,"tmp/attention_traj_videos/")
    aa_movie_sname = movie_path + movie_name + "aa.mp4" 
    as_movie_sname = movie_path + movie_name + "as.mp4"
    aa_writer = imageio.get_writer(aa_movie_sname, fps=1)
    as_writer = imageio.get_writer(as_movie_sname, fps=1)

     
    for t in range(1,at_pl_t,2):
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

    for i,op_traj_dict in enumerate(op_traj_dict_list):
        if i%100 == 0:
            attention_weights = op_traj_dict['attention_weights']
            normalized_weights = F.softmax(attention_weights, dim=-1)

            norm_fname = join(ROOT, 'tmp/normalized_att_heatmaps/')
            norm_fname += save_model_name[:-3] + '_' + 'valId_' + str(i) + ".png"

            info_string = f"Val_on_best_model-traj-{i}_BId-{-1}_"

            plot_attention_weights(normalized_weights, set_idx=0, scale_each_row=True, cmap='Reds',
                                        log_wandb=True,fname=norm_fname, info_string=info_string, wandb_fname='attention map(validation on trained model)')


ARGS_QR = False
if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")


    parser = argparse.ArgumentParser()

    # parser.add_argument('--env', type=str, default='ContGrid-v5')
    # parser.add_argument('--dataset', type=str, default='medium')
    # parser.add_argument('--rtg_scale', type=int, default=1000)

    # parser.add_argument('--max_eval_ep_len', type=int, default=100)
    # parser.add_argument('--num_eval_ep', type=int, default=10)

    # parser.add_argument('--dataset_dir', type=str, default='data/')
    # parser.add_argument('--log_dir', type=str, default='dt_runs/')

    # parser.add_argument('--context_len', type=int, default=20)
    # parser.add_argument('--n_blocks', type=int, default=3)
    # parser.add_argument('--embed_dim', type=int, default=32)
    # parser.add_argument('--n_heads', type=int, default=1)
    # parser.add_argument('--dropout_p', type=float, default=0.1)

    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--wt_decay', type=float, default=1e-4)
    # parser.add_argument('--warmup_steps', type=int, default=10000)

    # parser.add_argument('--max_train_iters', type=int, default=200)
    # parser.add_argument('--num_updates_per_iter', type=int, default=100)

    # parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='single_run')
    parser.add_argument('--quick_run', type=bool, default=False)
    args = parser.parse_args()

    cfg_name = join(ROOT,"cfg/contGrid_v6.yaml")
    sweeep_cfg_name = join(ROOT,"cfg/contGrid_v6_sweep.yaml")
    print(f'args.mode = {args.mode}')

    if args.mode == 'single_run':
        print("----- beginning single_run ------")
        train(args, cfg_name)


    elif args.mode == 'sweep':
        print("----- beginning sweeeeeeeeeeeeep ------")
        ARGS_QR = args.quick_run
        print(f"ARGS_QR={ARGS_QR}")
        sweep_cfg = read_cfg_file(cfg_name=sweeep_cfg_name)
        sweep_id = wandb.sweep(sweep_cfg)
        wandb.agent(sweep_id, function=sweep_train)

