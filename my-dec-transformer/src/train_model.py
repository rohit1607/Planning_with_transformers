import argparse
# from asyncore import read
from logging import raiseExceptions
import os
import sys
import random
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import gym
import gym_examples

from model_min_dt import DecisionTransformer
sys.path.insert(0, '/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/')
from utils.utils import read_cfg_file, log_and_viz_params
from src_utils import cgw_trajec_dataset, evaluate_on_env, visualize_output, visualize_input
import tqdm
import wandb
wandb.login()





def train(mode, args, cfg_name, params2_name):


    env_name = "gym_examples/contGrid-v5"
    cfg_name = cfg_name
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%m-%d-%H-%M")

    if mode == 'args':
        dataset = args.dataset          # medium / medium-replay / medium-expert
        rtg_scale = args.rtg_scale      # normalize returns to go
        if args.env == "ContGrid_v5":
            env_name = ""
            rtg_target = 100
            dataset_name = ""

        else:
            raise NotImplementedError

        max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
        num_eval_ep = args.num_eval_ep          # num of evaluation episodes

        batch_size = args.batch_size            # training batch size
        lr = args.lr                            # learning rate
        wt_decay = args.wt_decay                # weight decay
        warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

        # total updates = max_train_iters x num_updates_per_iter
        max_train_iters = args.max_train_iters
        num_updates_per_iter = args.num_updates_per_iter

        context_len = args.context_len      # K in decision transformer
        n_blocks = args.n_blocks            # num of transformer blocks
        embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
        n_heads = args.n_heads              # num of transformer heads
        dropout_p = args.dropout_p          # dropout probability

        # TODO: put in path
        # load data from this file
        dataset_path = f'{args.dataset_dir}/{dataset_name}.pkl'

        # saves model and csv in this directory
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # training and evaluation device
        device = torch.device(args.device)

    elif mode == 'cfg':
        cfg = read_cfg_file(cfg_name=cfg_name)
        params2 = read_cfg_file(cfg_name=params2_name)
        dataset_name =cfg['train_params']['dataset_name']
        wandb_exp_name = "my-dt-" + dataset_name + "__" + start_time_str
        wandb.init(project="my_decision_transformer",
            name = wandb_exp_name,
            config=cfg
            )
        cfg=wandb.config
        train_ps = cfg['train_params']
        rtg_target = cfg['grid_params']['space_scale']

        max_eval_ep_len = train_ps['max_eval_ep_len']  # max len of one episode
        num_eval_ep = train_ps['num_eval_ep']          # num of evaluation episodes

        rtg_scale = train_ps['rtg_scale']
        batch_size = train_ps['batch_size']            # training batch size
        lr = train_ps['lr']                            # learning rate
        wt_decay = train_ps['wt_decay']                # weight decay
        warmup_steps = train_ps['warmup_steps']        # warmup steps for lr scheduler

        # total updates = max_train_iters x num_updates_per_iter
        max_train_iters = train_ps['max_train_iters']
        num_updates_per_iter = train_ps['num_updates_per_iter']

        context_len = train_ps['context_len']      # K in decision transformer
        n_blocks = train_ps['n_blocks']            # num of transformer blocks
        embed_dim = train_ps['embed_dim']          # embedding (hidden) dim of transformer
        n_heads = train_ps['n_heads']              # num of transformer heads
        dropout_p = train_ps['dropout_p']          # dropout probability

        # load data from this file
        dataset_path = train_ps['dataset_path']
        dataset_name = train_ps["dataset_name"]
        # saves model and csv in this directory
        log_dir = train_ps['log_dir']
        if not os.path.exists('log_dir'):
            os.makedirs('log_dir')

        # training and evaluation device
        device = torch.device(train_ps['device'])
        

    else:
        raise NotImplementedError
        



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
    traj_dataset = cgw_trajec_dataset(dataset_path, context_len, rtg_scale)
    # visualise input
    traj_stats = (traj_dataset.state_mean,traj_dataset.state_std)
    visualize_input(traj_dataset, stats=traj_stats)

    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )

    data_iter = iter(traj_data_loader)

    ## get state stats from dataset
    # TODO: implement 'get_state_stats' method in dataset extractor class
    state_mean, state_std = traj_dataset.get_state_stats()

    env = gym.make(env_name)
    env.setup(cfg, params2)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"act_dim = {act_dim}")


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

    max_score = -1.0
    total_updates = 0
    action_loss=None
    i_train_iter=None
    wandb.watch(model, log_freq=1, log="all", log_graph=True)
    # p_log = log_and_viz_params(model)

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()
        wandb.log({"loss": action_loss, "epochs":i_train_iter})
        
        for _ in range(num_updates_per_iter):
            #TODO: Find meaning of try/except here
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

            # print(f"CHECK: actions.shape = {actions.shape}")
            # print(f"CHECK: states.shape = {states.shape}")
            # print(f"CHECK: timesteps.shape = {timesteps.shape}")
            # print(f"CHECK: rtg.shape = {returns_to_go.shape}")

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
        results, op_traj_dict_list = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std)
        # visualize output
        if i_train_iter%20 == 0:                        
            visualize_output(op_traj_dict_list, i_train_iter, stats=traj_stats)
            print(f"actions; \n {op_traj_dict_list[0]['actions']}")
       
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        # eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

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
        # TODO: save model after completing evaluation
        print("max score: " + format(max_score, ".5f"))
        # if eval_d4rl_score >= max_d4rl_score:
        #     print("saving max d4rl score model at: " + save_best_model_path)
        #     torch.save(model.state_dict(), save_best_model_path)
        #     max_d4rl_score = eval_d4rl_score

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)
    
    torch.onnx.export(model, (timesteps,states, actions, returns_to_go), "model.onnx",
        #  input_names="states,actions,rtg,timesteps",
        #  output_names= "pred_actions"
         )
        
    wandb.save("model.onnx")
    wandb.finish()
    # p_log.visualize_wb(savefig= "../tmp/params3.png")


if __name__ == "__main__":

    print(f"cuda available: {torch.cuda.is_available()}")


    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='ContGrid-v5')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=100)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    cfg_name = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/cfg/contGrid_v5.yaml"
    params2_name ="/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/data/DG3/params.yml"
    train('cfg', args, cfg_name, params2_name)