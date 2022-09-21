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
from src_utils import compute_val_loss, get_data_split, cgw_trajec_dataset, plot_attention_weights, visualize_output
import tqdm
import pickle
import pprint
import seaborn as sns


# separate function for evaluating at different start state
def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    val_idx, val_traj_set, test_start_state,
                    # num_eval_ep=None,
                    max_test_ep_len=120,
                    state_mean=None, state_std=None, render=False,
                    comp_val_loss = False):

    eval_batch_size = 1  # required for forward pass
    num_eval_ep= len(val_idx)
    results = {}
    total_reward = 0
    total_timesteps = 0
    success_count = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # print(f"******** Verify: state_dim= {state_dim} ")
    # print(f"******** Verify: act_dim= {act_dim} ")
    print(" ======== evaluating model ================")
    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()


    with torch.no_grad():
        op_traj_dict_list = []
        for i in range(num_eval_ep):
            # rzn = np.random.randint(env.n_rzns, size=1)[0]
            # ith element of val_idx is the rzn'th realization of the velocity field
            # the above also corresponds to the i'th element of the val_set
            rzn = val_idx[i]
            env.set_rzn(rzn)
            # print(f"current state: {env.state}")

            op_traj_dict = {}
            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            # env.reset() resets in the scale (nT, xlim, ylim)
            running_state = np.array(env.reset(reset_state=test_start_state))
            running_reward = 0
            running_rtg = rtg_target / rtg_scale
            reached_target = False
            episode_returns = 0
            # print(f"******** Verify: type(running_state)= {type(running_state)} ")

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1])
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, _ = env.step(act.cpu().numpy())

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward
                episode_returns += running_reward

                if render:
                    env.render()
                if done:
                    states[0, t+1] = torch.from_numpy(running_state).to(device)
                    states[0, t+1] = (states[0, t+1] - state_mean) / state_std
                    attention_weights = model.blocks[0].attention.attention_weights
                    # if done and getting positive reward
                    if running_reward > 0: 
                        reached_target = True   
                        success_count += 1
                 
                    break
            
            op_traj_dict['states'] = states.cpu()
            op_traj_dict['actions'] = actions.cpu()
            op_traj_dict['rtg'] = rewards_to_go.cpu()
            op_traj_dict['t_done'] = t+1
            op_traj_dict['attention_weights'] = attention_weights
            op_traj_dict['success'] = reached_target
            op_traj_dict['episode_returns'] = episode_returns

            op_traj_dict_list.append(op_traj_dict)
    
    results['avg_val_loss'] = None
    if comp_val_loss:
        val_loss_list = compute_val_loss(val_traj_set, op_traj_dict_list, max_test_ep_len)
        assert(len(val_loss_list) == num_eval_ep), print("length mismatch")
        results['avg_val_loss'] = np.mean(val_loss_list)
    
    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    results['eval/success_ratio'] = success_count / num_eval_ep
   
    success_rtg = 0
    for dict_ in op_traj_dict_list:
        if dict_['success'] == True:
            success_rtg += dict_['episode_returns']
    results['eval/avg_returns_per_success'] = success_rtg / success_count

        
    return results, op_traj_dict_list



sweeep_cfg_name = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/cfg/contGrid_v5_sweep.yaml" 
sweeep_cfg = read_cfg_file(cfg_name=sweeep_cfg_name)
env_name = "gym_examples/contGrid-v5"

tmp_cfg = sweeep_cfg['parameters']
cfg = {}
for key in tmp_cfg.keys():
    try:
        cfg[key] = tmp_cfg[key]['value']
    except:
        cfg[key] = tmp_cfg[key]['values'][0]


print(cfg)
space_scale = cfg['space_scale'] 

params2 = read_cfg_file(cfg_name=cfg['params2_name'] )
rtg_target = cfg['rtg_target'] 

split_tr_tst_val = cfg['split_tr_tst_val'] 
split_ran_seed = cfg['split_ran_seed'] 

max_eval_ep_len = cfg['max_eval_ep_len']   # max len of one episode
num_eval_ep = cfg['num_eval_ep']        # num of evaluation episodes

rtg_scale = cfg['rtg_scale'] 
batch_size = cfg['batch_size']            # training batch size
lr = cfg['lr']                             # learning rate
wt_decay = cfg['wt_decay']                # weight decay
warmup_steps = cfg['warmup_steps']        # warmup steps for lr scheduler

# total updates = max_train_iters x num_updates_per_iter
max_train_iters = cfg['max_train_iters'] 
num_updates_per_iter = cfg['num_updates_per_iter'] 
comp_val_loss = cfg['comp_val_loss'] 

context_len = cfg['context_len']      # K in decision transformer
n_blocks = cfg['n_blocks']           # num of transformer blocks
embed_dim = cfg['embed_dim']           # embedding (hidden) dim of transformer
n_heads = cfg['n_heads']           # num of transformer heads
dropout_p = cfg['dropout_p']          # dropout probability

# load data from this file
dataset_path = cfg['dataset_path'] 
dataset_name = cfg['dataset_name'] 

device = torch.device(cfg['device'] )

model_path = '/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/log/my_dt_DG3_model_09-08-12-34.p'
model_name = model_path.split('/')[-1]
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

test_start_state = [20, 40, 50]

test_start_state_list = [[0, 20, 20],
                        #  [10, 20 ,20],
                        #  [20, 20 ,20],
                        #  [30, 40 ,50],
                        #  [40, 40, 50],
                        #  [0, 40, 20],
                        #  [0, 60 ,20],
                         ]



for test_start_state in test_start_state_list:
    results, op_traj_dict_list = evaluate_on_env(model, device, context_len, 
                                                env, rtg_target, rtg_scale,
                                                val_idx, val_traj_set, test_start_state,
                                                # num_eval_ep,
                                                max_eval_ep_len, 
                                                state_mean = train_traj_stats[0], 
                                                state_std = train_traj_stats[1],
                                                comp_val_loss = comp_val_loss)

    string = 'eval_st_' + str(test_start_state)
    visualize_output(op_traj_dict_list, string, stats=train_traj_stats, env=env, log_wandb=False)

    # print(f"results = {results}")
    pprint.pprint(results)

for i,op_traj_dict in enumerate(op_traj_dict_list):

    if i%100 == 0:
        attention_weights = op_traj_dict['attention_weights']
        normalized_weights = F.softmax(attention_weights, dim=-1)
        fname = '/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/attention_heatmaps/'
        fname += model_name[:-2] + '_' + 'valId_' + str(i) 

        norm_fname = '/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/normalized_att_heatmaps/'
        norm_fname += model_name[:-2] + '_' + 'valId_' + str(i) 

        plot_attention_weights(attention_weights, fname)
        plot_attention_weights(normalized_weights, norm_fname)




# for name, p in zip (model.named_parameters(), model.parameters()):
#     print(name, p.data)