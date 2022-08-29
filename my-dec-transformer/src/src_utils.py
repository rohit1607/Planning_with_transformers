import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
import wandb

def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


class cgw_trajec_dataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):
        
        self.context_len = context_len
        # TODO: Change if not pkl file
        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.trajectories = [traj for traj in self.trajectories if traj['done']]
        print(f"***********CHECK: len(self.trajectories) ={len(self.trajectories)}")
        min_len = 10**6     #high intiialization to update later
        states = []
        for traj in self.trajectories:
            traj_len = traj['states'].shape[0]
            min_len =  min(min_len, traj_len)
            states.append(traj['states'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale
            
        # TODO: *** Verify correctness ***
        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['states'] = (traj['states'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['states'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['states'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['states'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask



def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=120,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"******** Verify: state_dim= {state_dim} ")
    print(f"******** Verify: act_dim= {act_dim} ")

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

    """
    The use of "with torch. no_grad()" is like a loop where every tensor 
    inside the loop will have requires_grad set to False. It means any tensor 
    with gradient currently attached with the current computational graph is now detached 
    from the current graph.
    """
    print(f"****** Verify: num_eval_ep = {num_eval_ep}")
    with torch.no_grad():
        op_traj_dict_list = []
        for _ in range(num_eval_ep):
            rzn = np.random.randint(env.n_rzns, size=1)[0]
            env.set_rzn(rzn)

            op_traj_dict = {}
            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)

            # init episode
            running_state = np.array(env.reset())
            running_reward = 0
            running_rtg = rtg_target / rtg_scale
            print(f"******** Verify: type(running_state)= {type(running_state)} ")
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

                if render:
                    env.render()
                if done:
                    break
            
            op_traj_dict['states'] = states.cpu()
            op_traj_dict['actions'] = actions.cpu()
            op_traj_dict['rtg'] = rewards_to_go.cpu()

            op_traj_dict_list.append(op_traj_dict)
    
    print(f"****** Verify; len(op_traj_dict_list) = {len(op_traj_dict_list)}")

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    return results, op_traj_dict_list


import matplotlib.pyplot as plt
def visualize_output(op_traj_dict_list, iter_i, 
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False
                        ):
 
    path = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/last_exp_figs/"
    fname = path + "pred_traj_epoch_" + str(iter_i) + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Policy execution after {iter_i} epochs")

    if traj_idx==None:
        traj_idx = [k for k in range(len(op_traj_dict_list))]

    print(f" ***** Verify: traj_idx = {traj_idx}")
    for idx,traj in enumerate(op_traj_dict_list):
        if idx in traj_idx:
            states = traj['states']
            actions = traj['actions']
            rtg = traj['rtg']
            print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if stats!=None:
                mean, std = stats
                states = (states*std) + mean
                print("===== Note: rescaling states to original scale for viz=====")
        
            # Plot sstates
            plt.plot(states[0,:,1], states[0,:,2])
            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=0.5)

            # Plot policy at visites states
            _, nstates,_ = states.shape
            
            if plot_policy:
                for i in range(nstates):
                    plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))
        
    # plot target area and set limits
    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        print("****VERIFY: env.target_pos: ", env.target_pos)
        target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
        ax.add_patch(target_circle)

    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"pred_traj_fig": wandb.Image(fname)})


    return fig

def visualize_input(traj_dataset, 
                    stats=None, 
                    env=None, 
                    log_wandb=True,
                    traj_idx=None,      #None=all, list of rzn_ids []

                    ):
 
    print(" ---- Visualizing input ---- ")
    path = "/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp/last_exp_figs/"
    fname = path + "input_traj"  + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Input trajectory")

    if traj_idx==None:
        traj_idx = [k for k in range(len(traj_dataset))]
    
    for idx, traj in enumerate(traj_dataset):
        if idx in traj_idx:
            timesteps, states, actions, returns_to_go, traj_mask = traj
            if stats != None:
                mean, std = stats
                states = (states*std) + mean
                print("===== Note: rescaling states to original scale for viz=====")
            print(" ---- -------------- ---- ")

            plt.plot(states[:,1], states[:,2], label='input_traj')


    if env != None:
        plt.xlim([0,env.xlim])
        plt.ylim([0, env.ylim])
        print("****VERIFY: env.target_pos: ", env.target_pos)
        target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
        ax.add_patch(target_circle)

    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"input_traj_fig": wandb.Image(fname)})
    plt.cla()