import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
import wandb
from sklearn.model_selection import train_test_split
from operator import itemgetter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib.colors as colors
from os.path import join
from root_path import ROOT


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_data_split(traj_dataset, split_ratio=[0.6, 0.2, 0.2], random_seed=42):
    """"

    traj_dataset: contains traj infor for all rzns
                    list of dictionaries. dictionary contains states, actions, rtgs
    returns:
    idx_split: test_idx, train_idx, val_idx
    set_split:
    """
    n_trajs =  len(traj_dataset)
    all_idx = [i for i in range(n_trajs)]
    # split all idx to train and (test+val)
    test_val_size = split_ratio[1] + split_ratio[2]
    train_idx, test_val_idx = train_test_split(all_idx, 
                                        test_size=test_val_size,
                                        random_state=random_seed,
                                        shuffle=True)
    # split (test+val into test and val)
    val_size = split_ratio[2]/test_val_size
    test_idx, val_idx = train_test_split(test_val_idx, test_size=val_size)
    idx_split = (test_idx, train_idx, val_idx)

    train_traj_set = itemgetter(*train_idx)(traj_dataset)
    test_traj_set = itemgetter(*test_idx)(traj_dataset)
    val_traj_set = itemgetter(*val_idx)(traj_dataset)
    
    set_split = (train_traj_set, test_traj_set, val_traj_set)

    return idx_split, set_split

class cgw_trajec_dataset(Dataset):
    def __init__(self, trajectories, context_len, rtg_scale, state_dim=None):
        """
        trajectories; 
        context_len:
        rtg_scale:
        state_dim: if = 5, then state is 'txyuv'
        """
  
        self.context_len = context_len
        # TODO: Change if not pkl file

        self.trajectories = trajectories
        total_n_trajs = len(trajectories)
        self.trajectories = [traj for traj in self.trajectories if traj['done']]
        print(f"\n Making dataset out of successful trajectories. \n \
                No. of successful trajs / total trajs = {len(self.trajectories)} / {total_n_trajs} \n")

        min_len = 10**6     #high intiialization to update later
        states = []
        # print(f" ****** Verify: len of first traj (in cgw)= {len(self.trajectories[0]['states'])} ")
        # print(f" ****** Verify: first traj (in cgw)= {self.trajectories[0]['states']} ")

        for traj in self.trajectories:
            traj_len = traj['states'].shape[0]
            min_len =  min(min_len, traj_len)
            states.append(traj['states'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        # normalize states
        for traj in self.trajectories:
            traj['states'] = (traj['states'] - self.state_mean) / self.state_std
        # print(f" ****** Verify: first normalized traj (in cgw)= {self.trajectories[0]['states']} ")



    def get_state_stats(self):
        return self.state_mean, self.state_std

    # TODO: Verify it returns the no. of trajectories
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['states'].shape[0]
        # print(f"****** VERIFY: traj_len = {traj_len}")
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




class cgw_trajec_test_dataset(Dataset):
    def __init__(self, trajectories, context_len, rtg_scale, train_stats):
        """
        train_stats: mean and std of states in training set
        """
        self.context_len = context_len
        self.trajectories = trajectories
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
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
       
        # IMPORTANT: normalize states with mean and std from training data
        self.state_mean_tr, self.state_std_tr = train_stats
        for traj in self.trajectories:
            traj['states'] = (traj['states'] - self.state_mean_tr) / self.state_std_tr



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
                    val_idx, val_traj_set,
                    num_eval_ep=None, # None: use all episodes in val set
                    max_test_ep_len=120,
                    state_mean=None, state_std=None, render=False,
                    comp_val_loss = False):

    eval_batch_size = 1  # required for forward pass
    if num_eval_ep == None:
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

    """
    The use of "with torch. no_grad()" is like a loop where every tensor 
    inside the loop will have requires_grad set to False. It means any tensor 
    with gradient currently attached with the current computational graph is now detached 
    from the current graph.
    """

    with torch.no_grad():
        op_traj_dict_list = []
        for i in range(num_eval_ep):
            # rzn = np.random.randint(env.n_rzns, size=1)[0]
            # ith element of val_idx is the rzn'th realization of the velocity field
            # the above also corresponds to the i'th element of the val_set
            rzn = val_idx[i]
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

                running_state, running_reward, done, info = env.step(act.cpu().numpy())
                # print(f"****** Verify:\n running_state =  info= {info}   \n done = {done}")
                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward
                episode_returns += running_reward

                if render:
                    env.render()
                if done:
                    if t+1 < env.nT :
                        states[0, t+1] = torch.from_numpy(running_state).to(device)
                        states[0, t+1] = (states[0, t+1] - state_mean) / state_std
                    # attention weights of last block
                    attention_weights = model.blocks[-1].attention.attention_weights
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
        # val_loss_list = compute_val_loss(val_traj_set, op_traj_dict_list, max_test_ep_len)
        # assert(len(val_loss_list) == num_eval_ep), print("length mismatch")
        # results['avg_val_loss'] = np.mean(val_loss_list)
        pass
    
    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    results['eval/success_ratio'] = success_count / num_eval_ep
    success_rtg = 0
    for dict_ in op_traj_dict_list:
        if dict_['success'] == True:
            success_rtg += dict_['episode_returns']
    if success_count > 0:
        results['eval/avg_returns_per_success'] = success_rtg / success_count
    else:
        results['eval/avg_returns_per_success'] = None

    return results, op_traj_dict_list

def compute_val_loss(val_traj_set, op_traj_dict_list, max_test_ep_len):
    """
    val_traj_set: list of dicts containing data from validation set
                    Note: val_traj_set[0]["actions"].shape = (lenght-till-done, 1)
    op_traj_dict_list: list of dicts containing data from model predictions on 
                        rzns from the validation set
                    Note: op_traj_dict_list[0]["actions"].shape = torch.Size([1, 120, 1])              
    """
    assert(len(val_traj_set)==len(op_traj_dict_list)), print(f"len mismatch: {len(val_traj_set)} != {len(op_traj_dict_list)}")
    n_val_trajs = len(val_traj_set)
    val_loss_list = []
    print("***** verify shape", val_traj_set[0]["actions"].shape) 
    tmp_tlen = val_traj_set[0]["actions"].shape[0]
    print("***** verify traj_len", tmp_tlen) 
    print("***** verify shape", op_traj_dict_list[0]["actions"].shape)
    print("***** verify items beyond traj_len", 
            op_traj_dict_list[0]["actions"][0,tmp_tlen-1:,0])

    # Need to pad data from val set
    for i in range(n_val_trajs):
        traj_len = val_traj_set[i]['actions'].shape[0]
        pad_len = max_test_ep_len - traj_len
        pad = np.zeros((pad_len,1))
        val_traj_actions = np.concatenate((val_traj_set[i]["actions"], pad), axis=0)
        pred_actions = op_traj_dict_list[i]["actions"].detach().numpy()[0,:,:]
        # print(val_traj_actions.shape, pred_actions.shape) 
        val_loss = np.square(pred_actions - val_traj_actions).mean(axis=0)
        # print(f"val_loss.shape = {val_loss.shape}") #(1,)
        val_loss_list.append(val_loss[0])
    
    return val_loss_list

def plot_vel_field(env,t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
    # Make modes the last axis
    Ui = np.transpose(env.Ui,(0,2,3,1))
    Vi = np.transpose(env.Vi,(0,2,3,1))
    vx_grid = env.U[t,:,:] + np.dot(Ui[t,:,:,:],env.Yi[t,r,:])
    vy_grid = env.V[t,:,:] + np.dot(Vi[t,:,:,:],env.Yi[t,r,:])
    vx_grid = np.flipud(vx_grid)
    vy_grid = np.flipud(vy_grid)
    Xs = np.arange(0,env.xlim) + (env.dxy/2)
    Ys = np.arange(0,env.ylim) + (env.dxy/2)
    X,Y = np.meshgrid(Xs, Ys)
    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
    plt.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)



def visualize_output(op_traj_dict_list, 
                        iter_i = 0, 
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        at_time=None,
                        color_by_time=True,
                        plot_flow=True,

                        ):
 
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "pred_traj_epoch_" + str(iter_i) + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Policy execution after {iter_i} epochs")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(op_traj_dict_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    if color_by_time:
        t_dones = []
        for op_traj_dict in op_traj_dict_list:
            t_dones.append(op_traj_dict['t_done'])
        vmin = min(t_dones)
        vmax = max(t_dones)
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)


    for idx,traj in enumerate(op_traj_dict_list):
        if idx in traj_idx:
            states = traj['states']
            actions = traj['actions']
            rtg = traj['rtg']
            t_done =  traj['t_done']
            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done

            if stats!=None:
                mean, std = stats
                states = (states*std) + mean
        
            # Plot sstates
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if color_by_time:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=scalarMap.to_rgba(t_done))
            else:
                plt.plot(states[0,:t_done+1,1], states[0,:t_done+1,2])

            if show_scatter:
                plt.scatter(states[0,:,1], states[0,:,2],s=0.5)

            # Plot policy at visites states
            _, nstates,_ = states.shape
            if plot_policy:
                for i in range(nstates):
                    plt.arrow(states[0,i,1], states[0,i,2], np.cos(actions[0,i,0]), np.sin(actions[0,i,0]))
    if color_by_time:
        cbar = plt.colorbar(scalarMap, label="Arrival Time")

    # plot target area and set limits
    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
        ax.add_patch(target_circle)
        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time)
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({"pred_traj_fig": wandb.Image(fname)})


    return fig


def viz_op_traj_with_attention(op_traj_dict_list, 
                        mode='a_a_attention',       #or 'a_s_attention'
                        stats=None, 
                        env=None, 
                        log_wandb=True, 
                        plot_policy=False,
                        traj_idx=None,      #None=all, list of rzn_ids []
                        show_scatter=False,
                        plot_flow=False,
                        at_time=None,
                        ):
 
    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + mode + "_val_" + ".png" 
    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"{mode}")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(op_traj_dict_list))]

    # print(f" ***** Verify: traj_idx = {traj_idx}")
    for idx,traj in enumerate(op_traj_dict_list):
        if idx in traj_idx:
            states = traj['states']
            actions = traj['actions']
            rtg = traj['rtg']
            t_done =  traj['t_done']
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done

            at_weights = traj['attention_weights'][0,0,:,:].cpu().detach().numpy()
            a_s_wts_scaled = scale_attention_rows(at_weights[2::3,1::3])
            a_a_wts_scaled = scale_attention_rows(at_weights[2::3,2::3])

            # print(f"******* Verify: visualize_op: states.shape= {states.shape}")
            if stats!=None:
                mean, std = stats
                states = (states*std) + mean
        
            # Plot states
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            if mode == 'a_a_attention':
                for t in range(t_done):
                    plt.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Reds(a_a_wts_scaled[t_done-1,t]))
            elif mode == 'a_s_attention':
                for t in range(t_done):
                    plt.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Greens(a_s_wts_scaled[t_done-1,t]))
            else:
                raise Exception("invalid argument for mode")

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
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
        ax.add_patch(target_circle)

        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time)

    plt.savefig(fname, dpi=300)
    wandbfname = "pred_trajs_with_" + mode
    if log_wandb:
        wandb.log({wandbfname: wandb.Image(fname)})

    plt.close()
    return fname


def visualize_input(traj_dataset, 
                    stats=None, 
                    env=None, 
                    log_wandb=True,
                    traj_idx=None,      #None=all, list of rzn_ids []
                    wandb_fname='input_traj_fig',
                    info_str='',
                    at_time=None,
                    color_by_time=True,
                    plot_flow=True,
                    ):
 
    print(" ---- Visualizing input ---- ")

    path = join(ROOT, "tmp/last_exp_figs/")
    fname = path + "input_traj"  + ".png"

    fig = plt.figure()
    plt.cla()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    title = "Input_traj_" + info_str
    plt.title(title)

    # print(f" **** Verify: len of first traj = {len(traj_dataset[0][1])}")
    # print(f" **** Verify: first traj = {traj_dataset[0][1]}")
    # print(f" **** Verify: first traj_mask = {traj_dataset[0][4]}")

    if stats!=None:
        print("===== Note: rescaling states to original scale for viz (in visualize_input)=====")

    if traj_idx==None:
        traj_idx = [k for k in range(len(traj_dataset))]
   
    if color_by_time:
        t_dones = []
        for idx, traj in enumerate(traj_dataset):
            if idx in traj_idx:
                timesteps, states, actions, returns_to_go, traj_mask = traj
                t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for idx, traj in enumerate(traj_dataset):
        if idx in traj_idx:
            timesteps, states, actions, returns_to_go, traj_mask = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
           
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done
           
           
            if stats != None:
                mean, std = stats
                states = (states*std) + mean
                states = states*(traj_mask.reshape(-1,1))

            if color_by_time:
                plt.plot(states[:t_done,1], states[:t_done,2], color=scalarMap.to_rgba(t_done) )
            else:
                plt.plot(states[:t_done,1], states[:t_done,2], )
            plt.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # print(f"bcords: {states[-1,1], states[-1,2]}")

    if env != None:
        plt.xlim([0, env.xlim])
        plt.ylim([0, env.ylim])
        # print("****VERIFY: env.target_pos: ", env.target_pos)
        target_circle = plt.Circle(env.target_pos, env.target_rad, color='r', alpha=0.3)
        ax.add_patch(target_circle)

        if plot_flow and at_time!=None:
            plot_vel_field(env,at_time)    
    plt.savefig(fname, dpi=300)

    if log_wandb:
        wandb.log({wandb_fname: wandb.Image(fname)})
    plt.cla()


def plot_attention_weights(weight_mat, set_idx=0, 
                                        scale_each_row=False, 
                                        cmap='bwr', 
                                        log_wandb = False,
                                        fname='attention_heatmap.png',
                                        info_string = '',
                                        wandb_fname = ''
                                        
                                        ):

    """
    weight_mat: weight matrix expected shape = 1(or 64),1,210,210 or B,N,T,T where T is 3*context_len
    idx: sample index of batch
    scale_each_row: scales each row INDEPENDENTLY to lie between 0 and 1 for visualization
    """
    plt.cla()
    plt.clf()
    weights = weight_mat.cpu().detach().numpy()
    shape = weights.shape
    
    # Plot attenetion scores for the ith trajectory/sample among the batch (for training batch)
    weights = weights[set_idx,0,:,:]
    
    scale_str = ''
    if scale_each_row:
        scale_str = '_scaled_rows_'
        weights = scale_attention_rows(weights)
        # for i in range(shape[2]):   # shape[2] is T (no. of rows)
        #     min_wi = np.min(weights[i,0:i+1])
        #     del_wi = np.max(weights[i,0:i+1]) - min_wi
        #     weights[i,0:i+1] -= min_wi
        #     if del_wi != 0:
        #         weights[i,0:i+1] /= del_wi

    fig, ax = plt.subplots()

    shw = ax.imshow(weights, cmap=mpl.colormaps[cmap])
    bar = plt.colorbar(shw)


    title =  scale_str + info_string + "setIdx-" +  str(set_idx)
    plt.title(title)
    # plt.figure(figsize=(12,10))
    # ax = sns.heatmap(weights, linewidth=0.05)
    plt.savefig(fname, dpi=600)
    if log_wandb:
        if wandb_fname == None:
            wandb_fname =  "attention map"  
        wandb.log({wandb_fname: wandb.Image(fname)})



def scale_attention_rows(at_weights):
    """
    at_weights: 2d matrix of attntion weights
    """
    shape = at_weights.shape
    # print(f"**** Verify type= {type(at_weights)}")
    # print(f"**** Verify shape= {shape}")

    assert(len(shape)==2), f"Invalid shape < {len(shape)} > of weight matrix"
    for i in range(shape[0]):   # shape[0] is T (no. of rows)
        min_wi = np.min(at_weights[i,0:i+1])
        del_wi = np.max(at_weights[i,0:i+1]) - min_wi
        at_weights[i,0:i+1] -= min_wi
        if del_wi != 0:
            at_weights[i,0:i+1] /= del_wi
    return at_weights