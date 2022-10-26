import os
import sys
from wsgiref.headers import tspecials
import numpy as np
import random
import torch

from root_path import ROOT
from os.path import join

sys.path.insert(0, ROOT)
from utils.utils import read_cfg_file, log_and_viz_params
from src_utils import get_data_split, cgw_trajec_dataset, plot_attention_weights, visualize_output, evaluate_on_env
from src_utils import viz_op_traj_with_attention, cgw_trajec_test_dataset, visualize_input
import seaborn as sns
import wandb
from datetime import datetime
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcol
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pickle
from src_utils import scale_attention_rows

class paper_plots:
    def __init__(self, env,  op_traj_dict_list, stats, paper_plot_info, non_dim_plots=True, save_dir='../tmp/'):
        self.env = env
        self.op_traj_dict_list = op_traj_dict_list
        self.stats = stats  #training mean and variance for normalization
        self.paper_plot_info = paper_plot_info
        self.save_dir = save_dir
        self.non_dim_plots = non_dim_plots

    def plot_all(self):
        self.plot_traj_by_arr()
        self.plot_traj_att()
        self.plot_att_heatmap()


    def plot_vel_field(self,ax, t,r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
        # Make modes the last axis
        Ui = np.transpose(self.env.Ui,(0,2,3,1))
        Vi = np.transpose(self.env.Vi,(0,2,3,1))
        vx_grid = self.env.U[t,:,:] + np.dot(Ui[t,:,:,:],self.env.Yi[t,r,:])
        vy_grid = self.env.V[t,:,:] + np.dot(Vi[t,:,:,:],self.env.Yi[t,r,:])
        vx_grid = np.flipud(vx_grid)
        vy_grid = np.flipud(vy_grid)
        Xs = np.arange(0,self.env.xlim) + (self.env.dxy/2)
        Ys = np.arange(0,self.env.ylim) + (self.env.dxy/2)
        X,Y = np.meshgrid(Xs, Ys)
        ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
        v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
        im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)
        return im


    def plot_traj_by_arr(self, traj_dataset, set_str=""):
        info = self.paper_plot_info["trajs_by_arr"]
        t_dones = []
        for traj in traj_dataset:
            _,_,_,_, traj_mask = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # vmax = 51
        # print(f"---- Info: tdone: min={vmin}, max={vmax} ")
        plt.hist(t_dones)
        plt.savefig("../tmp/tdone-hist")
        fig, ax = plt.subplots()

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax, r=199)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
           
            mean, std = self.stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # if idx>10:
            #     break


        cbar_fontsize = 12
        cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1, 3)])
        cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
        
        cbarv = fig.colorbar(im, ax=ax)
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        fname = info["fname"] + set_str
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        return


    def plot_val_ip_op3_op5(self, stats, traj_dataset, 
                            txy_stats, txy_op_traj_dict_list, 
                            at_time=None):
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15,5))

        info = self.paper_plot_info["trajs_ip_op3_op5"]
        t_dones = []
        for traj in traj_dataset:
            _,_,_,_, traj_mask = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data.
            t_dones.append(t_done)
        vmin = min(t_dones)
        vmax = max(t_dones)
        # vmax = 51

        # Make a user-defined colormap.
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('YlOrRd')
        sm = cm.ScalarMappable(norm=cNorm, cmap=cmap)

        ax = axs[0]
        self.setup_ax(ax)       
        im = self.plot_vel_field(ax,t=vmax,r=199)
        # traj_dataset=random.shuffle(traj_dataset)
        for idx, traj in enumerate(traj_dataset):
            timesteps, states, actions, returns_to_go, traj_mask = traj
            t_done = int(np.sum(traj_mask.numpy()))   # no. of points to plot. No need to plot masked data
           
            mean, std = stats
            states = (states*std) + mean
            states = states*(traj_mask.reshape(-1,1))

            ax.plot(states[:t_done,1], states[:t_done,2], color=sm.to_rgba(t_done), alpha=1 )
            ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)
            # if idx>10:
            #     break

        ax= axs[1]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field(ax,t=vmax,r=199)
        for idx, traj in enumerate(txy_op_traj_dict_list):
            # print(traj['states'])
            states = traj['states']
            t_done =  traj['t_done']

            mean, std = txy_stats
            states = (states*std) + mean

            ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
            # ax.scatter(states[-1,1], states[-1,2], alpha=0.5, zorder=10000, s=5)

        ax = axs[2]
        self.setup_ax(ax, show_ylabel=False)
        im = self.plot_vel_field(ax,t=vmax,r=199)
        for idx,traj in enumerate(self.op_traj_dict_list):
            states = traj['states']
            t_done =  traj['t_done']
 
            mean, std = stats
            states = (states*std) + mean
        
            # Plot sstates
            # shape: (eval_batch_size, max_test_ep_len, state_dim)
            ax.plot(states[0,:t_done+1,1], states[0,:t_done+1,2], color=sm.to_rgba(t_done))
                

        # cbar_fontsize = 12
        # cbar = fig.colorbar(sm, ax=ax, ticks=[i for i in range(vmin, vmax+1)])
        # cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
        
        # cbarv = fig.colorbar(im, ax=ax)
        # cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)

        cbar_fontsize = 12
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Arrival Time (non-dim units)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.65 )
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)


        fname = info["fname"] 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)







    def plot_att_trajs_at_t(self,ax,at_time,mode,stats):
        for idx,traj in enumerate(self.op_traj_dict_list):
            states = traj['states']
            t_done =  traj['t_done']
            # print(f"-------- verify : states= {states}")
            if at_time != None:
                assert(at_time >= 1), f"Can only plot at_time >= 1 only"
                # if at_time > t_done, just plot for t_done
                t_done = min(at_time, t_done)
            else:
                at_time = t_done
           
            # Rescale
            mean, std = stats
            states = (states*std) + mean

            at_weights = traj['attention_weights'][0,0,:,:].cpu().detach().numpy()
            a_s_wts_scaled = scale_attention_rows(at_weights[2::3,1::3])
            a_a_wts_scaled = scale_attention_rows(at_weights[2::3,2::3])

            alpha = 0.7
            ax.scatter(states[0,at_time,1], states[0,at_time,2], c='k', marker='p')
            if mode == 'a_a_attention':
                for t in range(t_done):
                    im =ax.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Reds(a_a_wts_scaled[t_done-1,t]), zorder=10, alpha=alpha)
            elif mode == 'a_s_attention':
                for t in range(t_done):
                    im = ax.plot(states[0,t:t+2,1], states[0,t:t+2,2], 
                                c=cm.Greens(a_s_wts_scaled[t_done-1,t]), zorder=10, alpha=alpha)
            else:
                raise Exception("invalid argument for mode")
        return im

    def plot_traj_by_att(self, mode):
        if mode == 'a_a_attention':
            cmap = cm.Reds
        elif mode == 'a_s_attention':
            cmap = cm.Greens
        else:
            raise Exception("invalid argument for mode")
        info = self.paper_plot_info["trajs_by_att"]
        nplots = len(info["ts"])
       
        fig, axs = plt.subplots(1,nplots, sharey=True, figsize=(15,5))
        # fig.suptitle('')    #title of overall plot
        show_ylabel = True
        for i in range(nplots):
            if i>=1:
                show_ylabel=False
            ax = axs[i]
            at_time = info["ts"][i]
            self.setup_ax(ax,show_ylabel=show_ylabel)
            self.plot_att_trajs_at_t(ax, at_time, mode, self.stats)
            im = self.plot_vel_field(ax,at_time,g_strmplot_lw=0.5, g_strmplot_arrowsize=0.5)
            ax.set_title(f"t={at_time}")
        
        # colorbars
        # cbar1_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar_fontsize = 12
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Attention Weights (scaled)", fontsize=cbar_fontsize)
     
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.65)
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        fname = info["fname"] + "_" + mode
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches = 'tight', dpi=600)
        return


    def setup_ax(self, ax, show_xlabel= True, 
                            show_ylabel=True, 
                            show_states=True,
                            show_xticks=True,
                            show_yticks=True,):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0,self.env.xlim])
        ax.set_ylim([0,self.env.ylim])
        ax.xaxis.set_ticks(np.arange(0,self.env.xlim +25,25))
        ax.yaxis.set_ticks(np.arange(0,self.env.xlim +25,25))

        if not show_xticks:
            ax.tick_params(axis='x',       
                            which='both',      
                            bottom=False,      
                            labelbottom=False)
        if not show_yticks:
            ax.tick_params(axis='y',       
                    which='both',      
                    left=False,      
                    labelleft=False)
        
        xlabel = f"X "
        ylabel = f"Y "
        if self.non_dim_plots == True:
            xlabel += "(Non-Dim)"
            ylabel += "(Non-Dim)"
        if show_xlabel:
            ax.set_xlabel(xlabel)
        if show_ylabel:
            ax.set_ylabel(ylabel)
        if show_states:
            ax.scatter(self.env.start_pos[0], self.env.start_pos[1], color='k', marker='o')
        
            if self.env.target_pos.ndim == 1:
                ax.scatter(self.env.target_pos[0], self.env.target_pos[1], color='k', marker='*')
                target_circle = plt.Circle(self.env.target_pos, self.env.target_rad, color='r', alpha=0.3)
                ax.add_patch(target_circle)
            elif self.env.target_pos.ndim > 1:
                for target_pos in self.env.target_pos:
                    ax.scatter(target_pos[0], target_pos[1], color='k', marker='*')
                    target_circle = plt.Circle(target_pos, self.env.target_rad, color='r', alpha=0.3)
                    ax.add_patch(target_circle)



    def plot_att_heatmap(self, set_idx=0, sample_idx=0):
        """
        attention_weights: weight matrix expected shape = 1(or 64),1,210,210 or B,N,T,T where T is 3*context_len
        idx: sample index of batch
        scale_each_row: scales each row INDEPENDENTLY to lie between 0 and 1 for visualization
        """
        info = self.paper_plot_info["att_heatmap"]
        op_traj_dict = self.op_traj_dict_list[set_idx]
        # attention weigghts in the last block
        attention_weights = op_traj_dict['attention_weights']
        # normalized_weights = F.softmax(attention_weights, dim=-1)

        weights = attention_weights.cpu().detach().numpy()
        shape = weights.shape
        
        # Plot attenetion scores for the ith trajectory/sample among the batch (for training batch)
        weights = weights[sample_idx,0,:,:]
        
        # scale each row visualization
        weights = scale_attention_rows(weights)

        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_ticks(np.arange(0,180,25))
        ax.yaxis.set_ticks(np.arange(0,180, 25))
        shw = ax.imshow(weights, cmap=cm.Reds)
        cbar = plt.colorbar(shw)
        cbar_fontsize = 12
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        # cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), shrink=0.65)
        cbar.set_label("Attention Weights (scaled)", fontsize=cbar_fontsize)

        # title =  scale_str + info_string + "setIdx-" +  str(set_idx)
        # plt.title(title)
        # plt.figure(figsize=(12,10))
        # ax = sns.heatmap(weights, linewidth=0.05)
        fname = info["fname"]
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches="tight", dpi=600)

        return

    def plot_vel_mmoc(self, r=0, g_strmplot_lw=1, g_strmplot_arrowsize=1):
        """
        Plot velocity mean, modes, coefs
        """
        info = self.paper_plot_info["vel_field"]
        ts = info["ts"]
        t_names = [0,60,120]

        Ui = np.transpose(self.env.Ui,(0,2,3,1))
        Vi = np.transpose(self.env.Vi,(0,2,3,1))
        Xs = np.arange(0,self.env.xlim) + (self.env.dxy/2)
        Ys = np.arange(0,self.env.ylim) + (self.env.dxy/2)
        X,Y = np.meshgrid(Xs, Ys)

        # Plot Means
        fig, axs = plt.subplots(1,3, sharey=True, figsize=(15,5))
        for i,t in enumerate(ts):
            ax = axs[i]
            ax.set_title(f"t={str(t_names[i])}")
            self.setup_ax(ax,show_ylabel=True if i==0 else False,show_states=False)
            vx_grid = self.env.U[t,:,:] 
            vy_grid = self.env.V[t,:,:] 
            vx_grid = np.flipud(vx_grid)
            vy_grid = np.flipud(vy_grid)

            ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=1.5*g_strmplot_arrowsize, arrowstyle='->')
            v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
            im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)
        
        cbar_fontsize = 12
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        fname = info["fname"] + "_means"
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches="tight", dpi=600)

        for t_id,t in enumerate(ts):
            fig, axs = plt.subplots(2,2, sharey=True, figsize=(5,5))
            for i in range(2):
                for j in range(2):
                    ax= axs[i,j]
                    m = 2*i + j
                    ax.set_title(f"Mode {m+1}")
                    self.setup_ax(ax,show_xlabel= False, show_ylabel=False,
                                    show_states=False,
                                    show_xticks=False, show_yticks=False,
                                    )
                    vx_grid = Ui[t,:,:,m]
                    vy_grid = Vi[t,:,:,m]
                    vx_grid = np.flipud(vx_grid)
                    vy_grid = np.flipud(vy_grid)

                    ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
                    v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
                    im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)
            fname = info["fname"] + "_mode_" + str(t_id)
            save_name = join(self.save_dir,fname)
            plt.savefig(save_name, bbox_inches="tight", dpi=600)

        # another figure just to get color bar
        fig, axs = plt.subplots(2,2, sharey=True, figsize=(5,5))
        for i in range(2):
            for j in range(2):
                ax= axs[i,j]
                m = 2*i + j
                ax.set_title(f"Mode {m+1}")
                self.setup_ax(ax,show_xlabel= False, show_ylabel=False,
                                show_states=False,
                                show_xticks=False, show_yticks=False,
                                )
                vx_grid = Ui[t,:,:,m]
                vy_grid = Vi[t,:,:,m]
                vx_grid = np.flipud(vx_grid)
                vy_grid = np.flipud(vy_grid)

                ax.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
                v_mag_grid = (vx_grid**2 + vy_grid**2)**0.5
                im = ax.contourf(X, Y, v_mag_grid, cmap = "Blues", alpha = 0.5, zorder = -1e5)
        cbar_fontsize = 12
        cbarv = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.80)
        cbarv.set_label("Velocity Magnitude", fontsize=cbar_fontsize)
        fname = info["fname"] + "_crop_colorbar_" 
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox_inches="tight", dpi=600)
        # vx_grid = self.env.U[t,:,:] + np.dot(Ui[t,:,:,:],self.env.Yi[t,r,:])
        # vy_grid = self.env.V[t,:,:] + np.dot(Vi[t,:,:,:],self.env.Yi[t,r,:])
        # vx_grid = np.flipud(vx_grid)
        # vy_grid = np.flipud(vy_grid)

    def plot_loss_returns(self, mat=None):

        dummy_mat = np.random.random((100,4))
        print(dummy_mat.shape)
        x =  np.arange(100).reshape(100,1)
        y1 =  np.exp(-x)
        y1 = dummy_mat + y1
        y2 = np.log(x)
        y2 = dummy_mat + y2
        sns.set()

        info = self.paper_plot_info["loss_avg_returns"]

        fig, axs = plt.subplots(1,2, figsize=(12,5))
        fs = 20
        for ax in axs:
            ax.set_xlim([0,100])
            ax.xaxis.set_ticks(np.arange(0,110,10))
            ax.set_xlabel('No.of updates', fontsize=fs)

        # TODO: change limits

        ymean = np.mean(y1,axis=1)
        ystd = np.std(y1,axis=1)
        ax = axs[0]
        ax.set_ylabel('Loss', fontsize=fs)
        ax.set_ylim([0,2])
        ax.yaxis.set_ticks(np.linspace(0,2,11))
        ax.plot(x[:,0], ymean, color='b', label='Loss')
        ax.fill_between(x[:,0], ymean-ystd, ymean+ystd, color='b', alpha=0.2)
        ax.legend()


        ymean = np.mean(y2,axis=1)
        ystd = np.std(y2,axis=1)
        ax = axs[1]
        ax.set_ylabel('Average Returns', fontsize=fs)
        ax.set_ylim([0,8])
        ax.yaxis.set_ticks(np.linspace(0,8,11))  
        ax.plot(x[:,0], ymean, color='g', label='Avg. returns')
        ax.fill_between(x[:,0], ymean-ystd, ymean+ystd, color='g', alpha=0.2)     
        ax.legend()  
        plt.subplots_adjust( left= 0.1, right=0.9, top=0.9, bottom=0.2, wspace=0.3)
        fname = info["fname"]
        save_name = join(self.save_dir,fname)
        plt.savefig(save_name, bbox="tight", dpi=600)

