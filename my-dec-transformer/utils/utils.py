
import pickle
import torch
import numpy as np
import random
import time
import math
import yaml
import matplotlib.pyplot as plt
def save_yaml(savepath, data):
    with open(savepath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_angle_in_0_2pi(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)


def read_cfg_file(cfg_name, print_dict=False):
    with open(cfg_name, "r") as file:
        try:
            cfg = yaml.full_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            print("*"*20)
    # if print_dict:
    #     print("="*20)
    #     for item in cfg.keys():
    #         print(f"{item}:")
    #         for subitem in cfg[item].keys():
    #             print(f"\t {subitem}: \t {cfg[item][subitem]}")
    #     print("="*20)

    return cfg

class log_and_viz_params:
    def __init__(self, model):
        self.model = model
        self.wb_log_dict = {}
        self.grad_log_dict = {}
        # No. of weights and bias arrays
        self.N = sum(1 for dummy in model.named_parameters()) 
        self.log_counts = 0
        for n_param, param in zip(model.named_parameters(), model.parameters()):
            self.wb_log_dict[n_param[0]] = []
            self.grad_log_dict[n_param[0]] = []
        

    def log_params(self, device=None):
        self.log_counts += 1

        if device=='cuda':
            self.model = self.model.cpu()
            for n_param, param in zip(self.model.named_parameters(), self.model.parameters()):
                self.wb_log_dict[n_param[0]].append(param.cpu().detach().numpy().reshape(-1,1))
                self.grad_log_dict[n_param[0]].append(param.grad.cpu().numpy().reshape(-1,1))

        else:
            for n_param, param in zip(self.model.named_parameters(), self.model.parameters()):
                self.wb_log_dict[n_param[0]].append(param.detach().numpy().reshape(-1,1))
                self.grad_log_dict[n_param[0]].append(param.grad.numpy().reshape(-1,1))



    def print_logs(self, idx=-1):
        return


    def visualize_wb(self, size=4, savefig=None):

        height_ratios = [size for i in range(self.N)] 
        fig, axs = plt.subplots(self.N, 2, gridspec_kw={'height_ratios': height_ratios })
        fig.set_figheight(self.N*size)

        named_pars = self.model.named_parameters()
        pars = self.model.parameters()
        for i, (n_param, param) in enumerate(zip(named_pars, pars)):
            key = n_param[0]
            axs[i,0].set_title(key)
            (odim, idim) = self.wb_log_dict[key][0].shape #should be independent of i
            print("verify odim , idim = ", odim, idim)

            layer_wts= np.array(self.wb_log_dict[key])
            layer_grads = np.array(self.grad_log_dict[key])
            n_wts = odim*idim
            for j in range(n_wts):
                axs[i,0].plot(layer_wts[:,j,0])
                axs[i,1].plot(layer_grads[:,j,0])

        fig.tight_layout()
        if savefig != None:
            plt.savefig(savefig, dpi= 1200)
        
        return