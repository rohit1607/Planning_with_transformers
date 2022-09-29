"""
To visualize weights and gradients with self written utility
Was oringinally inside src folder
"""

from pickletools import optimize
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
from root_path import ROOT
sys.path.insert(0, ROOT)
from utils.utils import log_and_viz_params
# wandb.login()
# wandb.init(project="param_viz_test",
#             name = "exp_1",
#             config = {"lr": 0.001,
#                         "n_epochs": 200,
#                         "h_dim": 10}
#                 )
# config = wandb.config

config = {"lr": 0.001,
            "n_epochs": 200,
            "h_dim": 3}


X = torch.Tensor(np.arange(20))
# print(f"X={X}")
Y = X**2
# print(type(Y))

class simpleMLP(nn.Module):
    def __init__(self,i_dim, h_dim, o_dim):
        super().__init__()

        self.lin1 = nn.Linear(i_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, o_dim)
        self.model = nn.Sequential(self.lin1,
                                nn.ReLU(),
                                self.lin2,
                                nn.ReLU(),
                                self.lin3,
                                nn.ReLU()
                            )
                            
  

    def forward(self, x):
        
        return self.model(x)


# lr = config.lr
# n_epochs = config.n_epochs
# h_dim = config.h_dim
lr = 0.001
h_dim = 3
n_epochs = 100
model = simpleMLP(1,h_dim,1)
print(f"model.params: {model.parameters()}")
optimizer = torch.optim.Adam(model.parameters(),
                            lr = lr
                            )
params = model.parameters()
# print(type(params))

def viz_weights_and_grads(model):
    for n_param, param in zip(model.named_parameters(), model.parameters()):
        print(f"----- n_param[0] ; {n_param[0]}")      # param name
        
        weights = param.detach().numpy()
        grads = param.grad
        print(f"shape: {weights.shape}")   # param value
        # print(f"tensors:\n {param}")
        print(f"weight values:\n {weights}")
        print(f"grad vals:\n {grads}")
        print()



# def crit(y, pred_y):
#     return F.mse_loss(pred_y, y, reduction='mean')
p_log = log_and_viz_params(model)

loss = None
# wandb.watch(model,criterion=crit, log="all", log_freq=1, log_graph=True)
for i in range(n_epochs):
    # wandb.log({"loss": loss})
    j = 0
    for x,y in zip(X,Y):
        j+=1
        # print(x, y)
        pred_y = model.forward(x.unsqueeze(dim=0))
        y = y.unsqueeze(dim=0)
        # print(pred_y)
        loss = F.mse_loss(pred_y, y, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j%2 == 0:
            p_log.log_params()

viz_weights_and_grads(model)
p_log.visualize_wb(savefig= "../tmp/params2.png")
# wandb.finish()

# test_X = X 
# pred_test_Y = model.forward(test_X[:].reshape(len(X),1))
# plt.plot(pred_test_Y.detach().numpy())
# plt.savefig("y=xsq")


