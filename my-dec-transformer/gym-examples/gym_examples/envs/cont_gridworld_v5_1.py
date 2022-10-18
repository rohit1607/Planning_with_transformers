""""
cont_gridworld_v5_1 as per updated api: https://www.gymlibrary.ml/content/environment_creation/
# Contnuous states. n Discrete Actions; multiple possible starting points; with velocity field
v5_1: multiple positive reward temrinal states
"""

import gym
import numpy as np
import math
from os.path import join
import sys
# For render
# VIEWPORT_W = 600
# VIEWPORT_H = 400
# SCALE = 30


# Contnuous states. n Discrete Actions; multiple possible starting points; with velocity field
class ContGridWorld_v5_1(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self, state_dim=2, action_dim=1, action_range=[0.0,2*math.pi], grid_dim=[10.,10.],start_pos=[[5.0,5.0]], target_pos=[[8.0,8.0]], target_rad=1, F=1):
        # super(ContGridWorld_v5, self).__init__()
        print(f"**** initializing ContGridWorld_v5_1 environment *****")
        #default values
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.grid_dim = grid_dim
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.target_rad = target_rad
        self.F = F
        self.rzn = 0

        # self.action_space = gym.spaces.Discrete(self.n_actions)
        self.action_space = gym.spaces.Box(low=self.action_range[0], 
                                            high=self.action_range[1], 
                                            shape=(self.action_dim,))

        self.observation_space = gym.spaces.Box(low=0, high=self.grid_dim[0], shape=(self.state_dim,)) #square domain
        
        return

    def setup(self, cfg, params2, rzn=0):
        """
        To be called expicitly to retreive params
        cfg: dict laoded from a yaml file
        start_pos: list of lists(starting positions); e.g start_pos=[[5.0,2.0]] 
        """

        self.rzn = rzn
        self.cfg = cfg
        self.params2 = params2
        self.params = cfg
        self.state_dim = int(self.params["state_dim"])
        self.action_dim = int(self.params["action_dim"])
        self.a_min, self.a_max = self.params["action_range"]
        self.vel_fname = self.params["vel_field"]
        self.del_t = self.params["del_t"]
        self.vmax_by_F = self.params["vmax_by_F"]
        self.space_scale = self.params['space_scale']
        self.xlim, self.ylim = self.params2["grid_dims"]
        self.target_rad = self.params['target_radius']
        # self.action_space = gym.spaces.Discrete(self.n_actions)
        self.action_space = gym.spaces.Box(low=self.a_min, 
                                    high=self.a_max, 
                                    shape=(self.action_dim,))
        self.observation_space = gym.spaces.Box(low=0, high=self.xlim, shape=(self.state_dim,)) #square domain

        # print(self.observation_space, self.action_space)
        self.start_pos = np.array(self.params2['start_pos_us'])*self.space_scale
        # NOTE: self.target_pos in v5_1, is a list target positions or a 2d Array with 
        #       each row having a target position
        self.target_pos = np.array(self.params2['end_pos_us'])*self.space_scale
        self.F = self.params["F"]
        self.state = self.reset()

        # self.target_state = np.array(self.target_pos, dtype=np.float32).reshape(2,).copy()


        # Load vel field
        self.U = np.load(join(self.vel_fname,"all_u_mat.npy"))
        self.V = np.load(join(self.vel_fname,"all_v_mat.npy"))
        self.Ui = np.load(join(self.vel_fname,"all_ui_mat.npy"))
        self.Vi = np.load(join(self.vel_fname,"all_vi_mat.npy"))
        self.Yi = np.load(join(self.vel_fname,"all_Yi.npy"))
        self.vel_shape = self.U.shape # (t,i,j)
        
        self.dxy = float(self.xlim)/self.U.shape[1]
        self.nT = self.U.shape[0]
        self.n_rzns = self.Yi.shape[1]
        self.n_modes = self.Yi.shape[2]


        # replace nans with 0s and scale velocity as per vmax_by_F factor
        self.U[np.isnan(self.U)] = 0
        self.V[np.isnan(self.V)] = 0
        self.Ui[np.isnan(self.Ui)] = 0
        self.Vi[np.isnan(self.Vi)] = 0
        self.Yi[np.isnan(self.Yi)] = 0

        self.scale_velocity()

        self.Umax, self.Vmax = np.max(self.U), np.max(self.V)
        self.Umean, self.Vmean = np.mean(np.abs(self.U)), np.mean(np.abs(self.V))
        print("="*20)
        print("start_pos: ",self.start_pos)
        print(f"start_pos.shape={self.start_pos.shape}")
        print(f"target_pos= {self.target_pos}")
        print(f"n_rzns = {self.n_rzns}")
        print(f"Ui.shape = {self.Ui.shape}")        
        print("xlim: ", self.xlim)
        print(f"Umax={self.Umax}")
        print(f"Vmax={self.Vmax}")
        print(f"Umean={self.Umean}")
        print(f"Vmean={self.Vmean}")
        print("="*20)

    def set_rzn(self, rzn):
        assert(int(rzn) < self.n_rzns), print("Error: Invalid rzn_id")
        self.rzn = int(rzn)


    def scale_velocity(self):
        self.speed = np.sqrt(self.U**2 + self.V**2)
        self.max_speed = np.max(self.speed)
        if self.max_speed > 0:
            self.scale_factor= self.F * self.vmax_by_F / self.max_speed
        else:
            self.scale_factor = 1
        self.U *= self.scale_factor
        self.V *= self.scale_factor
        self.Ui *= self.scale_factor
        self.Vi *= self.scale_factor


    def get_velocity(self, state):
        """
        extract velocity from vel matrices u and v [:,i,j]; i down; j right
        vel: (U,V)
        """
       
        t, x, y = state
        # print(x, self.dxy)
        j = int(x // self.dxy)
        i = int((self.ylim - y) // self.dxy)
        t = int(t)
        # print(f"*** verify; x,y = {x},{y}")
        # print(f"*** verify; i,j = {i}, {j} \n")
        # vx = vel_field_data[0][t, i, j] + np.matmul(vel_field_data[2][t, :, i, j],vel_field_data[4][t, rzn,:])
        # vy = vel_field_data[1][t, i, j] + np.matmul(vel_field_data[3][t, :, i, j], vel_field_data[4][t, rzn,:])

        u = self.U[t,i,j] + np.matmul(self.Ui[t,:,i,j],self.Yi[t,self.rzn,:])
        v = self.V[t,i,j] + np.matmul(self.Vi[t,:,i,j],self.Yi[t,self.rzn,:])
        return u, v


    def transition(self, action, add_noise=False):
        # action *= (2*np.pi/self.n_actions)
     
        u,v = self.get_velocity(self.state)
        self.state[0] += 1
        self.state[1] += (self.F*math.cos(action) + u)*self.del_t
        self.state[2] += (self.F*math.sin(action) + v)*self.del_t
        # add noise
        # self.state += 0.05*np.random.randint(-3,4)


    def is_outbound(self, check_state = [float('inf'),float('inf'),float('inf')]):
        status = False
        # if no argument, check status for self.state
        if check_state[0] == float('inf') and check_state[1] == float('inf') and check_state[2] == float('inf'):
            check_state = self.state
        lims = [self.nT, self.xlim, self.ylim]
        for i in range(self.state_dim):
            if check_state[i] >= lims[i] or check_state[i]<0:
                status = True
                break
        # extra condition to check if y is 0 -> i is 100 --> out of bounds of vel matrix
        if check_state[2] == 0:
            status = True
        return status


    def has_reached_target(self):
        status = False

        for target_pos in self.target_pos:
            if np.linalg.norm(self.state[1:] - target_pos) <= self.target_rad:
                status = True
                break

        return status


    def step(self, action):
        old_s = self.state  #to restore position in case of outbound
        self.transition(action)
        self.reward = -1
        self.trunc = False
        has_reached_target = self.has_reached_target()
        is_outbound = self.is_outbound()
        if has_reached_target:
            self.reward = self.xlim
            self.done = True
        elif is_outbound:
            self.reward = -self.xlim
            self.done = True
            self.state =  old_s
        info = {"is_outbound": is_outbound, "has_reached_target":has_reached_target} # what do i put here
        return np.array(self.state), self.reward, self.done, info


    def reset(self, reset_state=[float('inf'),float('inf'),float('inf')]):
        # self.state = [x,y]
        # print("in_reset:", self.start_pos)
        reset_state = np.array(reset_state, dtype=np.float32)
        if reset_state[0] == float('inf') and reset_state[1] == float('inf') and reset_state[2] == float('inf'):
            # idx = np.random.randint(0, len(self.start_pos))
            x0, y0 = self.start_pos.copy()
            reset_state = [0, x0, y0]
        self.state = np.array(reset_state, dtype=np.float32)
        self.done = False
        self.reward = 0
        # self.target_state = reset_state
        return np.array(self.state,dtype=np.float32)


    def render(self):
        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        #     self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
        
        pass


    def seed(self, seed=None):
        pass