""""
cont_gridworld_v5 as per updated api: https://www.gymlibrary.ml/content/environment_creation/
# Contnuous states. n Discrete Actions; multiple possible starting points; with velocity field

"""

import gym
import numpy as np
import math
from os.path import join

# For render
# VIEWPORT_W = 600
# VIEWPORT_H = 400
# SCALE = 30

#TODO: chang env to v6 since we will try using continuous actions
# Contnuous states. n Discrete Actions; multiple possible starting points; with velocity field
class ContGridWorld_v5(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self, state_dim=2, action_dim=1, action_range=[0.0,2*math.pi], grid_dim=[10.,10.],start_pos=[[5.0,5.0]], target_pos=[8.0,8.0], target_rad=1, F=1):
        # super(ContGridWorld_v5, self).__init__()
        print(f"**** initializing ContGridWorld_v5 environment *****")
        #default values
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.grid_dim = grid_dim
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.target_rad = target_rad
        self.F = F

        # self.action_space = gym.spaces.Discrete(self.n_actions)
        self.action_space = gym.spaces.Box(low=self.action_range[0], 
                                            high=self.action_range[1], 
                                            shape=(self.action_dim,))

        self.observation_space = gym.spaces.Box(low=0, high=self.grid_dim[0], shape=(self.state_dim,)) #square domain
        
        return

    def setup(self, cfg, params2):
        """
        To be called expicitly to retreive params
        cfg: dict laoded from a yaml file
        start_pos: list of lists(starting positions); e.g start_pos=[[5.0,2.0]] 
        """
  
        self.cfg = cfg
        self.params2 = params2
        self.params = self.cfg["grid_params"]
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
        self.target_pos = np.array(self.params2['end_pos_us'])*self.space_scale
        self.F = self.params["F"]
        self.state = self.reset()

        self.target_state = np.array(self.target_pos, dtype=np.float32).reshape(2,).copy()


        # Load vel field
        self.U = np.load(join(self.vel_fname,"u.npy"))
        self.V = np.load(join(self.vel_fname,"v.npy"))
        self.vel_shape = self.U.shape # (t,i,j)
        self.dxy = float(self.xlim)/self.U.shape[1]
        self.nT = self.U.shape[0]

        # replace nans with 0s and scale velocity as per vmax_by_F factor
        self.U[np.isnan(self.U)] = 0
        self.V[np.isnan(self.V)] = 0
        self.scale_velocity()

        self.Umax, self.Vmax = np.max(self.U), np.max(self.V)
        self.Umean, self.Vmean = np.mean(np.abs(self.U)), np.mean(np.abs(self.V))
        print("="*20)
        print("init: ",self.start_pos)
        print(f"start_pos.shape={self.start_pos.shape}")
        print("xlim: ", self.xlim)
        print(f"Umax={self.Umax}")
        print(f"Vmax={self.Vmax}")
        print(f"Umean={self.Umean}")
        print(f"Vmean={self.Vmean}")
        print("="*20)


    def scale_velocity(self):
        self.speed = np.sqrt(self.U**2 + self.V**2)
        self.max_speed = np.max(self.speed)
        
        self.scale_factor= self.F * self.vmax_by_F / self.max_speed
        self.U *= self.scale_factor
        self.V *= self.scale_factor


    def get_velocity(self, state, vel):
        """
        extract velocity from vel matrices u and v [:,i,j]; i down; j right
        vel: (U,V)
        """
        U, V = vel        
        x, y = state
        j = int(x // self.dxy)
        i = int((self.ylim - y) // self.dxy)

        return U[0,i,j], V[0,i,j] #TODO: assuming static velocity so putting arbitrary t index


    def transition(self, action, add_noise=False):
        # action *= (2*np.pi/self.n_actions)
        vel = (self.U, self.V)
        u,v = self.get_velocity(self.state, vel)
        self.state[0] += (self.F*math.cos(action) + u)*self.del_t
        self.state[1] += (self.F*math.sin(action) + v)*self.del_t
        # add noise
        # self.state += 0.05*np.random.randint(-3,4)


    def is_outbound(self, check_state = [float('inf'), float('inf')]):
        status = False
        # if no argument, check status for self.state
        if check_state[0] == float('inf') and check_state[1] == float('inf'):
            check_state = self.state
        lims = [self.xlim, self.ylim]
        for i in range(self.state_dim):
            if check_state[i] >= lims[i] or check_state[i]<0:
                status = True
                break
        return status


    def has_reached_target(self):
        status = False
        if np.linalg.norm(self.state - self.target_state) <= self.target_rad:
            status = True
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
        return self.state, self.reward, self.done, info


    def reset(self, reset_state=[float('inf'), float('inf')]):
        # self.state = [x,y]
        # print("in_reset:", self.start_pos)
        reset_state = np.array(reset_state, dtype=np.float32)
        if reset_state[0] == float('inf') and reset_state[1] == float('inf'):
            idx = np.random.randint(0, len(self.start_pos))
            reset_state = self.start_pos.copy()
        self.state = reset_state
        self.done = False
        self.reward = 0
        # self.target_state = reset_state
        return self.state


    def render(self):
        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        #     self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
        
        pass


    def seed(self, seed=None):
        pass