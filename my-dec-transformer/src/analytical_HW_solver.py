from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import time

class analytical_opt_HW:

    def __init__(self, X0, Xf, u, v, w):
        self.a = 0 #random_init
        self.X0 = self.x0, self.y0 = X0
        self.Xf = self.xf, self.yf = Xf
        self.x0, self.y0 = X0
        self.xf, self.yf = Xf
        self.u = u
        self.v = v
        self.w = w 
        self.opt_path = None
    
    def reset(self, X0, Xf, u, v, w):
        self.a = 0 #random_init
        self.X0 = self.x0, self.y0 = X0
        self.Xf = self.xf, self.yf = Xf
        self.x0, self.y0 = X0
        self.xf, self.yf = Xf
        self.u = u
        self.v = v
        self.w = w 
        self.opt_path = None


    def X1(self, a=None):
        if a == None:
            a = self.a
        x1 = self.x0 + (self.u*np.cos(a)+self.v)*self.w/(self.u*np.sin(a))
        y1 = self.y0 + self.w 
        return (x1, y1)

    def T(self,a):
        x0,y0 = self.X0
        xf,yf = self.Xf
        u = self.u
        w = self.w
        v = self.v

        t1 = w/(u*np.sin(a))
        t2 = (1/u)*np.sqrt((xf - x0 - (u*np.cos(a)+v)*t1)**2 + (yf - w - y0)**2)
        return t1 + t2

    def optimize(self, a0):
        res = minimize(self.T, a0, method='Nelder-Mead', tol=1e-6)
        self.a = res.x[0]
        return res

    def get_opt_path(self, n_waypoints):
        x1,y1 = self.X1()
        d1 = np.linalg.norm(np.array((x1,y1))-np.array(self.X0))
        d2 = np.linalg.norm(np.array(self.Xf) - np.array((x1,y1)))
        d1_frac = d1/(d1+d2)
        n_pts_1 = int(n_waypoints*d1_frac)
        n_pts_2 = int(n_waypoints*(1-d1_frac))
        path1 = np.linspace(self.X0, (x1,y1), n_pts_1, endpoint=False) #(n1, 2)
        path2 = np.linspace((x1,y1), self.Xf, n_pts_2) #  #(n1, 2)
        # print(path1.shape, path2.shape)
        self.opt_path = np.concatenate((path1, path2), axis=0)
        # print(self.opt_path.shape)
        return self.opt_path

    def plot_opt_path(self, savepath=None):
        plt.plot(self.opt_path[:,0], self.opt_path[:,1])
        plt.scatter([self.x0], [self.y0])
        plt.scatter(self.xf, self.yf)
        plt.xlim([0,100])
        plt.ylim([0,100])
        # plt.show()
        if savepath != None:
            plt.savefig(savepath, dpi = 200)


def test():
    X0 = (50.0,50.0)
    w = 10.0
    Xf = (50.0, 70.0)
    u = 1
    v = -1
    hw_opt = analytical_opt_HW(X0, Xf, u, v, w)
    res =  hw_opt.optimize(a0=.1)
    hw_opt.get_opt_path(100)
    savepath="/home/rohit/Documents/Research/Planning_with_transformers/Decision_transformer/my-dec-transformer/tmp"
    fname = "analytical_opt_path.png"
    # hw_opt.plot_opt_path(savepath=join(savepath, fname))

if __name__ == "__main__":
    start = time.time()
    for i in range(200):
        test()
        print(i)
    end = time.time()

    print(f"time taken for 200 runs = {end-start} ")

