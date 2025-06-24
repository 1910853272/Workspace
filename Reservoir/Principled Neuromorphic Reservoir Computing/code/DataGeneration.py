# -*- coding: utf-8 -*-
"""

"""

import torch
import numpy as np
from scipy.integrate import solve_ivp
from jitcdde import jitcdde, y, t, jitcdde_lyap
from types import SimpleNamespace
from scipy.io import loadmat
from scipy.interpolate import interp1d

##
## Lorenz '63
##
# Initial version of this function comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code
class Lorenz63():

    def __init__(
        self,
        warmup = 5.,
        traintime = 10.,
        testtime = 120.,
        dt = 0.025,
        method = 'RK23',
        seed = None,
    ):
        super(Lorenz63, self).__init__()

        self.sigma = 10
        self.betaLor = 8 / 3
        self.rho = 28   
        # Lyapunov time of the system
        self.lyaptime=1.104
        
        # time step
        self.dt=dt     
        # units of time to warm up for. Need to have warmup_pts >= 1
        self.warmup=warmup
        # units of time to train for
        self.traintime=traintime
        # units of time to test for
        self.testtime=testtime
        # total time to run for
        self.maxtime = self.warmup+self.traintime+self.testtime
        
        # discrete-time versions of the times defined above
        self.warmup_pts=round(self.warmup/self.dt)
        self.traintime_pts=round(self.traintime/self.dt)
        self.warmtrain_pts=self.warmup_pts+self.traintime_pts
        self.testtime_pts=round(self.testtime/self.dt)
        self.maxtime_pts=round(self.maxtime/self.dt)
        self.lyaptime_pts=round(3*self.lyaptime/self.dt)        
             
        # t values for whole evaluation time
        # (need maxtime_pts + 1 to ensure a step of dt)
        self.t_eval=np.linspace(0,self.maxtime,self.maxtime_pts+1)

        if seed==None:
            point_init = [17.67715816276679, 12.931379185960404, 43.91404334248268]
        else:
            np.random.seed(seed=seed)
            point_init = np.zeros((3))
            point_init[0] = 18.0*(2*(np.random.rand(1)-0.5))
            point_init[1] = 24.0*(2*(np.random.rand(1)-0.5))
            point_init[2] = 6+39.0*(np.random.rand(1))
            point_init = point_init.tolist()
               
        self.data = solve_ivp(self.solve_system, (0, self.maxtime),  point_init, t_eval=self.t_eval, method=method)
        
        # total variance of the data
        self.total_var=np.var(self.data.y[0,:])+np.var(self.data.y[1,:])+np.var(self.data.y[2,:])
              
    # Generate data from the system
    def solve_system(self, t, y):
        dy0 = self.sigma * (y[1] - y[0])
        dy1 = y[0] * (self.rho - y[2]) - y[1]
        dy2 = y[0] * y[1] - self.betaLor * y[2]
        
        # since Lorenz63 is 3-dimensional, dy/dt should be an array of 3 values
        return [dy0, dy1, dy2]

##
## Lorenz '63 with noise
##
# Initial version of this function comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code
class Lorenz63Noisy():

    def __init__(
        self,
        warmup = 5.,
        traintime = 10.,
        testtime=120.,
        seed = None,
    ):
        super(Lorenz63Noisy, self).__init__()

        self.sigma = 10
        self.betaLor = 8 / 3
        self.rho = 28   
        # Lyapunov time of the system
        self.lyaptime=1.104
        
        # time step
        self.dt=0.025      
        # units of time to warm up for. Need to have warmup_pts >= 1
        self.warmup=warmup
        # units of time to train for
        self.traintime=traintime
        # units of time to test for
        self.testtime=testtime
        # total time to run for
        self.maxtime = self.warmup+self.traintime+self.testtime
        
        # discrete-time versions of the times defined above
        self.warmup_pts=round(self.warmup/self.dt)
        self.traintime_pts=round(self.traintime/self.dt)
        self.warmtrain_pts=self.warmup_pts+self.traintime_pts
        self.testtime_pts=round(self.testtime/self.dt)
        self.maxtime_pts=round(self.maxtime/self.dt)
        self.lyaptime_pts=round(3*self.lyaptime/self.dt)        
             
        # t values for whole evaluation time
        # (need maxtime_pts + 1 to ensure a step of dt)
        self.t_eval=np.linspace(0,self.maxtime,self.maxtime_pts+1)

        # generate Gaussian random numbers at step dt and then interpolate for integrator
        self.rng = np.random.default_rng(seed=1)
        
        # width of the Gaussian distribution
        self.width=1.
        
        self.xran = self.rng.normal(0.,self.width,self.maxtime_pts+1)
        self.yran = self.rng.normal(0.,self.width,self.maxtime_pts+1)
        self.zran = self.rng.normal(0.,self.width,self.maxtime_pts+1)
        self.xran_i = interp1d(self.t_eval,self.xran)
        self.yran_i = interp1d(self.t_eval,self.yran)
        self.zran_i = interp1d(self.t_eval,self.zran)


        if seed==None:
            point_init = [17.67715816276679, 12.931379185960404, 43.91404334248268]
        else:
            np.random.seed(seed=seed)
            point_init = np.zeros((3))
            point_init[0] = 18.0*(2*(np.random.rand(1)-0.5))
            point_init[1] = 24.0*(2*(np.random.rand(1)-0.5))
            point_init[2] = 6+39.0*(np.random.rand(1))
            point_init = point_init.tolist()
               
        self.data = solve_ivp(self.solve_system, (0, self.maxtime),  point_init, t_eval=self.t_eval, method='RK23')
        
        # total variance of the data
        self.total_var=np.var(self.data.y[0,:])+np.var(self.data.y[1,:])+np.var(self.data.y[2,:])
              
    # Generate data from the system
    def solve_system(self, t, y):
        dy0 = self.sigma * (y[1] - y[0]) + self.xran_i(t)
        dy1 = y[0] * (self.rho - y[2]) - y[1] + self.yran_i(t)
        dy2 = y[0] * y[1] - self.betaLor * y[2]+ self.zran_i(t)
        
        # since Lorenz63 is 3-dimensional, dy/dt should be an array of 3 values
        return [dy0, dy1, dy2]  

##
## Double-Scroll
##
# Initial version of this function comes from the following GitHub project (@author: Dan):
# https://github.com/quantinfo/ng-rc-paper-code
class DoubleScroll():

    def __init__(
        self,
        warmup = 1.,
        traintime = 100.,
        testtime=800.,
    ):
        super(DoubleScroll, self).__init__()

        self.r1 = 1.2
        self.r2 = 3.44
        self.r4 = 0.193
        self.alpha = 11.6
        self.ir = 2*2.25e-5 
        # Lyapunov time of the system
        self.lyaptime=7.8125
        
        # time step
        self.dt=0.25      
        # units of time to warm up for. Need to have warmup_pts >= 1
        self.warmup=warmup
        # units of time to train for
        self.traintime=traintime
        # units of time to test for
        self.testtime=testtime
        # total time to run for
        self.maxtime = self.warmup+self.traintime+self.testtime
        
        # discrete-time versions of the times defined above
        self.warmup_pts=round(self.warmup/self.dt)
        self.traintime_pts=round(self.traintime/self.dt)
        self.warmtrain_pts=self.warmup_pts+self.traintime_pts
        self.testtime_pts=round(self.testtime/self.dt)
        self.maxtime_pts=round(self.maxtime/self.dt)
        self.lyaptime_pts=round(3*self.lyaptime/self.dt)        
         
        # t values for whole evaluation time
        # (need maxtime_pts + 1 to ensure a step of dt)
        self.t_eval=np.linspace(0,self.maxtime,self.maxtime_pts+1)
          
        self.data = solve_ivp(self.solve_system, (0, self.maxtime), [0.37926545,0.058339,-0.08167691] , t_eval=self.t_eval, method='RK23')
        
        # total variance of the data
        self.total_var=np.var(self.data.y[0,:])+np.var(self.data.y[1,:])+np.var(self.data.y[2,:])
                
    # Generate data from the system
    def solve_system(self, t, y):              
        dV = y[0]-y[1] # V1-V2
        g = (dV/self.r2)+self.ir*np.sinh(self.alpha*dV)
        dy0 = (y[0]/self.r1)-g 
        dy1 = g-y[2]
        dy2 = y[1]-self.r4*y[2]
      
        # since Double-Scroll is 3-dimensional, dy/dt should be an array of 3 values
        # y[0] = V1, y[1] = V2, y[2] = I
        return [dy0, dy1, dy2] 

##
## Mackey-Glass
##


# time step
dt = 3.0 #0.2
# units of time to warm up for
warmup=90.
# units of time to train for
traintime=1800.
# Lyapunov time of double-scroll system
lyaptime=185 # From numerical estimates 

class MackeyGlass():

    def __init__(
        self,
        warmup = 90.,
        traintime = 1800.,
        testtime=6000.,
        seed = 311,
    ):
        super(MackeyGlass, self).__init__()

        self.tau = 17
        self.nmg = 10
        self.beta = 0.2
        self.gamma = 0.1
        self.lyaptime=185.
        
        # time step
        self.dt=3.0      
        # units of time to warm up for. Need to have warmup_pts >= 1
        self.warmup=warmup
        # units of time to train for
        self.traintime=traintime
        # units of time to test for
        self.testtime=testtime
        # total time to run for
        self.maxtime = self.warmup+self.traintime+self.testtime
        
        # discrete-time versions of the times defined above
        self.warmup_pts=round(self.warmup/self.dt)
        self.traintime_pts=round(self.traintime/self.dt)
        self.warmtrain_pts=self.warmup_pts+self.traintime_pts
        self.testtime_pts=round(self.testtime/self.dt)
        self.maxtime_pts=round(self.maxtime/self.dt)
        self.lyaptime_pts=round(3*self.lyaptime/self.dt)        
         
        # t values for whole evaluation time
        # (need maxtime_pts + 1 to ensure a step of dt)
        self.t_eval=np.linspace(0,self.maxtime,self.maxtime_pts+1)

        self.solve_system = [ self.beta * y(0,t-self.tau) / (1 + y(0,t-self.tau)**self.nmg) - self.gamma*y(0) ]

        np.random.seed(seed=seed)    
        DDE = jitcdde(self.solve_system)
        DDE.constant_past([0.5+0.4*np.random.rand(1)[0]])
        DDE.step_on_discontinuities()
        self.data = SimpleNamespace(**{'y': 0})
        self.data.y = np.zeros((1,self.maxtime_pts))
        count = 0
        for time in np.arange(DDE.t, DDE.t+self.maxtime, self.dt):
            self.data.y[0,count] = DDE.integrate(time)[0]
            count += 1

        # total variance of the data
        self.total_var=np.var(self.data.y[0,:])

##
## Kuramoto-Sivashinsky
##

class KuramotoSivashinskyLong():

    def __init__(
        self,
        device
    ):
        super(KuramotoSivashinskyLong, self).__init__()

        self.lyaptime=20.        
        # time step
        self.dt=0.5      
        # units of time to warm up for. Need to have warmup_pts >= 1
        self.warmup=200
        # units of time to train for
        self.traintime=2000. 
        # units of time to test for
        self.testtime=2000 
        # total time to run for
        self.maxtime = self.warmup+self.traintime+self.testtime
        
        # discrete-time versions of the times defined above
        self.warmup_pts=round(self.warmup/self.dt)
        self.traintime_pts=round(self.traintime/self.dt)
        self.warmtrain_pts=self.warmup_pts+self.traintime_pts
        self.testtime_pts=round(self.testtime/self.dt)
        self.maxtime_pts=round(self.maxtime/self.dt)
        self.lyaptime_pts=round(3*self.lyaptime/self.dt)        
         
        # t values for whole evaluation time
        # (need maxtime_pts + 1 to ensure a step of dt)
        self.t_eval=torch.linspace(0,self.maxtime,self.maxtime_pts+1)
        
        # Load pre-computed runs of the Kuramoto-Sivashinsky system
        self.DATA = loadmat('/data/Kuramoto_Sivashinsky_precomputed.mat')
        self.DATA = torch.from_numpy(self.DATA["data"]).to(device)

    # Choose one of the precomputed time series
    def assign_data(self, seed = 0,):
        self.data = SimpleNamespace(**{'y': 0})
        self.data.y = self.DATA[:,:,seed]

        # total variance of the data
        self.total_var=torch.sum(torch.var(self.data.y, dim=1))

