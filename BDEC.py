import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from ula import GaussianMixtureULA
from BDLS import BDLS_algo
from Exploration_Component import exploration_component
from Metropolized_Transition_Operator import MetropolizedTransitionOperator

class BDEC:
    def __init__(self, init_M, init_S, init_W, thr, B, N, weights, means, covs, 
                 delta_t, T,init_particles_hot,init_particles_target,J,
                 temp_hot=0.05,temp_target=1):
        self.M = init_M
        self.S = init_S  
        self.W = init_W
        self.thr = thr
        self.B = B
        self.N = N
        self.J = J
        self.weights = weights
        self.means = means 
        self.covs = covs
        self.temp_hot = 0.05
        self.temp_target = 1
        self.delta_t = delta_t
        self.T = T
        self.init_particles_hot = init_particles_hot
        self.init_particles_target = init_particles_target
        self.current_particles_hot = self.init_particles_hot
        self.current_particles_target = self.init_particles_target
        self.is_new = False

    def update_hot(self,m):
        self.sampler_hot=GaussianMixtureULA(self.weights, self.means, self.covs,
                                             self.temp_hot, self.delta_t)
        for t in range(self.T+m):  
            self.current_particles_hot=self.sampler_hot.langevin_update(
                self.current_particles_hot)

    def exploration_component(self,j):
        for i in range(j):
            ec=exploration_component(self.M,self.S,self.W,self.thr,self.B,
                                     self.N,self.weights,self.means,self.covs,
                                     self.temp_hot,self.delta_t,self.T,
                                     self.current_particles_hot)
            self.is_new,self.M,self.S,self.W=ec.mode_finder()
        
    
    def update_target(self,m):
        if self.is_new==False:
            for i in range(self.T+m):
                print(i)
                self.sampler_target=GaussianMixtureULA(self.weights, self.means, 
                                              self.covs, self.temp_target, 
                                              self.delta_t)
                self.current_particles_target=(
                    self.sampler_target.langevin_update(
                        self.current_particles_target))
                self.sampler_target=BDLS_algo(self.weights, self.means,
                                              self.covs, self.temp_target, 
                                              self.delta_t, self.T,
                                              self.current_particles_target)
                self.current_particles_target=self.sampler_target.BD()
                
        else:
            for i in range(self.T+m):
                print(i)
                mto=MetropolizedTransitionOperator(self.weights, self.means, 
                                                   self.covs, self.temp_target,
                                                   self.delta_t, self.T,self.M,
                                                   self.S,self.W,
                                                   self.current_particles_target
                                                   )
                self.current_particles_target=mto.update_particles()
                self.sampler_target=BDLS_algo(self.weights, self.means, 
                                              self.covs, self.temp_target, 
                                              self.delta_t,self.T,
                                              self.current_particles_target)                
                self.current_particles_target=self.sampler_target.BD()
    
    def run(self):
        for j in range(1,self.J+1):
            for m in range((self.J-1)*self.T):
                self.update_hot(m)
                self.exploration_component(j)
                self.update_target(m)
                print(m)
        return self.current_particles_target

