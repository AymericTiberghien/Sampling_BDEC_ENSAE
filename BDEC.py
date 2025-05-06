import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from ula import GaussianMixtureULA
from BDLS import BDLS_algo
from Exploration_Component import exploration_component
from Metropolized_Transition_Operator import MetropolizedTransitionOperator

#This file brings together all the other files to implement the main algorithm 
# of the paper step by step. This algorithm is separated into three steps, we 
# will go over those using the corresponding implemented functions

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
        # self.init_particles_hot = init_particles_hot
        # self.init_particles_target = init_particles_target
        self.current_particles_hot = init_particles_hot
        self.current_particles_target = init_particles_target
        self.is_new = False

    #This function updates the population of hot particles.

    def update_hot(self,m):
        self.sampler_hot=GaussianMixtureULA(self.weights, self.means, self.covs,
                                             self.temp_hot, self.delta_t)
        for t in range(self.T+m):  
            self.current_particles_hot=self.sampler_hot.langevin_update(
                self.current_particles_hot)
            
    #This function applies the exploration component to update the mode 
    # information

    def exploration_component(self,j):
        for i in range(j):
            ec=exploration_component(self.M,self.S,self.W,self.thr,self.B,
                                     self.N,self.weights,self.means,self.covs,
                                     self.temp_hot,self.delta_t,self.T,
                                     self.current_particles_hot)
            self.is_new,self.M,self.S,self.W=ec.mode_finder()


    #This function updates the population of target particles using the mode 
    # information whether a new mode was found or not. 
    
    def update_target(self,m):
        if self.is_new==False:
            #If no new mode has been found during the exploration process, the 
            # target particles go through a simple BDLS process as described in 
            # BDLS.py
            for i in range(m,self.T+m+1):
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
            #If a new mode has been found during the exploration process, the 
            # target particles population go through the MH update process that 
            # was described in Metropolized_Transition_Operator.py . They then 
            # go through a birth-death process
            for i in range(m,self.T+m+1):
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
                
    #This function runs the whole algorithm J number of times.
    def run(self):
        particules=[]
        for j in range(1,self.J+1):
            print("Debut de la boucle "+str(j))
            self.update_hot((self.J-1)*self.T)
            self.exploration_component(j)
            self.update_target((self.J-1)*self.T)
            particules.append(self.current_particles_target)
            self.particules=particules
        

