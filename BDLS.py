import numpy as np
from ula import GaussianMixtureULA

#This file implements a classic Birth-Death Langevin Sampling algorithm. 
# This is used to have the particles go through the BDLS process when needed, 
# see detail in BDEC.py

def kernel(x,y,sigma=0.05):
    coeff = 1.0 / ((2 * np.pi) * sigma**2)
    return coeff * np.exp(- ((x-y) ** 2) / (2 * sigma ** 2))

class BDLS_algo :
    def __init__(self, weights, means, covs, temp, delta_t, T,init_particles):
        self.weights = weights
        self.means = means
        self.covs = covs
        self.temp = temp
        self.delta_t = delta_t
        self.T = T
        self.init_particles = init_particles
        self.sampler=GaussianMixtureULA(self.weights, self.means, self.covs, 
                                        self.temp, self.delta_t)
        self.current_particles=self.init_particles

    #Here we use the langevin update previously presented in ULA.py
        
    def updated_particles(self):
        self.current_particles=self.sampler.langevin_update(
            self.current_particles)

    #Implementing first step of computing the particles' instantaneous 
    # birth-death rate
    def compute_r(self):
        N=len(self.current_particles)
        r=np.zeros(N)
        for i in range(N):
            kernel_values=[kernel(self.current_particles[i],
                                  self.current_particles[j])for j in range(N)]
            r[i]=(
                np.log(np.mean(kernel_values)) 
            + float(self.sampler.V(self.current_particles[i]))
            )
        return r
    
    #Implementing the second part of the computation of the rate as well as 
    # using this instantaneous birth-death rate to kill (or duplicate) each 
    # particle. Of course, we want to keep the same number of particles at all 
    # times so when one particle is duplicated (resp. killed), another one 
    # uniformly chosen from the rest is killed (resp.duplicated).

    def BD(self):
        r=self.compute_r()
        N=len(self.current_particles)
        old_particles=list(self.current_particles)
        r_barre=np.mean(r)
        beta_barre=r-r_barre
        new_particles=old_particles.copy()
        for i,x_i in enumerate(old_particles):
            u=np.random.rand()
            if beta_barre[i]>0:
                proba=1-np.exp((-beta_barre[i])*self.delta_t)
                if u<proba:
                    j=np.random.choice([k for k in range(N) if k!=i])
                    new_particles[i]=old_particles[j] 
            else:
                proba=1-np.exp((beta_barre[i])*self.delta_t)
                if u<proba:
                    j=np.random.choice([k for k in range(N) if k!=i])
                    new_particles[j]=x_i 
        self.current_particles=np.array(new_particles)
        return self.current_particles

    def run(self):
        for t in range(self.T):
            self.updated_particles()
            self.BD()
        return self.current_particles
                



    
    
    
