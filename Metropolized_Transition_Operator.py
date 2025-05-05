import numpy as np
from scipy.stats import multivariate_normal
from ula import GaussianMixtureULA

class MetropolizedTransitionOperator:
    def __init__(self, weights, means, covs, temp, delta_t, T,M,S,W,
                 particles_target):
        self.weights = weights
        self.means = means
        self.covs = covs
        self.temp = temp
        self.delta_t = delta_t
        self.T = T
        self.M = M
        self.S = S
        self.W = W
        self.particles_target = particles_target
        self.sampler=GaussianMixtureULA(weights, means, covs, temp, delta_t)

    def sample_rho_hat(self, n_samples=2000):
        m = len(self.M)
        comp_indices = np.random.choice(
            np.arange(m),
            size=n_samples,
            p=self.W
        )
        
        new_particles = np.zeros((n_samples, self.M[0].shape[0]))
        for j, i in enumerate(comp_indices):
            new_particles[j] = multivariate_normal.rvs(
                mean=self.M[i],
                cov=self.S[2*i:2*i+2,:]
            )
        self.new_particles = new_particles
    
    
    def rho_hat_values(self):
        rho_hat_values_new_particles = np.zeros(len(self.new_particles))
        rho_hat_values_target = np.zeros(len(self.particles_target))
        for i in range(len(self.new_particles)):
            for m in range(len(self.M)):
                rho_hat_values_new_particles[
                    i
                    ] += self.W[m]*multivariate_normal.pdf(self.new_particles[i
                                                                              ],
                                                            mean=self.M[m], 
                                                            cov=self.S[2*m:2*m+2,:])
                rho_hat_values_target[
                    i
                    ] += self.W[m]*multivariate_normal.pdf(
                        self.particles_target[i],mean=self.M[m],
                        cov=self.S[2*m:2*m+2,:])
        return rho_hat_values_target, rho_hat_values_new_particles
    
    def acceptance_ratio(self):
        rho_hat_values_target, rho_hat_values_new_particles=self.rho_hat_values(

        )
        density_vals_target=self.sampler.density(self.particles_target)[1]
        density_vals_new_particles=self.sampler.density(self.new_particles)[1]
        acceptance_ratio=np.zeros(len(self.particles_target))
        for i in range(len(self.particles_target)):
            acceptance_ratio[i]=np.min((1.0,((rho_hat_values_new_particles[i]*
                                             density_vals_target[i])/
                                             (rho_hat_values_target[i]*
                                              density_vals_new_particles[i]))))
        return acceptance_ratio 

    def update_particles(self):
        self.sample_rho_hat(n_samples=2000)
        acceptance_ratio=self.acceptance_ratio()
        u= np.random.uniform(0,1,len(self.particles_target))
        accept=u<acceptance_ratio
        for i in range(len(self.particles_target)):
            if accept[i]:
                self.particles_target[i]=self.new_particles[i]
            else:
                self.particles_target[i]=self.particles_target[i]
        return self.particles_target  

    
    

        


