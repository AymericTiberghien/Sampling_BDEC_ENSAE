import numpy as np
from scipy.optimize import minimize
from ula import GaussianMixtureULA



class exploration_component:
    def __init__(self, M,S,W, thr, B, N,weights, means, covs,temp, delta_t, T,
                 particles_hot):
        self.M = M
        self.S = S  
        self.W = W
        self.thr = thr
        self.B = B
        self.N = N
        self.weights = weights
        self.means = means 
        self.covs = covs
        self.temp = temp
        self.delta_t = delta_t
        self.T = T
        self.particles_hot = particles_hot
        self.sampler=GaussianMixtureULA(weights, means, covs, temp, delta_t)

    def y_barre(self):
        k = np.random.choice(self.N, self.B)
        y_barre = self.particles_hot[k]
        return y_barre
    
    def nearest_spd(self,A,epsilon=1e-8):
        eigvals, eigvecs = np.linalg.eigh(A)
    
        eigvals_clipped = np.clip(eigvals, a_min=epsilon, a_max=None)
    
        return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    def find_mode_quasi_newton(self,b):
        y_b_barre = np.array(self.y_barre())[b]
        res= minimize(fun=self.sampler.V,
                      x0=y_b_barre,
                      jac=self.sampler.grad_V,
                      method='BFGS',
                      options={
                        "gtol": 1e-4, #Could consider setting to a lower value for more accurate convergence but iterations take too long
                        "eps":1e-6,      
                        "maxiter": 200 #Consider setting to a lower value for faster iterations but careful trade-off with convergence  
                    }
                      )
        mu_b_barre = res.x
        sigma_b_barre=-np.linalg.inv(self.sampler.hess_V(mu_b_barre))
        sigma_b_barre=self.nearest_spd(sigma_b_barre,epsilon=1e-8)
        return mu_b_barre, sigma_b_barre
    
    def dista(self,b):
        mu_b_barre, sigma_b_barre = self.find_mode_quasi_newton(b)
        d=np.zeros(len(self.M))
        for i in range(len(self.M)):
            diff=self.M[i]-mu_b_barre
            d[i]=  1/2 *max((diff.T @ np.linalg.inv(sigma_b_barre) @ diff),
                            diff.T@np.linalg.inv(self.S[2*i:2*i+2,:])@diff)
        return d
    
    def update_mode(self,b):
        mu_b_barre, sigma_b_barre = self.find_mode_quasi_newton(b)
        d=self.dista(b)
        if min(d)>self.thr:
            self.M=np.vstack((self.M,mu_b_barre))
            self.S=np.vstack((self.S,sigma_b_barre))
            new_weights=np.zeros(len(self.M))
            _,density_vals=self.sampler.density(self.M)
            denom=0
            for i in range(len(self.M)):
                denom+=density_vals[i]*np.sqrt(np.abs(np.linalg.det(
                    self.S[2*i:2*i+2,:])))
            for i in range(len(self.M)):
                new_weights[i]=density_vals[i]*np.sqrt(np.abs(
                    np.linalg.det(self.S[2*i:2*i+2,:])))/denom
            self.W=new_weights
        

    def mode_finder(self):
        is_new=False
        for b in range(self.B):
            d=self.dista(b)
            self.update_mode(b)
            if min(d)>self.thr:
                is_new=True
        return is_new, self.M, self.S, self.W

