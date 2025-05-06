import numpy as np
from scipy.optimize import minimize
from ula import GaussianMixtureULA

# This file is used to implement Algorithm 3 from the paper: the exploration 
# component.

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

    #From the current population of 'hot' particles we extract a batch from it.

    def y_barre(self):
        k = np.random.choice(self.N, self.B)
        y_barre = self.particles_hot[k]
        return y_barre
    
    #This function is used to make sure that the hessian matrix coming from the 
    # following quasi-Newton optimization function


    def nearest_spd(self,A, epsilon=1e-8):
    
        A = np.array(A)
    
    # Étape 1 : symétrisation (utile si A n'est pas symétrique)
        A_sym = (A + A.T) / 2
    
    # Étape 2 : décomposition spectrale
        eigvals, eigvecs = np.linalg.eigh(A_sym)
    
    # Étape 3 : forcer les valeurs propres à être strictement positives
        eigvals_clipped = np.clip(eigvals, a_min=epsilon, a_max=None)
    
    # Étape 4 : reconstruction de la matrice SPD
        A_spd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    # Étape 5 : re-symétrisation pour éviter l’asymétrie numérique
        A_spd = (A_spd + A_spd.T) / 2

        return A_spd

        
    
        
    #From the batch of hot particles previously selected, we perform 
    # quasi-Newton optimization to obtain a potentially new mode candidate
    #as well as the covariance matrix linked to it. We chose the BFGS method for
    # the quasi-Newton optimization, other could be tested as well
    
    def find_mode_quasi_newton(self,b):
        y_b_barre = np.array(self.y_barre())[b]
        res= minimize(fun=self.sampler.V,
                      x0=y_b_barre,
                      jac=self.sampler.grad_V,
                      method='BFGS',
                      options={
                        "gtol": 1e-4, #Could consider setting to a lower value 
                                      #for more accurate convergence but 
                                      # iterations take too long. Impact of this
                                      #  parameter is TBD
                        "eps":1e-6,      
                        "maxiter": 200 #Consider setting to a lower value for 
                                       #faster iterations but careful trade-off 
                                       # with convergence. Around 4 
                                       # minutes/iteration if set to 50 (J=10) and 
                                       # around 3 when set to 200 (J=5). Might 
                                       # not impact that much the duration of 
                                       # the iteration in the end. (TBD)
                    }
                      )
        mu_b_barre = res.x
        sigma_b_barre=-np.linalg.inv(self.nearest_spd(self.sampler.hess_V(mu_b_barre), epsilon=1e-8))
        # sigma_b_barre=self.nearest_spd(sigma_b_barre,epsilon=1e-8)
        return mu_b_barre, sigma_b_barre
    
    #This function implements the distance introduced in the paper to judge if 
    # the candidate mode is actually new.
    
    def dista(self,b):
        mu_b_barre, sigma_b_barre = self.find_mode_quasi_newton(b)
        d=np.zeros(len(self.M))
        for i in range(len(self.M)):
            diff=self.M[i]-mu_b_barre
            d[i]=  1/2 *max((diff.T @ np.linalg.inv(sigma_b_barre) @ diff),
                            diff.T@np.linalg.inv(self.S[2*i:2*i+2,:])@diff)
        return d
    
    #For each candidate mode we compute the distance from it to every current 
    # mode. If it is far enough from at least one current mode, we update the 
    # information we have on the modes with this candidate mode, its hessian. 
    # Then, we recalculate the weights of each mode always following the 
    # formulae given in the article.
    # thr is the threshold given in the article that depends on the 
    # dimensionality of the target distribution, in our case it is equal to 2.

    def update_mode(self,b):
        mu_b_barre, sigma_b_barre = self.find_mode_quasi_newton(b)
        d=self.dista(b)
        if min(d)>self.thr:
            self.M=np.vstack((self.M,mu_b_barre))
            self.S=np.vstack((self.S,sigma_b_barre))
            new_weights=np.zeros(len(self.M))
            density_vals=[self.sampler.density(self.M[i])[1] 
                          for i in range(len(self.M))]
            denom=0
            for i in range(len(self.M)):
                denom+=density_vals[i]*np.sqrt(np.abs(np.linalg.det(
                    self.S[2*i:2*i+2,:])))
            for i in range(len(self.M)):
                new_weights[i]=density_vals[i]*np.sqrt(np.abs(
                    np.linalg.det(self.S[2*i:2*i+2,:])))/denom
            self.W=new_weights
        
    #If at least one new mode is found, we update the variable is_new.

    def mode_finder(self):
        is_new=False
        for b in range(self.B):
            d=self.dista(b)
            self.update_mode(b)
            if min(d)>self.thr:
                is_new=True
        return is_new, self.M, self.S, self.W

