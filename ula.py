import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

#This file is the one that is very-specific to the example taken as we compute 
# the density values, its potential V, the gradient and the Hessien of the 
# target distribution chosen in this class. These functions are thus very 
# specific to the example.

class GaussianMixtureULA:
    """
    Classe combinant la définition de la distribution cible (mixture gaussienne)
    et l'algorithme ULA (Unadjusted Langevin Algorithm).
    """
    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        temp: float,
        delta_t: float
    ):
       
        self.weights = weights
        self.means = means
        self.covs = covs
        
        self.components = [
            multivariate_normal(mean=means[i], cov=covs[i])
            for i in range(len(weights))
        ]
       
        self.temp = temp
        self.delta_t = delta_t

    def density(self,y):

        raws=[0,0,0,0]
        for i in range(4):
            raws[i]=self.weights[i]*multivariate_normal.pdf(y, 
                                                            mean=self.means[i],
                                                              cov=self.covs[i])

        pdf_vals = np.array(raws)          
        mix_pdf  = np.array(np.sum(pdf_vals))
        
        return pdf_vals, mix_pdf
    

    def V(self, y: np.ndarray):
        _ ,density_vals = self.density(y)
        potential_value= -np.log(density_vals).flatten()
        return float(potential_value)
    
    #In the two following functions, we use the logsumexp function in order to 
    # avoid dividing by 0 when computing the gradient and the hessien. 
    # Without it, it leads to langevin update not working correctly

    def grad_V(self, x):
        x = np.asarray(x, float)
        log_pis = np.array([
            np.log(w) + comp.logpdf(x)
            for w, comp in zip(self.weights, self.components)
        ])                             

        L = logsumexp(log_pis)        

   
        r = np.exp(log_pis - L)        

   
        grad = np.zeros_like(x)
        for r_i, mu_i, cov_i in zip(r, self.means, self.covs):
            invC = np.linalg.inv(cov_i)
            diff = x - mu_i
            grad += r_i * (invC @ diff)

        return grad

    def hess_V(self, x):
        x = np.asarray(x, float)
    
        log_pis = np.array([
            np.log(w) + comp.logpdf(x)
            for w, comp in zip(self.weights, self.components)
        ])
        L = logsumexp(log_pis)
        r = np.exp(log_pis - L)      

        d = x.size
        g = np.zeros(d)
        H = np.zeros((d, d))

        for r_i, mu_i, cov_i in zip(r, self.means, self.covs):
            invC = np.linalg.inv(cov_i)
            diff = (x - mu_i).reshape(-1,1)  

        
            g_i = - (invC @ diff).flatten()
       
            H_i = invC @ diff @ diff.T @ invC - invC

            g += r_i * g_i
            H += r_i * H_i

    
        H_logpi = H - np.outer(g, g)
        return H_logpi

    #This function uses the gradient previously computed to update the particles.
    #Implements straight-forwardly Algorithm 2 from the paper. It depends on the
    #  temperature of the population of the particles being updated.

    def langevin_update(self, y: np.ndarray) -> np.ndarray:
        for i in range(len(y)):
            grads = self.grad_V(y[i])
            noise = np.random.normal(0,1,2)
            y[i] = (y[i] - self.delta_t * (1/self.temp) * grads 
                    + np.sqrt(2*self.delta_t) * noise
                    )
        return y