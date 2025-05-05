import numpy as np
from scipy.stats import multivariate_normal


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

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[np.newaxis, :]           

       
        raws = []
        for w, comp in zip(self.weights, self.components):
            val = comp.pdf(y)             
            val = np.atleast_1d(val)      
            raws.append(w * val)

       
        pdf_vals = np.stack(raws, axis=1)          
        mix_pdf  = pdf_vals.sum(axis=1) 
        
        return pdf_vals, mix_pdf

    def V(self, y: np.ndarray):
        _ ,density_vals = self.density(y)
        potential_value= -np.log(density_vals).flatten()
        return float(potential_value)

    def grad_V(self, y: np.ndarray):
        y = np.atleast_2d(y)

        pdf_vals, density_vals = self.density(y)
        
        grad_num = np.zeros_like(y)
        for i in range(len(self.weights)):
            diff = y - self.means[i]
            inv_cov = np.linalg.inv(self.covs[i])
            grad_num += pdf_vals[:, i:i+1] * (diff @ inv_cov.T)
        temp=-grad_num / density_vals
        return temp[0]

    def hess_V(self,y):
        pdf_vals, density_vals = self.density(y)
        terme_1_num=0
        terme_2_num=0
        for i in range(len(self.weights)):
            diff = y - self.means[i]
            inv_cov = np.linalg.inv(self.covs[i])
            terme_1_num += pdf_vals[:, i:i+1] * (inv_cov@np.outer(diff,diff)@inv_cov -inv_cov)
            terme_2_num += pdf_vals[:, i:i+1] * (inv_cov@diff)
        terme_1=terme_1_num/density_vals
        terme_3_num=terme_2_num@terme_2_num.T
        terme_3=terme_3_num/density_vals**2
        hess=terme_1-terme_3
        return hess


    def langevin_update(self, y: np.ndarray) -> np.ndarray:
        for i in range(len(y)):
            grads = self.grad_V(y[i])
            noise = np.random.normal(0,1,2)
            y[i] = (y[i] - self.delta_t * self.temp * grads 
                    + np.sqrt(2*self.delta_t) * noise
                    )
        return y