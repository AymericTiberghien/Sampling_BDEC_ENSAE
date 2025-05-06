import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


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
            raws[i]=self.weights[i]*multivariate_normal.pdf(y, mean=self.means[i], cov=self.covs[i])

        # print("raws"+str(raws))
        pdf_vals = np.array(raws)          
        mix_pdf  = np.array(np.sum(pdf_vals))
        # print("pdf_vals"+str(pdf_vals))
        # print("mix_pdf"+str(mix_pdf))
        
        return pdf_vals, mix_pdf

    def V(self, y: np.ndarray):
        _ ,density_vals = self.density(y)
        potential_value= -np.log(density_vals).flatten()
        return float(potential_value)

    def grad_V(self, x):
        x = np.asarray(x, float)
    # 1) log-densités p_i = log w_i + log N_i(x)
        log_pis = np.array([
            np.log(w) + comp.logpdf(x)
            for w, comp in zip(self.weights, self.components)
        ])                              # shape (m,)

    # 2) log π(x) via logsumexp
        L = logsumexp(log_pis)         # scalaire

    # 3) responsabilités r_i = exp(log_pis - L)
        r = np.exp(log_pis - L)        # shape (m,), somme ≈ 1

    # 4) ∇V = ∑_i r_i Σ_i^{-1}(x - μ_i)
        grad = np.zeros_like(x)
        for r_i, mu_i, cov_i in zip(r, self.means, self.covs):
            invC = np.linalg.inv(cov_i)
            diff = x - mu_i
            grad += r_i * (invC @ diff)

        return grad

    def hess_V(self, x):
        x = np.asarray(x, float)
    # 1) log-densités
        log_pis = np.array([
            np.log(w) + comp.logpdf(x)
            for w, comp in zip(self.weights, self.components)
        ])
        L = logsumexp(log_pis)
        r = np.exp(log_pis - L)      # responsabilités

        d = x.size
        g = np.zeros(d)
        H = np.zeros((d, d))

        for r_i, mu_i, cov_i in zip(r, self.means, self.covs):
            invC = np.linalg.inv(cov_i)
            diff = (x - mu_i).reshape(-1,1)  # (d,1)

        # g_i = ∇ log p_i = -invC @ diff
            g_i = - (invC @ diff).flatten()
        # H_i = ∇^2 log p_i
            H_i = invC @ diff @ diff.T @ invC - invC

            g += r_i * g_i
            H += r_i * H_i

    # Hessian de V = - Hessian de log π
    # mais log π Hessien = H - g g^T
        H_logpi = H - np.outer(g, g)
        return H_logpi


    def langevin_update(self, y: np.ndarray) -> np.ndarray:
        for i in range(len(y)):
            grads = self.grad_V(y[i])
            # print(grads)
            noise = np.random.normal(0,1,2)
            y[i] = (y[i] - self.delta_t * self.temp * grads 
                    + np.sqrt(2*self.delta_t) * noise
                    )
        return y