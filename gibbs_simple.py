import numpy as np
import scipy.stats

def gibbs_simple(S, y, prior, rng=None):
    """
    A no-frill Gibbs sampler

    Parameters
    ----------
    S : int, the number of samples
    y : data vector
    prior : dict, prior parameters
    rng : int, random seed
    """
    n = len(y)
    ybar = y.mean()

    # Initialize storage
    theta_samples = np.empty(S)
    sigma2_samples = np.empty(S)

    # Starting value as the sample variance and mean
    sigma2_samples[0] = y.var()
    theta_samples[0] = ybar

    # Big loop
    for s in range(1, S):
        # Update theta
        tau2_n = 1 / (1 / prior['tau2_0'] + n / sigma2_samples[s - 1])
        mu_n = tau2_n * (prior['mu_0'] / prior['tau2_0'] + n * ybar / sigma2_samples[s - 1])
        theta_samples[s] = scipy.stats.norm(mu_n, np.sqrt(tau2_n)).rvs(random_state=rng)

        # Update sigma2
        nu_n = prior['nu_0'] + n
        nu_sigma2_n = prior['nu_0'] * prior['sigma2_0'] + sum((y - theta_samples[s]) ** 2)
        sigma2_samples[s] = scipy.stats.invgamma(nu_n / 2, scale=nu_sigma2_n / 2).rvs(random_state=rng)

    return {'theta': theta_samples, 'sigma2': sigma2_samples}
