import numpy as np
import scipy.stats
import copy

class State(object):
    def __init__(self, theta, sigma2):
        self.theta = theta
        self.sigma2 = sigma2

class Storage(object):
    def __init__(self):
        self.states = []

    def add_state(self, state):
        self.states.append(copy.copy(state))

    def __getattr__(self, param):
        return np.array([getattr(state, param) for state in self.states])

class Model(object):
    def __init__(self, theta_prior, sigma2_prior, rng=None):
        self.theta_prior = theta_prior
        self.sigma2_prior = sigma2_prior
        self.rng = rng

    # Full Conditional for theta
    def cond_theta(self, state, data):
        n = len(data)
        mean_prior = self.theta_prior.kwds['loc']
        variance_prior = self.theta_prior.kwds['scale'] ** 2

        variance_post = 1 / (1 / variance_prior + n / state.sigma2)
        mean_post = variance_post * (mean_prior / variance_prior + n * data.mean() / state.sigma2)
        return scipy.stats.norm(mean_post, np.sqrt(variance_post))

    # Full Conditional for sigma2
    def cond_sigma2(self, state, data):
        n = len(data)
        shape_prior = self.sigma2_prior.kwds['a']
        scale_prior = self.sigma2_prior.kwds['scale']

        shape_post = shape_prior + n / 2
        scale_post = scale_prior + np.sum((data - state.theta) ** 2) / 2
        return scipy.stats.invgamma(shape_post, scale=scale_post)

    # A gibbs step iterate through theta, sigma2 and update them
    def gibbs_step(self, state, data):
        state.theta = self.cond_theta(state, data).rvs(random_state=self.rng)
        state.sigma2 = self.cond_sigma2(state, data).rvs(random_state=self.rng)

    # Joint density, used as the right hand side in the identity to be used in the unit test
    def joint_log_p(self, state, data):
        return self.theta_prior.logpdf(state.theta) + \
               self.sigma2_prior.logpdf(state.sigma2) + \
               (scipy.stats.norm(loc=state.theta, scale=np.sqrt(state.sigma2))
                .logpdf(data).sum())
