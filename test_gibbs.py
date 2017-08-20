import numpy as np
import scipy.stats
import pytest
import copy

from gibbs_simple import gibbs_simple
from gibbs_modularized import Model, State, Storage

@pytest.fixture
def random_model():
    theta_prior = scipy.stats.norm(loc=np.random.uniform(),
                                   scale=np.random.uniform())
    sigma2_prior = scipy.stats.invgamma(a=np.random.uniform(),
                                        scale=np.random.uniform())
    return Model(theta_prior=theta_prior,
                 sigma2_prior=sigma2_prior)

@pytest.fixture
def random_state():
    return State(np.random.uniform(), np.random.uniform())

@pytest.fixture
def random_data():
    return np.random.normal(size=10)

def test_cond_theta(random_model, random_state, random_data):

    cond = random_model.cond_theta(random_state, random_data)

    new_state = copy.deepcopy(random_state)
    new_state.theta = np.random.uniform()

    assert np.allclose(cond.logpdf(new_state.theta) - cond.logpdf(random_state.theta),
                       random_model.joint_log_p(new_state, random_data) - \
                       random_model.joint_log_p(random_state, random_data))

def test_cond_sigma2(random_model, random_state, random_data):

    cond = random_model.cond_sigma2(random_state, random_data)

    new_state = copy.deepcopy(random_state)
    new_state.sigma2 = np.random.uniform()

    assert np.allclose(cond.logpdf(new_state.sigma2) - cond.logpdf(random_state.sigma2),
                       random_model.joint_log_p(new_state, random_data) - \
                       random_model.joint_log_p(random_state, random_data))

def test_gibbs_simple(random_model, random_data):
    random_model.rng = 42

    # Run 1 iteration of gibbs_modular
    samples_gibbs_modular = Storage()
    my_state = State(theta=random_data.mean(), sigma2=random_data.var())
    samples_gibbs_modular.add_state(my_state)
    for i in range(1):
        random_model.gibbs_step(my_state, random_data)
        samples_gibbs_modular.add_state(my_state)

    # Run 1 iteration of gibbs_simple
    prior = {'mu_0': random_model.theta_prior.kwds['loc'],
             'tau2_0': random_model.theta_prior.kwds['scale'] ** 2,
             'nu_0': random_model.sigma2_prior.kwds['a'] * 2,
             'sigma2_0': random_model.sigma2_prior.kwds['scale'] / random_model.sigma2_prior.kwds['a']}
    samples_gibbs_simple = gibbs_simple(S=2, y=random_data, prior=prior, rng=42)

    # Compare the results of the two Gibbs
    print(samples_gibbs_simple['theta'], samples_gibbs_modular.theta)
    assert np.allclose(samples_gibbs_simple['theta'], samples_gibbs_modular.theta)
    assert np.allclose(samples_gibbs_simple['sigma2'], samples_gibbs_modular.sigma2)
