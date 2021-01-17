"""

Write a program to perform a Monte Carlo integration of the function f(x) = exp(􀀀x2)
between 0 and 100 using N = 10000 samples. The exact result is 0.88623

"""

import numpy as np

# Some constants

N_SAMPLES = 10000

# 3.1.1 Use simple sampling to estimate the value and MC error

def simple_sampling(n_samples=N_SAMPLES):
    # Returns a tuple of the MC estimate and error

    simple_x = np.random.uniform(size=n_samples) * 100
    simple_fx = np.exp(-np.square(simply_sampled))
    return np.mean(simple_fx), np.std(simple_fx, ddof=1)


print("For simple sampling, the estimate is {} and the MC error is {}".format(simple_pred, simple_mc_err))

def repeated_sampling(n_iter=50, n_samples=N_SAMPLES):
    res = np.zeros(n_iter)

    for i in range(n_iter):
        res[i] = simple_sampling(n_samples)
