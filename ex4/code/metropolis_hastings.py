"""
Single-spin Metropolis-Hastings Monte Carlo Algorithm

Author: Jesse Hoogland

Note: Some of this code was inspired by my bachelor's thesis project, and can be found at: https://github.com/jqhoogland/rgpy
"""

import math, sys

import numpy as np

# ------------------------------------------------------------

# Two classes for generating metropolis-hastings samples of the Ising model in 2D

# ------------------------------------------------------------

class IsingModel2D(object):
    """
    Ising model with the Hamiltonian:

    $$ H = - J \sum_{<i, j>} s_i s_j - h \sum_{i} s_i$$

    """

    def __init__(self,
                 lattice_width,
                 J,
                 h=0):
        self.lattice_width = lattice_width
        self.n_spins = self.lattice_width ** 2
        self.J = J
        self.h = h
        self.state = np.random.choice([-1., 1.], size=[lattice_width, lattice_width])

    def step(self):
        """
        Take one MC step.

        """

        i = np.random.randint(self.lattice_width)
        j = np.random.randint(self.lattice_width)
        s_ij = self.state[i, j]

        nearest_neighbors = [self.state[(i, (j + 1) % self.lattice_width)],
                             self.state[(i, (j - 1) % self.lattice_width)],
                             self.state[((i + 1) % self.lattice_width, j)],
                             self.state[((i - 1) % self.lattice_width, j)]]

        delta_energy = 2. * self.J * s_ij * np.sum(nearest_neighbors)

        #print(s_ij, nearest_neighbors, delta_energy)
        # If the energy difference is negative (thus always accepted), the following condition is also satisfied.
        # i.e. we do not need to use a `min(np.exp(delta_energy), 1)`
        if delta_energy <= 0 or np.random.uniform() < np.exp(-delta_energy):
            # print("Accepted")
            #print(self.state)
            self.state[i, j] = -s_ij
            #print(self.state)

    def get_state(self):
        return self.state

    def set_J(self, J):
        self.J = J


class SamplerMCMC(object):
    def __init__(self, model, n_samples):
        self.model = model # In principle SamplerMCMC could work with any lattice model, provided it has a `step` function
        self.n_samples = n_samples
        self.samples = np.zeros([n_samples, self.model.lattice_width, self.model.lattice_width])

    def run_MCMC(self, n_thermalization_steps, n_steps_between_samples):
        # Thermalizing/Burn-in
        for _ in range(n_thermalization_steps):
            self.model.step()

        # Generate and store the actual samples
        for i in range(self.n_samples):
            for _ in range(n_steps_between_samples + 1):
                self.model.step()

            self.samples[i] = self.model.get_state()

        return self.samples

    @staticmethod
    def binning_analysis(measurements, stop_n_before=3):
        # Used in the error analysis, implemented for assignment 3
        bins = np.zeros(int(np.log2(measurements.size))-stop_n_before)
        binned = measurements
        bins[0] = np.std(binned, ddof=1)/ np.sqrt(binned.size)

        for i in range(1, bins.size):
            binned = (binned + np.roll(binned, 1))[1::2]/2
            bins[i] = np.std(binned, ddof=1) / np.sqrt(binned.size)

        return bins

    def get_mcmc_error(self, measurements, stop_n_before=3):
        # we assume that the binning analysis has already converged within log_2 n_samples - 3 steps (=11 for this problem)
        bins = self.binning_analysis(measurements, stop_n_before)

        return bins[-1]

    def get_measurements(self, fn):
        # This is basically just a wrapper around a `map` function over `self.samples`
        measurements = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            measurements[i] = fn(self.samples[i], J=self.model.J, h=self.model.h)

        return measurements

    def get_expectation(self, fn, include_error=True):
        # Average over measurements of some observable with an optional error analysis
        measurements = self.get_measurements(fn)

        if not include_error:
            return np.mean(measurements)

        return np.mean(measurements), self.get_mcmc_error(measurements)

    def get_correlation_times(self, fn):
        measurements = self.get_measurements(fn)

        naive_error = np.std(measurements, ddof=1)
        true_error = self.get_mcmc_error(measurements)

        return ((true_error / naive_error) ** 2 - 1) / 2

    def get_filename(self):
        return "./samples/mcmc_L{}_J{}.np".format(self.model.lattice_width, str(self.model.J).replace('.', ''))

    def save_samples(self):
        self.samples.dump(self.get_filename())

    def load_samples(self, filename=None):
        if filename == None:
            filename = self.get_filename()

        self.samples = np.load(filename, allow_pickle=True)

# ------------------------------------------------------------

# A few functions for computing observables on given samples

# ------------------------------------------------------------

def get_magnetization(sample, J, h, per_site=True):
    magnetization = np.sum(sample)

    if per_site:
        magnetization /= np.size(sample)

    return magnetization

def get_order_param(sample, J, h, per_site=True):
   return np.abs(get_magnetization(sample, J, h, per_site))

def get_magnetization_squared(sample, J, h, per_site=False):
    return np.square(get_magnetization(sample, J, h, per_site))


def get_energy(sample, J, h, per_site=True):

    # divide by 2 for over-counting
    energy = -0.5 * np.sum(sample * (np.roll(sample, 1, axis= -1)
                                       +  np.roll(sample, 1, axis= -2)
                                       +  np.roll(sample, -1, axis= -1)
                                       + np.roll(sample, -1, axis= -2)))

    if per_site:
        energy /= np.size(sample)

    return energy
