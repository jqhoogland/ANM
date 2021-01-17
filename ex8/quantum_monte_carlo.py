import numpy as np
import scipy as sp

class GTO(object):
    def __init__(self, coeffs=np.array([0.1219492, 0.444529, 1.962079, 13.00773])):
        """

        """
        assert len(coeffs.shape) == 1

        self.coeffs = coeffs

    def _get_inv_ai_plus_bj(self, n_terms):
        return 1 / (self.coeffs[:n_terms].reshape((1, -1)) + self.coeffs[:n_terms].reshape(-1, 1))

    def get_overlap_term(self, n_terms):
        self.overlap_term = np.power(np.pi * self._get_inv_ai_plus_bj(n_terms), 3. / 2)

    def get_kinetic_term(self, n_terms):
        return 3 * (self.coeffs[:n_terms].reshape((1, -1)) * self.coeffs[:n_terms]) * np.power(np.pi, 5. / 3) * np.power( self._get_inv_ai_plus_bj(n_terms), 5. / 2)

    def get_coulomb_term(self, n_terms):
        return -2 * np.pi * self._get_inv_ai_plus_bj(n_terms)

    def get_hamiltonian(self, n_terms):
        return self.get_kinetic_term(n_terms) + self.get_coulomb_term

    def get_ground_state_energy(self, n_terms):
        return sp.linalg.eigh(self.get_hamiltonian(n_terms), self.overlap_term, eigvals=(0, n_eigvals))[0][0]

    def plot_wave_function(self, r_ini, r_fin, r_step):
        # plot.plot() stuff


def test_GTO_initialization():
    gto = GTO()

    assert gto.coeffs.shape == (4, )
    assert gto.overlap_term.shape == (4, 4)
    assert gto.kinetic_term.shape == (4, 4)
    assert gto.coulomb_term.shape == (4, 4)
