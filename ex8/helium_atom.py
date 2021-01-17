# Variational solution of the helium atom

import numpy as np
import scipy.linalg as la
from numpy import dot

COEFFS = [0.297104, 1.236745, 5.749982, 38.216677]

class HeAtom(object):

    def __init__(self, coeffs):
        self.max_n_coeffs = len(coeffs)
        self.coeffs = np.array(coeffs)

        self.overlap = np.array([[self.s_ij(p,q) for q in range(N)] for p in range(N)]) # overlap matrix
        self.kinetic = np.array([[self.t_ij(p,q) for q in range(N)] for p in range(N)]) # non-interacting matrix
        self.potential = np.array([[[[self.v_ijkl(i,j,k,l) for i in range(N)] for j in range(N)]
                                for k in range(N)] for l in range(N)]) # Hartree matrix


    def s_ij(self, p,q):
        """overlap matrix elements"""
        return (np.pi/(self.coeffs[p]+self.coeffs[q]))**1.5

    def t_ij(self, p,q):
        """non-interacting matrix elements"""
        return 3*self.coeffs[p]*self.coeffs[q]*np.pi**1.5/(self.coeffs[p]+self.coeffs[q])**2.5 - 4*np.pi/(self.coeffs[p]+self.coeffs[q])

    def v_ijkl(self, i,j,k,l):
        """Hartree matrix elements"""
        return 2*np.pi**2.5/(self.coeffs[i]+self.coeffs[j])/(self.coeffs[k]+self.coeffs[l])/np.sqrt(self.coeffs[i]+self.coeffs[j]+self.coeffs[k]+self.coeffs[l])

    def get_energy(self, d):
        return 2*dot(dot(self.kinetic, d), d) + dot(dot(dot(dot(self.potential, d), d), d), d)

    def get_ground_state(self, tol=1e-6, max_n_iter=100):
        dot = lambda x, y: np.dot(x, y)

        # Prepare and initialize an initial vector of components
        d = np.ones(self.max_n_coeffs)
        d /= dot(d,dot(s,d))

        prev_eps = 0
        curr_eps = 0

        ground_state_energy = np.inf

        for k in range(maxiter):
            fock_operator = self.kinetic + dot(dot(self.potential, d), d)
            eigvals, eigvecs = la.eigh(fock_operator, self.overlap)
            min_index = np.argmin(eigvals)

            prev_eps = curr_eps
            curr_eps = eigvals[min_index]

            # Update and renormalize the vector of components
            d  = eigvecs[:, min_index]
            d /= np.sqrt(dot(d, dot(self.overlap,d)))

            ground_state_energy = self.get_energy(d)

            print('Iteration:{},  eps:{},  energy:{}'.format(k, eps, ground_state_energy))

            if(abs(curr_eps - prev_eps) < tol):
                break

        return ground_state_energy
