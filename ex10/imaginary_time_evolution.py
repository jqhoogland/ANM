import numpy as np
import scipy as sp

from compute_energy import compute_energy

def get_heisenberg_local_hamiltonian(J=1., kind="4-tensor"):
    # 2-site Hamiltonian & unitary 2-site gate
    S_z = 0.5*np.array([ [1, 0], [0, -1] ])
    S_plus = np.array([ [0, 1], [0, 0] ])
    S_minus = np.array([ [0, 0], [1, 0] ])
    d = S_z.shape[0]

    local_hamiltonian_matrix = J*( np.kron(S_z, S_z) + 0.5*( np.kron(S_plus, S_minus) + np.kron(S_minus, S_plus) ) )
    local_hamiltonian_4_tensor = np.reshape(local_hamiltonian_matrix, [d,d,d,d])

    if kind == "matrix" or kind == "2-tensor":
        return local_hamiltonian_matrix
    elif kind == "tensor" or kind == "4-tensor":
        return local_hamiltonian_4_tensor
    else:
        raise ValueError("``kind`` must be one of ``['matrix', '2-tensor', 'tensor', '4-tensor']``. ``{}`` not recognized.".format(kind))

class ImaginaryTimeEvolution(object):
    def __init__(self, local_hamiltonian, bond_dim):
        # ``local_hamiltonian`` must be shape [d, d, d, d] where d is the local Hilbert space dimensionality ``local_dim``
        assert len(local_hamiltonian.shape) == 4

        self.local_dim = local_hamiltonian.shape[0]

        assert (self.local_dim == local_hamiltonian.shape[1]
                and self.local_dim == local_hamiltonian.shape[2]
                and self.local_dim == local_hamiltonian.shape[3])

        self.local_hamiltonian = local_hamiltonian
        self.bond_dim = bond_dim

    def _compute_energy(self, ket):
        compute_energy(ket, self.local_hamiltonian)

    def _contract(self, ket_1, ket_2, gate, direction):
        # Contract the tensor
        tens = np.tensordot(np.tensordot(ket1, gate, axes=[0,2]), ket2, axes=([1,4],[1,0]))
        tens = tens.reshape((tens.shape[0] * tens.shape[1], -1))

        # Perform the SVD
        u, s, v = np.linalg.svd(tens, full_matrices=False)

        # Restrict to the pre-determined bond dimension ``self.bond_dim``
        u = u[:, :self.bond_dim]
        s = s[:self.bond_dim]
        v = v[:self.bond_dim, :]

        # Combine ``s`` with either ``u`` or ``v`` depending on whether direction is left or right, respectively
        if direction == "right":
            v = np.dot(np.diag(s), v)
        elif direction == "left":
            u = np.dot(u, np.diag(s))
        else:
            raise ValueError("Direction must be one of ['right', 'left'], but has value {}".format(direction))

        # Keep coefficient of unity
        u = u / np.amax(u)
        v = v / np.amax(v)

        ket_1 = np.tranpose(u.reshape((-1, d, u.shape[1])), (1, 0, 2))
        ket_2 = np.tranpose(v.reshape((v.shape[1], d, -1)), (1, 0, 2))

        return ket_1, ket_2

    def sweep(self, ket, gate):
        for i in range(self.n_sites-1):
            ket[i], ket[i+1] = self._contract(ket[i], ket[i+1], gate, "right")

        for i in range(self.n_sites-2, -1, -1):
            ket[i], ket[i+1] = self._contract(ket[i], ket[i+1], gate, "left")

        return ket

    def get_energy_evolution_gate(self, time_step):
        return np.exp(-0.5 * time_step * self.local_hamiltonian)

    def evolve(self, init_ket, convergence_criteria, gate, tol=1e-4, max_n_sweeps=10000):
        n_sweeps = 0
        diff = np.inf
        ket = init_ket

        prev_criteria = convergence_criteria(init_ket)
        curr_criteria = prev_criteria

        while diff > tol and n_sweeps < max_n_sweeps:
            ket = self.sweep(ket, gate)

            prev_criteria = curr_criteria
            curr_critera = convergence_criteria(ket)

            diff = curr_criteria - prev_criteria

        return ket


    def find_groundstate(self, init_ket, time_step=1e-3, tol=1e-4, max_n_sweeps=10000):
        gate = self.get_energy_evolution_gate(time_step)
        convergence_criteria = self._compute_energy

        return self.evolve(init_ket, convergence_criteria, gate, tol, max_n_sweeps)
