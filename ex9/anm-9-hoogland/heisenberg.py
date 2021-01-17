"""

Advanced Numerical Methods in Many Body Physics

Exercise 7

Author: Jesse Hoogland

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

class Heisenberg(object):
    """
    This class represent a 1-dimensional Heisenberg spin chain.

    The Heisenberg Hamiltonian is :math:`\hat H = J \sum_{\langle i, j \rangle} \hat S_i^z \hat S_j^z + \frac{1}{2}(\hat S_i^+ \hat S_j^- + \hat S_i^- \hat S_j^+)`.

    This class represents either a spin-1/2 or spin-1 chain, as implemented in
    child classes.

    This class implements direct diagonalization and will therefore only work
    with relatively smaller chains.

    This class implements both sparse and dense matrix calculation, as specified
    by the ``mode`` argument.

    """

    def __init__(self, n_spins, bdry_conds, J, mode):
        """
        :param n_spins: Integer number of spins in the chain.
        :param bdry_conds: String describing the boundary conditions of the
            chain. One of either ``"open"`` or ``"periodic"``.
        :param J: Float coupling strength of the chain.
        :param mode: String describing the array processing mode. One of
            either ``"dense"`` or ``"sparse"``.
        """

        self.J = J
        self.n_spins = n_spins
        self.dimensionality = int((2 * self.spin + 1) ** self.n_spins)
        self.mode = mode
        self.init_spin_matrices()

        if bdry_conds == "open":
            self.neighbors = [[i, i + 1] for i in range(self.n_spins -1)]
        elif bdry_conds == "periodic":
            self.neighbors = [[i, i + 1] for i in range(self.n_spins -1)]
            self.neighbors.append([0, self.n_spins - 1])
        else:
            raise ValueError("Invalid value provided for `bdry_conds`. Should be one of `open` or `periodic`")

        if self.mode == "sparse":
            self.kron = lambda a, b: sp.kron(a, b, "csr")
            self.dot = lambda a, b: a.dot(b)
        elif self.mode == "dense":
            self.kron = lambda a, b: np.kron(a, b)
            self.dot = lambda a, b: np.dot(a, b)
        else:
            raise ValueError("Invalid value provided for `mode`. Should be one of `dense` or `sparse`")

        self.init_hamiltonian()

    def _init_spin_matrices(self):
        """
        Creates 4 matrices: an identity matrix, S_z matrix, S_plus matrix, and
        S_minus matrix.

        As these depend on the spin of the system (with dimensionality= 2s+1,
        where s is the spin), this is implemented in the child class.
        """

        raise NotImplementedError

    def init_spin_matrices(self):
        """
        Given numpy spin matrices produced by ``self._init_spin_matrices``
        (implemented in a child class), this converts those matrices to
        sparse format if the provided mode is ``sparse``.
        """

        [identity, S_z, S_plus, S_minus] = self._init_spin_matrices()

        if self.mode == "sparse":
            identity = sp.csr_matrix(identity)
            S_z = sp.csr_matrix(S_z)
            S_plus = sp.csr_matrix(S_plus)
            S_minus = sp.csr_matrix(S_minus)

        self.identity = identity
        self.S_z = S_z
        self.S_plus = S_plus
        self.S_minus = S_minus

    def get_ij_term(self, i, j, S_i, S_j):
        """
        Calculates the matrix representing the evaluation of spin-operator ``S_i``
        on spin ``i`` and spin-operator ``S_j`` on spin ``j``, with identity on
        all other spins.
        """
        res = 1 # first scalar, then 2x2 matrix, then 4x4, 8x8... (if spin-1/2)

        for idx in range(self.n_spins):
            if (idx == i):
                res = self.kron(S_i, res)
            elif (idx == j):
                res = self.kron(S_j, res)
            else:
                res = self.kron(self.identity, res)

        return res

    def get_diagonal_term(self, i, j):
        """
        Evaluates :math:`S_z \otimes S_z` on spins ``i`` and ``j``
        """

        return self.get_ij_term(i, j, self.S_z, self.S_z)

    def get_plus_minus_term(self, i, j):
        """
        Evaluates :math:`S_plus \otimes S_minus` on spins ``i`` and ``j``
        (in same order)
        """

        return self.get_ij_term(i, j, self.S_plus, self.S_minus)

    def get_minus_plus_term(self, i, j):
        """
        Evaluates :math:`S_minus \otimes S_plus` on spins ``i`` and ``j``
        (in same order)
        """

        return self.get_ij_term(i, j, self.S_minus, self.S_plus)

    def get_cross_term(self, i, j):
        """
        Evaluates :math:`\frac{1}{2}( S_plus \otimes S_minus + S_minus \otimes S_plus)`
        on spins ``i`` and ``j``
        """

        return (self.get_plus_minus_term(i, j) + self.get_minus_plus_term(i, j)) / 2

    def init_hamiltonian(self):
        """
        Prepares the Heisenberg Hamiltonian in matrix form.
        """

        hamiltonian = np.zeros([self.dimensionality, self.dimensionality])

        if self.mode == "sparse":
            hamiltonian = sp.csr_matrix(hamiltonian)

        for [i, j] in self.neighbors:
            hamiltonian += self.J * (self.get_diagonal_term(i, j) + self.get_cross_term(i, j))

        self.hamiltonian = hamiltonian

    def diagonalize(self, verbose=True, n_eigvals=None):
        """
        Returns the first ``n_eigvals`` eigenvalues (in increasing energy order)
        along with their corresponding eigenvectors.
        """

        if n_eigvals is None:
            n_eigvals = self.dimensionality

        if verbose:
            print("Heisenberg chain of {} spin-{} particles with J={}".format(self.n_spins, self.spin, self.J))
            print("Hamiltonian (in matrix form):\n{}\n".format(self.hamiltonian))

        if self.mode == "dense":
            heisenberg_eigvals, heisenberg_eigvecs = np.linalg.eigh(self.hamiltonian)
            heisenberg_eigvals = heisenberg_eigvals[:n_eigvals]
            heisenberg_eigvecs = heisenberg_eigvecs[:, :n_eigvals]
            n_bytes = self.hamiltonian.nbytes
        elif self.mode == "sparse":
            heisenberg_eigvals, heisenberg_eigvecs = spl.eigsh(self.hamiltonian, n_eigvals, which='SA')
            n_bytes = self.hamiltonian.data.nbytes

        if verbose:
            print("Number of bytes: {}".format(n_bytes))
            print("Eigenvalues:\n{}\n".format(heisenberg_eigvals))
            print("With corresponding eigenvectors (respectively):\n{}".format(heisenberg_eigvecs))

        return heisenberg_eigvals, heisenberg_eigvecs

    def get_energy(self, state):
        state = state.reshape([np.prod(state.shape), 1])
        res = np.dot((state.T * self.hamiltonian), state)
        assert res.shape == (1,1)
        return res[0,0]

class HeisenbergSpinHalf(Heisenberg):
    def __init__(self, n_spins=2, bdry_conds="open", J=1., mode="dense"):
        """
        :param n_spins: Integer number of spins in the chain (default=``2``).
        :param bdry_conds: String describing the boundary conditions of the
            chain. One of either ``"open"`` or ``"periodic"`` (default=``"open"``).
        :param J: Float coupling strength of the chain (default=``1.``).
        :param mode: String describing the array processing mode. One of
            either ``"dense"`` or ``"sparse"`` (default=``"dense"``).
        """

        self.spin = 0.5
        super(HeisenbergSpinHalf, self).__init__(n_spins, bdry_conds, J, mode)

    def _init_spin_matrices(self):
        """
        Creates 4 matrices: an identity matrix, S_z matrix, S_plus matrix, and
        S_minus matrix.

        For the spin-1/2 system these have size 2x2.
        """

        identity = np.array([[1., 0], [0, 1.]])
        S_z = np.array([[1.,0],[0, -1.]])/2.
        S_plus= np.array([[0,1.],[0,0]])
        S_minus=np.array([[0,0],[1.,0]])

        return [identity, S_z, S_plus, S_minus]

class HeisenbergSpinOne(Heisenberg):
    def __init__(self, n_spins=2, bdry_conds="open", J=1, mode="dense"):
        """
        :param n_spins: Integer number of spins in the chain (default=``2``).
        :param bdry_conds: String describing the boundary conditions of the
            chain. One of either ``"open"`` or ``"periodic"`` (default=``"open"``).
        :param J: Float coupling strength of the chain (default=``1.``).
        :param mode: String describing the array processing mode. One of
            either ``"dense"`` or ``"sparse"`` (default=``"dense"``).
        """

        self.spin = 1
        super(HeisenbergSpinOne, self).__init__(n_spins, bdry_conds, J, mode)

    def _init_spin_matrices(self):
        """
        Creates 4 matrices: an identity matrix, S_z matrix, S_plus matrix, and
        S_minus matrix.

        For the spin-1/2 system these have size 3x3.
        """

        identity = np.array([[1., 0, 0],[0, 1., 0], [0, 0, 1.]])
        S_z = np.array([[1.,0,0],[0,0,0],[0,0,-1.]])
        S_plus= np.array([[0,1.,0],[0,0,1.],[0,0,0]]) * np.sqrt(2)
        S_minus=np.array([[0,0,0],[1.,0,0],[0,1.,0]]) * np.sqrt(2)

        return [identity, S_z, S_plus, S_minus]
