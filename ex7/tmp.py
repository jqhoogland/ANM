S_plus = np.array([[0, 1], [0, 0]]);
S_minus = np.array([[0, 0], [1, 0]]);

# 1-spin chain , evaluate S_plus on spin 1
H_1_spin = np.kron(S_plus, 1) # = S_plus

# 2-spin chain, S_plus on spin 1, S_minus on spin 2
H_2_spin = np.kron(S_minus, np.kron(S_plus, 1)) # = S_plus

# 3-spin chain, S_plus on spin 1, S_minus on spin 2, ID on spin 3
H_3_spin_1 = np.kron(np.eye(2), np.kron(S_minus, np.kron(S_plus, 1))) # = S_plus
# 3-spin chain, ID on spin 1, S_plus on spin 2, S_minus on spin 3
H_3_spin_1 = np.kron(S_minus, np.kron(S_plus, np.kron(np.eye(2), 1)))) # = S_plus

n_spins = 3
open_boundary_neighbors = [(i, i + 1) for i in range(n_spins)]
periodic_boundary_neighbors = open_boundary_neighbors
periodic_boundary_neighbors.push((0, n_spins))
x
def get_ij_term_1(i, j, S_i, S_j, n_spins=3):
    basis_states = 1
    res = 1 # first a scalar, then 2x2 matrix, then 4x4, then 8x8, etc.

    for idx in range(n_spins):
        if (idx == i):
            res = np.kron(S_i, res)
        elif (idx == j):
            res = np.kron(S_j, res)
        else:
            res = np.kron(np.eye(2), res)

    return res






    get_ij_term(self, i, j, S_i, S_j):
    term = 1
    idx = 0

    while idx < i:
        term = self.kron(term, self.identity)
        idx += 1

    term = self.kron(term, S_i) # for i
    idx += 1

    while idx < j:
        term = self.kron(term, self.identity)
        idx += 1

    term = self.kron(term, S_j) # for j
    idx += 1

    while idx < self.n_spins:
        term = self.kron(term, self.identity)
        idx += 1

    return term

0000                                         1000
[[ 0.75  0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.25  0.5   0.    0.    0.    0.    0.    0.    0.    0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.5  -0.25  0.    0.5   0.    0.    0.    0.    0.    0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.25  0.    0.5   0.    0.    0.    0.    0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.5   0.   -0.25  0.    0.    0.    0.5   0.    0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.5   0.   -0.75  0.5   0.    0.    0.5   0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.5  -0.25  0.    0.    0.    0.5   0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.25  0.    0.    0.    0.5  0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.5   0.    0.    0.    0.25  0.    0.    0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.5   0.    0.    0.   -0.25  0.5   0.   0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.5   0.    0.    0.5  -0.75  0.   0.5   0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.5   0.    0.    0.   -0.25 0.    0.5   0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.5   0.   0.25  0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.5  0.   -0.25  0.5   0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   0.    0.5   0.25  0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.   0.    0.    0.    0.75]]
