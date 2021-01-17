# -*- coding: utf-8 -*-

import numpy as np

# ==============================================================
# Calculate energy from an MPS
#
# Inputs a list where each entry in the list contains an MPS-tensor of size d x D x D, where
# the first index is the physical index, the second the leg to left and the third to the right
# First MPS tensor should be of size d x 1 x D, the last d x D x 1
#
# H2site is the hamiltonian operator, should be of size d x d x d x d and cx allows for complex values
# Returns the total energy of the MPS tensor
# ==============================================================

def compute_energy(ket, H2site, cx=0):

    def ContractHam(H2site, ket1, ket2, bra1, bra2):
    	Tens = np.tensordot(ket1, H2site, axes=[0,2])
    	Tens = np.tensordot(Tens, ket2, axes=([1,4],[1,0]))
    	Tens = np.tensordot(Tens, bra1, axes=[1,0])
    	Tens = np.tensordot(Tens, bra2, axes=([4,1],[1,0]))
    	return np.transpose(Tens, (0,2,1,3))

    def ContractBraKet(ket, bra):
    	Tens = np.tensordot(ket, bra, axes=[0,0])
    	return np.transpose(Tens, (0,2,1,3))

    N = len(ket)
    if cx == 'complex':
        bra = {}
        for i in range(N):
            bra[i] = np.conj(ket[i])
    else:
        bra = ket

    HamLeft = ContractHam(H2site, ket[0], ket[1], bra[0], bra[1])
    Norm2Left = ContractBraKet(ket[0], bra[0])
    BraKetiplus1 = ContractBraKet(ket[1], bra[1])

    for i in range(1,N-1):
        BraKeti = BraKetiplus1
        BraKetiplus1 = ContractBraKet(ket[i+1], bra[i+1])
        HamLefti = ContractHam(H2site, ket[i], ket[i+1], bra[i], bra[i+1])
        HamLeft = np.tensordot(HamLeft,  BraKetiplus1, axes=([2,3],[0,1])) \
			+ np.tensordot(Norm2Left, HamLefti, axes=([2,3],[0,1]))
        Norm2Left = np.tensordot(Norm2Left, BraKeti, axes=([2,3],[0,1]))

    Norm2Left = np.tensordot(Norm2Left, BraKetiplus1, axes=([2,3],[0,1]))

    return HamLeft[0,0,0,0] / Norm2Left[0,0,0,0]
