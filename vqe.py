from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator
from pyquil import Program
from pyquil.gates import *
from scipy.optimize import minimize

import random
import numpy as np

sim = WavefunctionSimulator(random_seed=1337)

NUM_QUBITS = 5
NUM_LAYERS = 10

def ansatz(thetas):
    """
    Returns an ansatz program so I can later run expection
    with running ansatz with some Hamiltonian
    """
    counter = 0
    p = Program()
    for layer in range(NUM_LAYERS):
        for q_idx in range(NUM_QUBITS):
            p += RX(angle=thetas[counter], qubit=q_idx)
            counter += 1
            p += RZ(angle=thetas[counter], qubit=q_idx)
            counter += 1
            if q_idx > 0:
                p += CNOT(q_idx - 1, q_idx)
    return p

def func(thetas, hamiltonian):
    return sim.expectation(ansatz(thetas), hamiltonian)

def solve_vqe(hamiltonian: PauliSum) -> float:
    # Construct a variational quantum eigensolver solution to find the lowest
    # eigenvalue of the given hamiltonian
    # pass
    thetas = np.zeros(2 * NUM_QUBITS * NUM_LAYERS)
    for i in range(len(thetas)):
        thetas[i] = random.uniform(0, np.pi)

    res = minimize(func, thetas, hamiltonian, method='Nelder-Mead')
    return res.fun