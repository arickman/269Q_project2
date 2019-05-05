from pyquil.paulis import PauliSum
from pyquil.api import WavefunctionSimulator


sim = WavefunctionSimulator(random_seed=1337)


def solve_vqe(hamiltonian: PauliSum) -> float:
    # Construct a variational quantum eigensolver solution to find the lowest
    # eigenvalue of the given hamiltonian
    pass
