from typing import List
import numpy as np

from pyquil import Program
from pyquil.gates import *
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection

##
############# YOU MUST COMMENT OUT THESE TWO LINES FOR IT TO WORK WITH THE AUTOGRADER
import subprocess
subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)

declared_already = False


# Do not change this SEED value you or your autograder score will be incorrect.
qvm = QVMConnection(random_seed=1337)



def bit_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob) * np.asarray([[0, 1], [1, 0]])
    return [noisy_I, noisy_X]


def phase_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_Z = np.sqrt(prob) * np.asarray([[1, 0], [0, -1]])
    return [noisy_I, noisy_Z]


def depolarizing_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob/3) * np.asarray([[0, 1], [1, 0]])
    noisy_Y = np.sqrt(prob/3) * np.asarray([[0, -1], [1, 0]])
    noisy_Z = np.sqrt(prob/3) * np.asarray([[1, 0], [0, -1]])
    return [noisy_I, noisy_X, noisy_Y, noisy_Z]


def bit_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):

    ### Do your encoding step here
    code_register = []  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program()  # the Program that does the encoding

    #Setup and declaration of qubits and memory
    global declared_already
    if not declared_already : 
        print(1)
        rb = pq.declare('rb', 'BIT', 2)
        declared_already = True
    
        
    x = qubit
    x1 = QubitPlaceholder()
    x2 = QubitPlaceholder()
   
    #Ancilla qubits
    a1 = QubitPlaceholder()
    a2 = QubitPlaceholder()
    #To ensure the ancillas are zero
    pq += Program(CNOT(a1, x)) 
    pq += Program(CNOT(a2, x))

    #Encode with CNOT x x1; CNOT x x2
    pq += Program(CNOT(x, x1))
    pq += Program(CNOT(x, x2))
    code_register.append(x)
    code_register.append(x1)
    code_register.append(x2)


    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here

    #Decode with CNOT x a1;CNOT x1 a1; Measure a1[0]; CNOT x1 a2; CNOT x2 a2; Measure a2[1]
    pq += Program(CNOT(x, a1))
    pq += Program(CNOT(x1, a1))
    pq += Program(CNOT(x1, a2))
    pq += Program(CNOT(x2, a2))
    pq += MEASURE(a1, rb[0])
    pq += MEASURE(a2, rb[1])
    

    #Conditional Cases:
    if rb[0] == 0 and rb[1] == 1 : pq += Program(X(x2))
    elif rb[0] == 1 and rb[1] == 0 : pq += Program(X(x))
    elif rb[0] == 1 and rb[1] == 1 : pq += Program(X(x1))

    return pq, code_register


def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    ### Do your encoding step here
    code_register = []  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program()  # the Program that does the encoding

    #Setup and declaration of qubits and memory
    rp = pq.declare('rp', 'BIT', 2)

    x = qubit
    x1 = QubitPlaceholder()
    x2 = QubitPlaceholder()
   
    #Ancilla qubits
    a1 = QubitPlaceholder()
    a2 = QubitPlaceholder()
    #To ensure the ancillas are zero
    pq += Program(CNOT(a1, x)) 
    pq += Program(CNOT(a2, x))


    #Encode with CNOT x x1; CNOT x x2
    pq += Program(CNOT(x, x1))
    pq += Program(CNOT(x, x2))
    pq += Program(H(x))
    pq += Program(H(x1))
    pq += Program(H(x2))
    code_register.append(x)
    code_register.append(x1)
    code_register.append(x2)

    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)

    ### Do your decoding and correction steps here

    #Decode with CNOT x a1;CNOT x1 a1; Measure a1[0]; CNOT x1 a2; CNOT x2 a2; Measure a2[1]
    pq += Program(H(x))
    pq += Program(H(x1))
    pq += Program(H(x2))
    pq += Program(CNOT(x, a1))
    pq += Program(CNOT(x1, a1))
    pq += Program(CNOT(x1, a2))
    pq += Program(CNOT(x2, a2))
    pq += MEASURE(a1, rp[0])
    pq += MEASURE(a2, rp[1])

    #Conditional Cases:
    if rp[0] == 0 and rp[1] == 1 : pq += Program(Z(x2))
    elif rp[0] == 1 and rp[1] == 0 : pq += Program(Z(x))
    elif rp[0] == 1 and rp[1] == 1 : pq += Program(Z(x1))

    return pq, code_register


def shor(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    # Note that in order for this code to work properly, you must build your Shor code using the phase code and
    # bit code methods above
    code_register = []

    phase_program, phase_encoded = phase_code(qubit)
    x, x1, x2 = phase_encoded #Unpack phase encoded qubits


    bit_program, bit_encoded = bit_code(x)
    bit_program1, bit_encoded1 = bit_code(x1)
    bit_program2, bit_encoded2 = bit_code(x2)

    x, a, b = bit_encoded
    code_register.append(x)
    code_register.append(a)
    code_register.append(b)
    x1, a1, b1 = bit_encoded1
    code_register.append(x1)
    code_register.append(a1)
    code_register.append(b1)
    x2, a2, b2 = bit_encoded2
    code_register.append(x2)
    code_register.append(a2)
    code_register.append(b2)

    total_program = phase_program + bit_program + bit_program1 + bit_program2

    return total_program, code_register


def run_code(error_code, noise, trials=10):
    """ Takes in an error_code function (e.g. bit_code, phase_code or shor) and runs this code on the QVM"""
    pq, code_register = error_code(QubitPlaceholder(), noise=noise)
    ro = pq.declare('ro', 'BIT', len(code_register))
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, ro)]

    return qvm.run(address_qubits(pq), trials=trials)


def simulate_code(kraus_operators, trials, error_code) -> int:
    """
    :param kraus_operators: The set of Kraus operators to apply as the noise model on the identity gate
    :param trials: The number of times to simulate the program
    :param error_code: The error code {bit_code, phase_code or shor} to use
    :return: The number of times the code did not correct back to the logical zero state for "trials" attempts
    """
    # Apply the error_code to some qubits and return back a Program pq
    score = 0
    pq, code_register = error_code(QubitPlaceholder())
    ro = pq.declare('ro', 'BIT', len(code_register))

    # THIS CODE APPLIES THE NOISE FOR YOU
    kraus_ops = kraus_operators
    noise_data = Program()
    for qq in range(3):
        noise_data.define_noisy_gate("I", [qq], kraus_ops)
    pq = noise_data + pq

    # Run the simulation trials times using the QVM and check how many times it did not work
    # return that as the score. E.g. if it always corrected back to the 0 state then it should return 0.
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, ro)]
    results = qvm.run(address_qubits(pq), trials=trials)
    for trial in results:
        if any(trial) and not all(trial) : score += 1

    return score


# score = simulate_code(bit_flip_channel(0.5), 1000, bit_code)
# print(score)

# # score = simulate_code(bit_flip_channel(0.5), 1000, phase_code)
# # print(score) #Always zero which seems wrong

score = simulate_code(bit_flip_channel(0.5), 100, shor)
print(score)


