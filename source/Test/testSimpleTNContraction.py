"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""


from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, add_inputs, add_outputs
from source.Test.creatorCircuitQasmStr import CircuitCreator
from source.TDD import Ini_TDD, equal_tolerance
from qiskit import QuantumCircuit
import numpy as np
import unittest

# For generate the circuits
creator = CircuitCreator()
# To not generate multiple times the same circuit
small_circuit = None
medium_circuit = None


def simulate_circuit(circuit, is_ini_closed, is_final_closed):
    """
        Makes a simple simulation of a circuit with the minimum steps needed with this tool.
    """
    tn, all_indices, depth = cir_2_tn_lbl(circuit)
    n = get_real_qubit_num(circuit)
    state = [0] * n
    if is_ini_closed:
        add_inputs(tn, state, n)
    if is_final_closed:
        add_outputs(tn, state, n)
    Ini_TDD(all_indices)
    path = tn.get_seq_path()
    tdd = tn.cont_TN(path, False)
    return tdd


def create_small_circuit():
    """
        Creates a small circuit using the class CircuitCreator. Returns a QuantumCircuit object
    """
    global creator, small_circuit
    if small_circuit is None:
        small_circuit = QuantumCircuit.from_qasm_str(creator.create_small_circuit())
    return small_circuit

class TestSimpleTNContraction(unittest.TestCase):
    """
        Suite designed to testing the method 'cont_TN' of the TDD class. We use the minimum amount of steps
        needed to perform all the contraction process.
    """

    def test_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit(circuit, is_ini_closed=True, is_final_closed=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit(circuit, is_ini_closed=True, is_final_closed=False)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit(circuit, is_ini_closed=False, is_final_closed=True)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit(circuit, is_ini_closed=False, is_final_closed=False)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_open_open(), tdd.to_array()))