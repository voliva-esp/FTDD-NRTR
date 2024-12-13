"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""

from source.Test.creatorCircuitQasmStr import CircuitCreator
from source.TDD_Q import simulate
from qiskit import QuantumCircuit
import numpy as np
import unittest

creator = CircuitCreator()
small_circuit = None


def create_small_circuit():
    global creator, small_circuit
    if small_circuit is None:
        small_circuit = QuantumCircuit.from_qasm_str(creator.create_small_circuit())
    return small_circuit


class TestSimulateSlicing(unittest.TestCase):

    def test_slicing_max_1_simple_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, use_slicing=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_slicing_max_1_simple_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, use_slicing=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_slicing_max_1_simple_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, use_slicing=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_slicing_max_1_simple_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, use_slicing=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_slicing_max_1_tetris_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, use_tetris=True, use_slicing=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_slicing_max_1_tetris_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, use_tetris=True, use_slicing=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_slicing_max_1_tetris_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, use_tetris=True, use_slicing=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_slicing_max_1_tetris_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, use_tetris=True, use_slicing=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_slicing_max_2_simple_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, use_slicing=True, n_indices=2)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_slicing_max_2_simple_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, use_slicing=True, n_indices=2)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_slicing_max_2_simple_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, use_slicing=True, n_indices=2)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_slicing_max_2_simple_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, use_slicing=True, n_indices=2)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_slicing_max_2_tetris_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, use_tetris=True, use_slicing=True,
                       n_indices=2)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_slicing_max_2_tetris_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, use_tetris=True, use_slicing=True,
                       n_indices=2)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_slicing_max_2_tetris_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, use_tetris=True, use_slicing=True,
                       n_indices=2)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_slicing_max_2_tetris_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, use_tetris=True, use_slicing=True,
                       n_indices=2)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))
