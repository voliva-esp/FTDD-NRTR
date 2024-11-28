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
    """
        Creates a small circuit using the class CircuitCreator. Returns a QuantumCircuit object
    """
    global creator, small_circuit
    if small_circuit is None:
        small_circuit = QuantumCircuit.from_qasm_str(creator.create_small_circuit())
    return small_circuit


class TestSimulate(unittest.TestCase):
    """
        Suite designed to testing the method 'simulate'. We use the minimum amount of steps needed to perform all
        the contraction process as the original tool intended to, and not adding any new functionality.
    """

    def test_simple_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_simple_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_simple_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_simple_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_tetris_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, use_tetris=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_tetris_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, use_tetris=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_tetris_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, use_tetris=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_tetris_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, use_tetris=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_cotengra_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, contraction_method='cot')
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_cotengra_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_cotengra_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_cotengra_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_cotengra_tetris_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, use_tetris=True, contraction_method='cot')
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_cotengra_tetris_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, use_tetris=True, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_cotengra_tetris_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, use_tetris=True, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_cotengra_tetris_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, use_tetris=True, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))
