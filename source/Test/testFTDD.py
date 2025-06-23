"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

    WARNING!!! Only have been implemented the tests for totally close circuits with GTN. This is because we
    cannot ensure that the order of the indices of tensor are in the same between the TDD representation and
    this representation. Once this is sorted, then the test will be implemented.

"""

from source.Test.creatorCircuitQasmStr import CircuitCreator
from source.TDD import equal_tolerance
from source.TDD_Q import simulate
from qiskit import QuantumCircuit
import unittest

creator = CircuitCreator()
small_circuit = None
medium_circuit = None


def create_small_circuit():
    """
        Creates a small circuit using the class CircuitCreator. Returns a QuantumCircuit object
    """
    global creator, small_circuit
    if small_circuit is None:
        small_circuit = QuantumCircuit.from_qasm_str(creator.create_small_circuit())
    return small_circuit


def create_medium_circuit():
    """
        Creates a medium circuit using the class CircuitCreator. Returns a QuantumCircuit object
    """
    global creator, medium_circuit
    if medium_circuit is None:
        medium_circuit = QuantumCircuit.from_qasm_str(creator.create_medium_circuit())
    return medium_circuit


def adapt_tdd_result(tdd):
    if len(tdd) == 2:
        return [tdd[0], tdd[1]]
    i_mid = len(tdd) // 2
    return [adapt_tdd_result(tdd[:i_mid]), adapt_tdd_result(tdd[i_mid:])]


class TestFTDD(unittest.TestCase):
    """

    """

    """
    def test_ftdd_simple_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=True, backend="FTDD")
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd)
    """

    def test_ftdd_simple_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=True, is_output_closed=False, backend="FTDD")
        tdd_adapted = adapt_tdd_result(tdd)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_close_open(), tdd_adapted))

    def test_ftdd_simple_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=True, backend="FTDD")
        tdd_adapted = adapt_tdd_result(tdd)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_open_close(), tdd_adapted))

    def test_ftdd_simple_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate(circuit, is_input_closed=False, is_output_closed=False, backend="FTDD")
        tdd_adapted = adapt_tdd_result(tdd)
        print(tdd_adapted)
        print(creator.get_small_circuit_solution_open_open())
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_open_open(), tdd_adapted))
