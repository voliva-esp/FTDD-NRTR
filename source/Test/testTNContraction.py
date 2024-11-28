"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""


from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, add_inputs, add_outputs, apply_full_tetris, TNtoCotInput
from source.Test.creatorCircuitQasmStr import CircuitCreator
from qiskit import QuantumCircuit
from source.TDD import Ini_TDD
import cotengra as ctg
import numpy as np
import unittest

creator = CircuitCreator()
small_circuit = None


def get_path(tn, n, contraction_method):
    """
        Designed for using different methods for calculate the contraction path. If no valid method selected, then
        using the sequential method as default.
        tn -> The tensor network we want to calculate the path
        n  -> Number of qubits of the tensor network
        contraction_method -> Literal (str) that contains the desired contraction method we want to use
    """
    path = None
    if contraction_method == 'cot':
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        opt = ctg.HyperOptimizer(
            minimize=f'combo-{56}',
            max_repeats=512,
            max_time=5
        )
        tree = opt.search(tensor_list, open_indices, size_dict)
        path = tree.get_path()
    else:
        path = tn.get_seq_path()
    return path


def put_states(tn, n, is_input_closed, is_output_closed):
    """
        This method is used to close the input and output states if needed. Always with the state 0.
    """
    state = [0] * n
    if is_input_closed:
        add_inputs(tn, state, n)
    if is_output_closed:
        add_outputs(tn, state, n)


def simulate_circuit_before(circuit, is_input_closed, is_output_closed, contraction_method='seq'):
    """
        Simulates a circuit using the tetris optimization. In this method, we apply tetris BEFORE we close
        the input and output states (if needed).
    """
    tn, all_indices, depth = cir_2_tn_lbl(circuit)
    n = get_real_qubit_num(circuit)
    tn = apply_full_tetris(tn, depth)
    put_states(tn, n, is_input_closed, is_output_closed)
    Ini_TDD(all_indices)
    path = get_path(tn, n, contraction_method)
    tdd = tn.cont_TN(path, False)
    return tdd


def simulate_circuit_after(circuit, is_input_closed, is_output_closed, contraction_method='seq'):
    """
        Simulates a circuit using the tetris optimization. In this method, we apply tetris AFTER we close
        the input and output states (if needed).
    """
    tn, all_indices, depth = cir_2_tn_lbl(circuit)
    n = get_real_qubit_num(circuit)
    put_states(tn, n, is_input_closed, is_output_closed)
    tn = apply_full_tetris(tn, depth)
    Ini_TDD(all_indices)
    path = get_path(tn, n, contraction_method)
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


class TestTNContraction(unittest.TestCase):

    def test_before_tetris_sequential_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=True, is_output_closed=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_before_tetris_sequential_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=True, is_output_closed=False)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_before_tetris_sequential_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=False, is_output_closed=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_before_tetris_sequential_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=False, is_output_closed=False)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_before_tetris_cotengra_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=True, is_output_closed=True, contraction_method='cot')
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_before_tetris_cotengra_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=True, is_output_closed=False, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_before_tetris_cotengra_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=False, is_output_closed=True, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_before_tetris_cotengra_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_before(circuit, is_input_closed=False, is_output_closed=False, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_after_tetris_sequential_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=True, is_output_closed=True)
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_after_tetris_sequential_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=True, is_output_closed=False)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_after_tetris_sequential_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=False, is_output_closed=True)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_after_tetris_sequential_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=False, is_output_closed=False)
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))

    def test_after_tetris_cotengra_small_circuit_close_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=True, is_output_closed=True, contraction_method='cot')
        self.assertEqual(creator.get_small_circuit_solution_close_close(), tdd.to_array())

    def test_after_tetris_cotengra_small_circuit_close_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=True, is_output_closed=False, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_close_open(), tdd.to_array()))

    def test_after_tetris_cotengra_small_circuit_open_close(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=False, is_output_closed=True, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_close(), tdd.to_array()))

    def test_after_tetris_cotengra_small_circuit_open_open(self):
        global creator
        circuit = create_small_circuit()
        tdd = simulate_circuit_after(circuit, is_input_closed=False, is_output_closed=False, contraction_method='cot')
        self.assertTrue(np.array_equal(creator.get_small_circuit_solution_open_open(), tdd.to_array()))
