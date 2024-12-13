"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""
from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, add_inputs, add_outputs, get_order_max, apply_full_tetris
from source.Test.creatorCircuitQasmStr import CircuitCreator
from qiskit import QuantumCircuit
import unittest

creator = CircuitCreator()
small_circuit = None


def create_small_circuit():
    global creator, small_circuit
    if small_circuit is None:
        small_circuit = QuantumCircuit.from_qasm_str(creator.create_small_circuit())
    return small_circuit


def load_tn_circuit(is_ini_closed, is_final_closed, use_tetris):
    global creator
    circuit = create_small_circuit()
    tn, all_indices, depth = cir_2_tn_lbl(circuit)
    n = get_real_qubit_num(circuit)
    state = [0] * n
    if is_ini_closed:
        add_inputs(tn, state, n)
    if is_final_closed:
        add_outputs(tn, state, n)
    if use_tetris:
        tn = apply_full_tetris(tn, depth)
    return tn, all_indices


class TestSlicingMethods(unittest.TestCase):

    def test_slicing_method_get_max_1_simple_open(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_simple_open(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_simple_open(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_simple_close(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_simple_close(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_simple_close(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_tetris_open(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_tetris_open(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_tetris_open(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_tetris_close(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_tetris_close(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_tetris_close(self):
        tn, all_indices = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)
