"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""
from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, add_inputs, add_outputs, get_order_max, apply_full_tetris
from source.Test.creatorCircuitQasmStr import CircuitCreator
from qiskit import QuantumCircuit
import unittest

# For generate the circuits
creator = CircuitCreator()
# To not generate multiple times the same circuit
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


def load_tn_circuit(is_ini_closed, is_final_closed, use_tetris, circuit_type='small'):
    global creator
    circuit = create_small_circuit()
    if circuit_type == 'medium':
        circuit = create_medium_circuit()
    tn, all_indices, depth = cir_2_tn_lbl(circuit)
    n = get_real_qubit_num(circuit)
    state = [0] * n
    if is_ini_closed:
        add_inputs(tn, state, n)
    if is_final_closed:
        add_outputs(tn, state, n)
    if use_tetris:
        tn = apply_full_tetris(tn, depth)
    return tn, all_indices, n


class TestSlicingMethods(unittest.TestCase):

    def test_slicing_method_get_max_1_small_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_small_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_small_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_small_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_small_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_small_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_small_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_small_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_small_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_small_tetris_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True)
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_small_tetris_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True)
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_small_tetris_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True)
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_medium_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_medium_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_medium_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_4_medium_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 4)
        self.assertEqual(4, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_8_medium_simple_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 8)
        self.assertEqual(8, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_medium_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_medium_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_medium_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_4_medium_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 4)
        self.assertEqual(4, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_8_medium_simple_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=False,
                                             circuit_type='medium')
        indices = get_order_max(tn, 8)
        self.assertEqual(8, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_medium_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_medium_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_medium_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_4_medium_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 4)
        self.assertEqual(4, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_8_medium_tetris_open(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=False, is_final_closed=False, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 8)
        self.assertEqual(8, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_1_medium_tetris_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 1)
        self.assertEqual(1, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_2_medium_tetris_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 2)
        self.assertEqual(2, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)

    def test_slicing_method_get_max_3_medium_tetris_close(self):
        tn, all_indices, n = load_tn_circuit(is_ini_closed=True, is_final_closed=True, use_tetris=True,
                                             circuit_type='medium')
        indices = get_order_max(tn, 3)
        self.assertEqual(3, len(indices))
        for index in indices:
            self.assertIn(index, all_indices)