"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""

from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, add_inputs, add_outputs, TNtoCotInput, apply_full_tetris
from source.Test.creatorCircuitQasmStr import CircuitCreator
from qiskit import QuantumCircuit
from source.TDD import Ini_TDD
import unittest

# For generate the circuits
from source.TN import TensorNetwork

creator = CircuitCreator()
# To not generate multiple times the same circuit
small_circuit = None


def generate_circuit(circuit, is_ini_closed, is_final_closed):
    """
        Generates the quantum circuit and set up all the needed variables and structures.
        Return the Tensor Network, the depth of the circuit and the number of qubits. All of this
        is needed for using the method 'TNtoCotInput' that wants to be tested
    """
    tn, all_indices, depth = cir_2_tn_lbl(circuit)
    n = get_real_qubit_num(circuit)
    state = [0] * n
    if is_ini_closed:
        add_inputs(tn, state, n)
    if is_final_closed:
        add_outputs(tn, state, n)
    Ini_TDD(all_indices)
    return tn, depth, n


def create_small_circuit():
    """
        Creates a small circuit using the class CircuitCreator. Returns a QuantumCircuit object
    """
    global creator, small_circuit
    if small_circuit is None:
        small_circuit = QuantumCircuit.from_qasm_str(creator.create_small_circuit())
    return small_circuit


class TestTNtoCotInput(unittest.TestCase):
    """
        Suite designed to testing the method 'TNtoCotInput' of the TDD class. We use the minimum amount of steps
        needed. We will say that their return is correctly if the tensors in the 'tensor_list' are correctly set
        and the open indices are correctly set.
    """

    def test_TNtoCotInput_small_simple_open_open(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=False, is_final_closed=False)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(3, len(tensor_list))
        self.assertIn(('x0', 'x0_0', 'x1', 'x1_1'), tensor_list)
        self.assertIn(('x2', 'y2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'y1', 'x0_0', 'y0'), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(6, len(open_indices))
        self.assertIn('x0', open_indices)
        self.assertIn('x1', open_indices)
        self.assertIn('x2', open_indices)
        self.assertIn('y0', open_indices)
        self.assertIn('y1', open_indices)
        self.assertIn('y2', open_indices)

    def test_TNtoCotInput_small_simple_open_close(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=False, is_final_closed=True)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(6, len(tensor_list))
        self.assertIn(('x0', 'x0_0', 'x1', 'x1_1'), tensor_list)
        self.assertIn(('x2', 'y2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'y1', 'x0_0', 'y0'), tensor_list)
        self.assertIn(('y0',), tensor_list)
        self.assertIn(('y1',), tensor_list)
        self.assertIn(('y2',), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(3, len(open_indices))
        self.assertIn('x0', open_indices)
        self.assertIn('x1', open_indices)
        self.assertIn('x2', open_indices)
        self.assertNotIn('y0', open_indices)
        self.assertNotIn('y1', open_indices)
        self.assertNotIn('y2', open_indices)

    def test_TNtoCotInput_small_simple_close_open(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=True, is_final_closed=False)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(6, len(tensor_list))
        self.assertIn(('x0', 'x0_0', 'x1', 'x1_1'), tensor_list)
        self.assertIn(('x2', 'y2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'y1', 'x0_0', 'y0'), tensor_list)
        self.assertIn(('x0',), tensor_list)
        self.assertIn(('x1',), tensor_list)
        self.assertIn(('x2',), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(3, len(open_indices))
        self.assertNotIn('x0', open_indices)
        self.assertNotIn('x1', open_indices)
        self.assertNotIn('x2', open_indices)
        self.assertIn('y0', open_indices)
        self.assertIn('y1', open_indices)
        self.assertIn('y2', open_indices)

    def test_TNtoCotInput_small_simple_close_close(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=True, is_final_closed=True)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(9, len(tensor_list))
        self.assertIn(('x0', 'x0_0', 'x1', 'x1_1'), tensor_list)
        self.assertIn(('x2', 'y2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'y1', 'x0_0', 'y0'), tensor_list)
        self.assertIn(('x0',), tensor_list)
        self.assertIn(('x1',), tensor_list)
        self.assertIn(('x2',), tensor_list)
        self.assertIn(('y0',), tensor_list)
        self.assertIn(('y1',), tensor_list)
        self.assertIn(('y2',), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(0, len(open_indices))
        self.assertNotIn('x0', open_indices)
        self.assertNotIn('x1', open_indices)
        self.assertNotIn('x2', open_indices)
        self.assertNotIn('y0', open_indices)
        self.assertNotIn('y1', open_indices)
        self.assertNotIn('y2', open_indices)

    def test_TNtoCotInput_small_tetris_open_open(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=False, is_final_closed=False)
        tn = apply_full_tetris(tn, depth)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(3, len(tensor_list))
        self.assertIn(('x0', 'x0_0', 'x1', 'x1_1'), tensor_list)
        self.assertIn(('x2', 'y2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'y1', 'x0_0', 'y0'), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(6, len(open_indices))
        self.assertIn('x0', open_indices)
        self.assertIn('x1', open_indices)
        self.assertIn('x2', open_indices)
        self.assertIn('y0', open_indices)
        self.assertIn('y1', open_indices)
        self.assertIn('y2', open_indices)

    def test_TNtoCotInput_small_tetris_open_close(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=False, is_final_closed=True)
        tn = apply_full_tetris(tn, depth)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(3, len(tensor_list))
        self.assertIn(('x0', 'x0_0', 'x1', 'x1_1'), tensor_list)
        self.assertIn(('x2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'x0_0'), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(3, len(open_indices))
        self.assertIn('x0', open_indices)
        self.assertIn('x1', open_indices)
        self.assertIn('x2', open_indices)
        self.assertNotIn('y0', open_indices)
        self.assertNotIn('y1', open_indices)
        self.assertNotIn('y2', open_indices)

    def test_TNtoCotInput_small_tetris_close_open(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=True, is_final_closed=False)
        tn = apply_full_tetris(tn, depth)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(3, len(tensor_list))
        self.assertIn(('x0_0', 'x1_1'), tensor_list)
        self.assertIn(('y2', 'x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'y1', 'x0_0', 'y0'), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(3, len(open_indices))
        self.assertNotIn('x0', open_indices)
        self.assertNotIn('x1', open_indices)
        self.assertNotIn('x2', open_indices)
        self.assertIn('y0', open_indices)
        self.assertIn('y1', open_indices)
        self.assertIn('y2', open_indices)

    def test_TNtoCotInput_small_tetris_close_close(self):
        global creator
        circuit = create_small_circuit()
        tn, depth, n = generate_circuit(circuit, is_ini_closed=True, is_final_closed=True)
        tn = apply_full_tetris(tn, depth)
        tensor_list, open_indices, size_dict, arrays, oe_input = TNtoCotInput(tn, n)
        """ Tests if the tensors are correctly set """
        self.assertEqual(3, len(tensor_list))
        self.assertIn(('x0_0', 'x1_1'), tensor_list)
        self.assertIn(('x1_1', 'x1_2'), tensor_list)
        self.assertIn(('x1_2', 'x0_0'), tensor_list)
        """ Tests if the open indices are correctly set """
        self.assertEqual(0, len(open_indices))
        self.assertNotIn('x0', open_indices)
        self.assertNotIn('x1', open_indices)
        self.assertNotIn('x2', open_indices)
        self.assertNotIn('y0', open_indices)
        self.assertNotIn('y1', open_indices)
        self.assertNotIn('y2', open_indices)
