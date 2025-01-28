"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""

from source.Test.creatorCircuitQasmStr import CircuitCreator
from qiskit import QuantumCircuit
from source.TN import TensorNetwork
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


class TestContractingMethods(unittest.TestCase):
    """
        Suite designed for testing all the contracting methods implemented within the TensorNetwork class. It
        only tests that the path given by the desired contracting method it is the expected one, and do not
        make real contractions, so it is no need to create an actual tensor network. So be aware that the
        TensorNetworks created for the tests are not ones that can be actually contracted.
    """

    def test_seq_simple(self):
        tensors = [0, 1]
        tn = TensorNetwork(tensors)
        self.assertTrue(((0, 1),) == tn.get_seq_path())

    def test_seq_4(self):
        n = 4
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 2), (0, 1))
        self.assertTrue(expected_solution == tn.get_seq_path())

    def test_pair_simple(self):
        tensors = [0, 1]
        tn = TensorNetwork(tensors)
        self.assertTrue(((0, 1),) == tn.get_pairing_path())

    def test_pair_4(self):
        n = 4
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (0, 1))
        self.assertTrue(expected_solution == tn.get_pairing_path())

    def test_pair_5(self):
        n = 5
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (0, 2), (0, 1))
        self.assertTrue(expected_solution == tn.get_pairing_path())

    def test_pair_4_1_0(self):
        n = 4
        d_ini = 1
        d_fin = 0
        tn = TensorNetwork()
        expected_solution = ((1, 2), (1, 2), (1, 2))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_pair_5_1_0(self):
        n = 5
        d_ini = 1
        d_fin = 0
        tn = TensorNetwork()
        expected_solution = ((1, 2), (1, 2), (1, 3), (1, 2))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_pair_4_0_1(self):
        n = 4
        d_ini = 0
        d_fin = 1
        tn = TensorNetwork()
        expected_solution = ((0, 1), (0, 1), (1, 2))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_pair_5_0_1(self):
        n = 5
        d_ini = 0
        d_fin = 1
        tn = TensorNetwork()
        expected_solution = ((0, 1), (0, 1), (0, 3), (1, 2))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_pair_4_1_1(self):
        n = 4
        d_ini = 1
        d_fin = 1
        tn = TensorNetwork()
        expected_solution = ((1, 2), (1, 2), (2, 3))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_pair_5_1_1(self):
        n = 5
        d_ini = 1
        d_fin = 1
        tn = TensorNetwork()
        expected_solution = ((1, 2), (1, 2), (1, 4), (2, 3))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_pair_7_2_2(self):
        n = 7
        d_ini = 2
        d_fin = 2
        tn = TensorNetwork()
        expected_solution = ((2, 3), (2, 3), (2, 3), (2, 7), (4, 5), (4, 5))
        path = tn.get_pairing_path(n=n, d_ini=d_ini, d_fin=d_fin)
        self.assertTrue(expected_solution == path)

    def test_spair_simple(self):
        n = 2
        tensors_to_slice = []
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), )
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_noSlicing_4(self):
        n = 4
        tensors_to_slice = []
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_noSlicing_5(self):
        n = 5
        tensors_to_slice = []
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (0, 2), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_slicing_middle_5_1(self):
        n = 5
        tensors_to_slice = [2]
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (1, 2), (0, 1), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_slicing_middle_9_1(self):
        n = 9
        tensors_to_slice = [4]
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (5, 6), (1, 2), (1, 2), (2, 3), (0, 1), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_slicing_left_7_1(self):
        n = 7
        tensors_to_slice = [2]
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (1, 2), (1, 2), (2, 3), (0, 1), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_slicing_right_7_1(self):
        n = 7
        tensors_to_slice = [4]
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (3, 4), (1, 2), (0, 1), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)

    def test_spair_slicing_equal_14_2(self):
        n = 14
        tensors_to_slice = [4, 9]
        tensors = [i for i in range(n)]
        tn = TensorNetwork(tensors)
        expected_solution = ((0, 1), (0, 1), (10, 11), (1, 2), (1, 2), (7, 8), (2, 3), (2, 3), (4, 5), (0, 1), (0, 1),
                             (0, 2), (0, 1))
        path = tn.get_smart_pairing_path(tensors_to_slice=tensors_to_slice)
        print(path)
        self.assertTrue(expected_solution == path)


