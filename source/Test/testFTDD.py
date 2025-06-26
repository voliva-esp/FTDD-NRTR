"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""

from source.Test.creatorCircuitQasmStr import CircuitCreator
from source.TDD import equal_tolerance
from source.TDD_Q import PyTN_2_cTN, cir_2_tn_lbl, get_real_qubit_num, add_inputs
import source.cpp.build.cTDD as cTDD
from qiskit import QuantumCircuit
import unittest

creator = CircuitCreator()
small_circuit = None
medium_circuit = None
tricky_circuit = None


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


def create_tricky_circuit():
    """
        Creates a tricky circuit using the class CircuitCreator. Returns a QuantumCircuit object
    """
    global creator, tricky_circuit
    if tricky_circuit is None:
        tricky_circuit = QuantumCircuit.from_qasm_str(creator.create_tricky_circuit())
    return tricky_circuit


def adapt_tdd_result(tdd):
    if len(tdd) == 2:
        return [tdd[0], tdd[1]]
    i_mid = len(tdd) // 2
    return [adapt_tdd_result(tdd[:i_mid]), adapt_tdd_result(tdd[i_mid:])]


def make_sim(cir, uniq_config, path):
    tn, all_indices_lbl, depth = cir_2_tn_lbl(cir)
    n = get_real_qubit_num(cir)
    state = [0] * n
    add_inputs(tn, state, n)
    cTDD.Ini_TDD(all_indices_lbl, uniq_config, False)
    tn = PyTN_2_cTN(tn)
    tdd = tn.cont_TN(path, False).to_array()
    tdd_adapted = adapt_tdd_result(tdd)
    return tdd_adapted


def generate_path(n_qubits, block_gates, mid_gates):
    path = [(n_qubits, n_qubits + 1)]
    i = 1
    while i < block_gates + mid_gates - 1:
        pos = block_gates * 2 + mid_gates + n_qubits - i - 1
        path.append((n_qubits, pos))
        i += 1
    path.append((n_qubits, n_qubits + 1))
    i += 1
    while i < 2 * block_gates + mid_gates - 2:
        pos = block_gates * 2 + mid_gates + n_qubits - i - 1
        path.append((n_qubits, pos))
        i += 1
    while i < 2 * block_gates + mid_gates + n_qubits - 1:
        path.append((0, 1))
        i += 1
    return tuple(path)


class TestFTDD(unittest.TestCase):
    def test_normal_small_close_open(self):
        cir = create_small_circuit()
        n_bucket = 32000
        initial_gc_limit = 20
        initial_gc_lur = 0.9
        act_bucket = 32768
        cct_bucket = 32768
        uniqTabConfig = [initial_gc_limit, initial_gc_lur, n_bucket, act_bucket, cct_bucket]
        path = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
        result = make_sim(cir, uniqTabConfig, path)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_close_open(), result))

    def test_normal_tricky_close_open(self):
        cir = create_tricky_circuit()
        n_bucket = 32000
        initial_gc_limit = 20
        initial_gc_lur = 0.9
        act_bucket = 32768
        cct_bucket = 32768
        uniqTabConfig = [initial_gc_limit, initial_gc_lur, n_bucket, act_bucket, cct_bucket]
        path = generate_path(n_qubits=3, block_gates=6, mid_gates=4)
        result = make_sim(cir, uniqTabConfig, path)
        print(result)
        self.assertTrue(equal_tolerance(creator.get_tricky_circuit_solution_close_open(), result))

    def test_always_gc_small_close_open(self):
        cir = create_small_circuit()
        n_bucket = 32000
        initial_gc_limit = 0
        initial_gc_lur = 0
        act_bucket = 32768
        cct_bucket = 32768
        uniqTabConfig = [initial_gc_limit, initial_gc_lur, n_bucket, act_bucket, cct_bucket]
        path = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
        result = make_sim(cir, uniqTabConfig, path)
        gc = int(cTDD.get_count().split("\n")[3].split(": ")[2])
        self.assertTrue(4, gc)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_close_open(), result))

    def test_always_gc_tricky_close_open(self):
        cir = create_tricky_circuit()
        n_bucket = 32000
        initial_gc_limit = 0
        initial_gc_lur = 0
        act_bucket = 32768
        cct_bucket = 32768
        uniqTabConfig = [initial_gc_limit, initial_gc_lur, n_bucket, act_bucket, cct_bucket]
        path = generate_path(n_qubits=3, block_gates=6, mid_gates=4)
        result = make_sim(cir, uniqTabConfig, path)
        gc = int(cTDD.get_count().split("\n")[3].split(": ")[2])
        print(cTDD.get_count())
        print(gc)
        print(result)
        self.assertTrue(equal_tolerance(creator.get_tricky_circuit_solution_close_open(), result))

    def test_always_collide_small_close_open(self):
        cir = create_small_circuit()
        n_bucket = 32000
        initial_gc_limit = 0
        initial_gc_lur = 0
        act_bucket = 1
        cct_bucket = 1
        uniqTabConfig = [initial_gc_limit, initial_gc_lur, n_bucket, act_bucket, cct_bucket]
        path = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
        result = make_sim(cir, uniqTabConfig, path)
        gc = int(cTDD.get_count().split("\n")[3].split(": ")[2])
        self.assertTrue(4, gc)
        self.assertTrue(equal_tolerance(creator.get_small_circuit_solution_close_open(), result))

    def test_always_collide_tricky_close_open(self):
        cir = create_tricky_circuit()
        n_bucket = 32000
        initial_gc_limit = 0
        initial_gc_lur = 0
        act_bucket = 1
        cct_bucket = 1
        uniqTabConfig = [initial_gc_limit, initial_gc_lur, n_bucket, act_bucket, cct_bucket]
        path = generate_path(n_qubits=3, block_gates=6, mid_gates=4)
        result = make_sim(cir, uniqTabConfig, path)
        gc = int(cTDD.get_count().split("\n")[3].split(": ")[2])
        print(cTDD.get_count())
        print(gc)
        print(result)
        self.assertTrue(equal_tolerance(creator.get_tricky_circuit_solution_close_open(), result))
