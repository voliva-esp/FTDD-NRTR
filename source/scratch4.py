import time
from qiskit import QuantumCircuit
from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, squeezeTN, squeezeTN_ultra, add_inputs, add_outputs
from source.TN import TensorNetwork

'''
    Utility functions
'''
def PyTN_2_cTN(tn_lbl):
    # Create cTDD tensor network
    cTN = cTDD.TensorNetwork(tn_lbl.tn_type, tn_lbl.qubits_num)
    # Add tensors from PyTDD TN to cTDD TN
    for ts in tn_lbl.tensors:
        # Create C++ tensor
        data = ts.data.flatten()
        shape = ts.data.shape
        index_key = [ind.key for ind in ts.index_set]
        index_idx = [ind.idx for ind in ts.index_set]
        name = ts.name
        qubits_list = ts.qubits
        depth = ts.depth
        cTensor = cTDD.Tensor(data, list(shape), index_key, index_idx, name, qubits_list, depth)
        # Add C++ Tensor to C++ TN
        cTN.add_tensor(cTensor, False)
    return cTN

''' 
    Pick a quantum circuit for this demo 
'''
file_path = './Benchmarks/Verification/'
file_name = "qft_15"
path = None

cir = QuantumCircuit.from_qasm_file(file_path+file_name+'.qasm')

''' 
    Circuit-to-TN Transpilation 
'''
tn_lbl, all_indexs_lbl, depth = cir_2_tn_lbl(cir)
n = get_real_qubit_num(cir)

# Add initial state tensors
input_s = [0] * n
if input_s:
    add_inputs(tn_lbl, input_s, n)
    add_outputs(tn_lbl, input_s, n, depth)
print(f"Using input: {input_s}")

''' 
    Tetris-based Rank Simlification
'''
print(f"Number of tensors before tetris: {len(tn_lbl.tensors)}")
tensors_tetris = squeezeTN(tn_lbl.tensors, n, depth)
tensors_tetris = squeezeTN_ultra(tensors_tetris, n, depth)
tn_tetris = TensorNetwork(tensors_tetris, tn_lbl.tn_type, n)

path = tn_tetris.get_seq_path()

print("Starting cTDD test...")
# sys.path.append('./source/cpp/build/')
import source.cpp.build.cTDD as cTDD

print("Test FTDD, t1c1")
# cTDD Table parameters
load_factor = 1
alpha = 2
beta = alpha * load_factor

NBUCKET = int(alpha * 2**n)
INITIAL_GC_LIMIT = int(beta * 2**n)
INITIAL_GC_LUR = 0.9
ACT_NBUCKET = 32768
CCT_NBUCKET = 32768
uniqTabConfig = [INITIAL_GC_LIMIT, INITIAL_GC_LUR, NBUCKET, ACT_NBUCKET, CCT_NBUCKET]
print("Parameters set")

# Initialize cTDD
cTDD.Ini_TDD(all_indexs_lbl, uniqTabConfig, False)
print("cTDD initialized")
cTN = PyTN_2_cTN(tn_tetris)

'''
    Test FTDD
'''

# Contract TN in cTDD
t1 = time.perf_counter()
print("Contracting...")
ctdd = cTN.cont_TN(path, False)
print("Contraction done")
t2 = time.perf_counter()
dt = t2 - t1
print('FTDD contraction finished with time ', dt, 's')

# cTDD statistics
print("FTDD result # nodes: ", ctdd.node_number())
print("FTDD unique table size is: ", cTDD.get_unique_table_num())
print(cTDD.get_count())

print(f"Solution: {ctdd.to_array()}")

print('\n')
